use anyhow::{ensure, Context, Result};
use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MAGIC: &[u8; 16] = b"HIERARCHOS-TCP01";
const MAX_CHUNK_BYTES: usize = 64 * 1024 * 1024;
static SESSION: AtomicU64 = AtomicU64::new(1);

pub(super) struct Channel {
    stream: TcpStream,
    capacity: usize,
}

fn timeout() -> Result<Duration> {
    let seconds = std::env::var("HIERARCHOS_NETWORK_TIMEOUT_SECONDS")
        .unwrap_or_else(|_| "300".into())
        .parse::<u64>()
        .context("HIERARCHOS_NETWORK_TIMEOUT_SECONDS must be a positive integer")?;
    ensure!(seconds > 0, "network timeout must be positive");
    Ok(Duration::from_secs(seconds))
}
fn capacity(values: usize) -> Result<usize> {
    let bytes = values
        .checked_mul(4)
        .context("network chunk size overflow")?;
    ensure!(
        bytes > 0 && bytes <= MAX_CHUNK_BYTES,
        "network chunk must be 1..=16777216 FP32 values"
    );
    Ok(bytes)
}
fn configure(stream: TcpStream) -> Result<TcpStream> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(timeout()?))?;
    stream.set_write_timeout(Some(timeout()?))?;
    Ok(stream)
}
fn accept(listener: &TcpListener, deadline: Instant) -> Result<TcpStream> {
    loop {
        match listener.accept() {
            Ok((socket, _)) => return configure(socket),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                ensure!(Instant::now() < deadline, "network accept timed out");
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(error.into()),
        }
    }
}
fn connect(addr: SocketAddr, deadline: Instant) -> Result<TcpStream> {
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        ensure!(!remaining.is_zero(), "network connect timed out to {addr}");
        match TcpStream::connect_timeout(&addr, remaining.min(Duration::from_secs(2))) {
            Ok(socket) => return configure(socket),
            Err(error) => {
                if Instant::now() >= deadline {
                    return Err(error.into());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}
impl Channel {
    pub(super) fn accept(listener: TcpListener, values: usize) -> Result<(Self, TcpStream)> {
        let capacity = capacity(values)?;
        let deadline = Instant::now() + timeout()?;
        listener.set_nonblocking(true)?;
        let mut control = accept(&listener, deadline)?;
        let mut hello = [0u8; 40];
        control.read_exact(&mut hello)?;
        ensure!(&hello[..16] == MAGIC, "unknown network transport/version");
        ensure!(
            u64::from_le_bytes(hello[16..24].try_into()?) == capacity as u64,
            "peers disagree on network chunk capacity"
        );
        // The tensor stream is a separate socket: disposable BufReaders on the
        // upstream control protocol must never prefetch tensor bytes.
        let mut data = accept(&listener, deadline)?;
        let mut session = [0u8; 16];
        data.read_exact(&mut session)?;
        ensure!(
            session == hello[24..40],
            "tensor socket belongs to a different session"
        );
        control.write_all(MAGIC)?;
        Ok((
            Self {
                stream: data,
                capacity,
            },
            control,
        ))
    }
    pub(super) fn connect(addr: SocketAddr, values: usize) -> Result<(Self, TcpStream)> {
        let capacity = capacity(values)?;
        let deadline = Instant::now() + timeout()?;
        let mut control = connect(addr, deadline)?;
        let mut session = [0u8; 16];
        session[..8].copy_from_slice(
            &(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64).to_le_bytes(),
        );
        session[8..].copy_from_slice(
            &(SESSION.fetch_add(1, Ordering::Relaxed) ^ u64::from(std::process::id()))
                .to_le_bytes(),
        );
        control.write_all(MAGIC)?;
        control.write_all(&(capacity as u64).to_le_bytes())?;
        control.write_all(&session)?;
        let mut data = connect(addr, deadline)?;
        data.write_all(&session)?;
        let mut acknowledgement = [0u8; 16];
        control.read_exact(&mut acknowledgement)?;
        ensure!(&acknowledgement == MAGIC, "network handshake rejected");
        Ok((
            Self {
                stream: data,
                capacity,
            },
            control,
        ))
    }
    pub(super) fn send(&mut self, bytes: &[u8]) -> Result<()> {
        ensure!(
            !bytes.is_empty() && bytes.len() % 4 == 0 && bytes.len() <= self.capacity,
            "network chunk size exceeds capacity or is not FP32 aligned"
        );
        self.stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
        self.stream.write_all(bytes)?;
        Ok(())
    }
    pub(super) fn receive(&mut self, out: &mut [u8]) -> Result<usize> {
        let mut header = [0u8; 4];
        self.stream
            .read_exact(&mut header)
            .context("reading network chunk header")?;
        let len = u32::from_le_bytes(header) as usize;
        ensure!(
            len > 0 && len % 4 == 0 && len <= self.capacity && len <= out.len(),
            "invalid network chunk length {len} (capacity {}, destination {})",
            self.capacity,
            out.len()
        );
        self.stream
            .read_exact(&mut out[..len])
            .context("reading network tensor payload")?;
        Ok(len)
    }
    pub(super) fn flush(&mut self) -> Result<()> {
        Ok(self.stream.flush()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{NetworkGradientSource, NetworkSendChannel};

    #[test]
    fn tcp_preserves_fp32_bits_tails_and_control_messages() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let receiver = std::thread::spawn(move || -> Result<Vec<u32>> {
            let (mut source, mut control) =
                NetworkGradientSource::connect_with_bound_listener("tcp", listener, 4, 2)?;
            assert_eq!(
                source.backend(),
                crate::VulkanGradientTransportBackend::NetworkTcp
            );
            let mut line = String::new();
            std::io::BufRead::read_line(&mut std::io::BufReader::new(&mut control), &mut line)?;
            assert_eq!(line, "metadata\n");
            let mut values = [0.0; 4];
            assert_eq!(source.receive_into(&mut values)?, 4);
            let mut bits = values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(source.receive_into(&mut values)?, 1);
            bits.push(values[0].to_bits());
            control.write_all(b"done\n")?;
            Ok(bits)
        });
        let (mut sender, mut control) =
            NetworkSendChannel::connect_with_control("tcp", address, false, 4, 2)?;
        let expected = [0u32, 0x80000000, 0x3f800001, 0x7fc00123, 0xff800000];
        control.write_all(b"metadata\n")?;
        sender.send_chunk(
            &expected[..4]
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect::<Vec<_>>(),
        )?;
        sender.send_chunk(&[f32::from_bits(expected[4])])?;
        sender.flush()?;
        let mut reply = [0; 5];
        control.read_exact(&mut reply)?;
        assert_eq!(&reply, b"done\n");
        assert_eq!(receiver.join().unwrap()?, expected);
        Ok(())
    }

    #[test]
    fn tcp_rejects_invalid_and_truncated_frames() -> Result<()> {
        for (length, payload) in [(0u32, vec![]), (3, vec![]), (20, vec![]), (4, vec![1, 2])] {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let address = listener.local_addr()?;
            let writer = std::thread::spawn(move || {
                let mut stream = TcpStream::connect(address).unwrap();
                stream.write_all(&length.to_le_bytes()).unwrap();
                stream.write_all(&payload).unwrap();
            });
            let mut channel = Channel {
                stream: listener.accept()?.0,
                capacity: 16,
            };
            assert!(channel.receive(&mut [0u8; 16]).is_err());
            writer.join().unwrap();
        }
        Ok(())
    }

    #[test]
    fn tcp_rejects_invalid_capacity_before_connecting() {
        assert!(capacity(0).is_err());
        assert!(capacity(usize::MAX).is_err());
        assert!(capacity(MAX_CHUNK_BYTES / 4 + 1).is_err());
    }

    #[test]
    fn tcp_simultaneous_large_exchange_makes_progress() -> Result<()> {
        let left = TcpListener::bind("127.0.0.1:0")?;
        let right = TcpListener::bind("127.0.0.1:0")?;
        let left_addr = left.local_addr()?;
        let right_addr = right.local_addr()?;
        let count = 1024 * 1024;
        let peer = std::thread::spawn(move || -> Result<()> {
            let (mut source, _control) =
                NetworkGradientSource::connect_with_bound_listener("tcp", right, count, 2)?;
            let mut sink = NetworkSendChannel::connect("tcp", left_addr, false, count, 2)?;
            let sent = vec![-0.0; count];
            let mut received = vec![0.0; count];
            assert_eq!(
                sink.exchange_chunk(&mut source, &sent, &mut received)?,
                count
            );
            assert!(received.iter().all(|v| v.to_bits() == 1.0f32.to_bits()));
            Ok(())
        });
        let mut sink = NetworkSendChannel::connect("tcp", right_addr, false, count, 2)?;
        let (mut source, _control) =
            NetworkGradientSource::connect_with_bound_listener("tcp", left, count, 2)?;
        let mut received = vec![0.0; count];
        assert_eq!(
            sink.exchange_chunk(&mut source, &vec![1.0; count], &mut received)?,
            count
        );
        assert!(received.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()));
        peer.join().unwrap()?;
        Ok(())
    }
}
