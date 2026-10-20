//! Cross-machine training transports. TCP is available on Windows and Linux;
//! Linux libibverbs is an explicit `rdma` feature. `tcp` is a transport selector,
//! never a claim of RDMA, kernel bypass, or GPU-direct networking.
use crate::VulkanGradientTransportBackend;
use anyhow::{bail, Result};
use std::net::{SocketAddr, TcpListener, TcpStream};
#[cfg(all(feature = "rdma", target_os = "linux"))]
mod rdma;
mod tcp;
#[cfg(all(feature = "rdma", target_os = "linux"))]
pub use rdma::RdmaLink;

pub struct NetworkGradientSource {
    inner: Source,
}
enum Source {
    Tcp(tcp::Channel),
    #[cfg(all(feature = "rdma", target_os = "linux"))]
    Rdma(rdma::NetworkGradientSource),
}
pub struct NetworkSendChannel {
    inner: Sink,
}
enum Sink {
    Tcp(tcp::Channel),
    #[cfg(all(feature = "rdma", target_os = "linux"))]
    Rdma(rdma::NetworkSendChannel),
}

pub fn validate_transport(device: &str) -> Result<()> {
    if device == "tcp" {
        return Ok(());
    }
    #[cfg(all(feature = "rdma", target_os = "linux"))]
    if !device.is_empty() {
        return Ok(());
    }
    bail!("RDMA device {device:?} requires Linux and --features rdma; use --network-transport tcp on both peers")
}

impl NetworkGradientSource {
    pub fn connect(device: &str, addr: SocketAddr, chunk_len: usize, rings: usize) -> Result<Self> {
        Ok(Self::connect_with_control(device, addr, chunk_len, rings)?.0)
    }
    pub fn connect_with_control(
        device: &str,
        addr: SocketAddr,
        chunk_len: usize,
        rings: usize,
    ) -> Result<(Self, TcpStream)> {
        validate_transport(device)?;
        Self::connect_with_bound_listener(device, TcpListener::bind(addr)?, chunk_len, rings)
    }
    pub fn connect_with_bound_listener(
        device: &str,
        listener: TcpListener,
        chunk_len: usize,
        rings: usize,
    ) -> Result<(Self, TcpStream)> {
        validate_transport(device)?;
        anyhow::ensure!(rings > 0, "ring_count must be positive");
        if device == "tcp" {
            let (data, control) = tcp::Channel::accept(listener, chunk_len)?;
            return Ok((
                Self {
                    inner: Source::Tcp(data),
                },
                control,
            ));
        }
        #[cfg(all(feature = "rdma", target_os = "linux"))]
        {
            let (data, control) = rdma::NetworkGradientSource::connect_with_bound_listener(
                device, listener, chunk_len, rings,
            )?;
            return Ok((
                Self {
                    inner: Source::Rdma(data),
                },
                control,
            ));
        }
        #[cfg(not(all(feature = "rdma", target_os = "linux")))]
        {
            bail!("RDMA is unavailable")
        }
    }
    pub fn backend(&self) -> VulkanGradientTransportBackend {
        match &self.inner {
            Source::Tcp(_) => VulkanGradientTransportBackend::NetworkTcp,
            #[cfg(all(feature = "rdma", target_os = "linux"))]
            Source::Rdma(_) => VulkanGradientTransportBackend::NetworkRdma,
        }
    }
    pub fn receive_bytes_into(&mut self, out: &mut [u8]) -> Result<usize> {
        match &mut self.inner {
            Source::Tcp(channel) => channel.receive(out),
            #[cfg(all(feature = "rdma", target_os = "linux"))]
            Source::Rdma(channel) => channel.receive_bytes_into(out),
        }
    }
    pub fn receive_into(&mut self, out: &mut [f32]) -> Result<usize> {
        let bytes = self.receive_bytes_into(bytemuck::cast_slice_mut(out))?;
        anyhow::ensure!(bytes % 4 == 0, "received a non-FP32 chunk");
        Ok(bytes / 4)
    }
}
impl NetworkSendChannel {
    pub fn connect(
        device: &str,
        addr: SocketAddr,
        listener: bool,
        chunk_len: usize,
        rings: usize,
    ) -> Result<Self> {
        Ok(Self::connect_with_control(device, addr, listener, chunk_len, rings)?.0)
    }
    pub fn connect_with_control(
        device: &str,
        addr: SocketAddr,
        listener: bool,
        chunk_len: usize,
        rings: usize,
    ) -> Result<(Self, TcpStream)> {
        validate_transport(device)?;
        anyhow::ensure!(rings > 0, "ring_count must be positive");
        if device == "tcp" {
            let (data, control) = if listener {
                tcp::Channel::accept(TcpListener::bind(addr)?, chunk_len)?
            } else {
                tcp::Channel::connect(addr, chunk_len)?
            };
            return Ok((
                Self {
                    inner: Sink::Tcp(data),
                },
                control,
            ));
        }
        #[cfg(all(feature = "rdma", target_os = "linux"))]
        {
            let (data, control) = rdma::NetworkSendChannel::connect_with_control(
                device, addr, listener, chunk_len, rings,
            )?;
            return Ok((
                Self {
                    inner: Sink::Rdma(data),
                },
                control,
            ));
        }
        #[cfg(not(all(feature = "rdma", target_os = "linux")))]
        {
            bail!("RDMA is unavailable")
        }
    }
    pub fn backend(&self) -> VulkanGradientTransportBackend {
        match &self.inner {
            Sink::Tcp(_) => VulkanGradientTransportBackend::NetworkTcp,
            #[cfg(all(feature = "rdma", target_os = "linux"))]
            Sink::Rdma(_) => VulkanGradientTransportBackend::NetworkRdma,
        }
    }
    pub fn send_chunk(&mut self, values: &[f32]) -> Result<()> {
        match &mut self.inner {
            Sink::Tcp(channel) => channel.send(bytemuck::cast_slice(values)),
            #[cfg(all(feature = "rdma", target_os = "linux"))]
            Sink::Rdma(channel) => channel.send_chunk(values),
        }
    }
    pub fn flush(&mut self) -> Result<()> {
        match &mut self.inner {
            Sink::Tcp(channel) => channel.flush(),
            #[cfg(all(feature = "rdma", target_os = "linux"))]
            Sink::Rdma(channel) => channel.flush(),
        }
    }
    /// Receive concurrently with a blocking send so two TCP peers cannot fill
    /// both socket buffers and deadlock on chunks larger than the OS buffers.
    pub fn exchange_chunk(
        &mut self,
        receiver: &mut NetworkGradientSource,
        values: &[f32],
        output: &mut [f32],
    ) -> Result<usize> {
        std::thread::scope(|scope| {
            let sending = scope.spawn(|| self.send_chunk(values));
            let received = receiver.receive_into(output);
            sending
                .join()
                .map_err(|_| anyhow::anyhow!("network sender thread panicked"))??;
            received
        })
    }
}
