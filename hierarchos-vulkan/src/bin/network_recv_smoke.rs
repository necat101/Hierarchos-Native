//! Smoke test for the in-crate `distributed::RdmaLink` receive path.
//!
//! Runs as the RDMA *listener*; expects the R0 spike client (or another sender)
//! to connect and push `count` messages of `size` bytes. Registers a buffer,
//! posts receives, polls completions, and verifies the byte pattern.
//!
//! Usage: network_recv_smoke <device> <control-port> <size> <count>

use std::net::SocketAddr;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let device = args.get(1).expect("device").clone();
    let port: u16 = args.get(2).expect("port").parse().unwrap();
    let size: usize = args.get(3).expect("size").parse().unwrap();
    let count: u64 = args.get(4).expect("count").parse().unwrap();

    let addr: SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();
    println!("recv smoke: device={device} listen={addr} size={size} count={count}");

    let link = hierarchos_vulkan::distributed::RdmaLink::connect(
        &device,
        addr,
        true, // is_listener
    )
    .expect("rdma connect");
    println!("  QP ready (RTS)");

    let mut buf: Vec<u8> = vec![0u8; size];
    let mr = link.register(&mut buf).expect("register");
    link.post_receives(&mr, count).expect("post receives");
    println!("  posted {count} receives");

    let start = Instant::now();
    let mut done = 0u64;
    while done < count {
        let n = link.poll_recv(64).expect("poll");
        done += n as u64;
    }
    let elapsed = start.elapsed();

    // verify pattern (i % 251) across the buffer
    let bad = buf
        .iter()
        .enumerate()
        .take(4096)
        .filter(|(i, b)| **b != (*i % 251) as u8)
        .count();
    if bad > 0 {
        panic!("verification failed: {bad} bad bytes");
    }
    let bytes = count as f64 * size as f64;
    println!(
        "RECV-SMOKE OK: {:.3} GB in {:?} -> {:.2} GB/s (verified)",
        bytes / 1e9,
        elapsed,
        bytes / elapsed.as_secs_f64() / 1e9
    );
}
