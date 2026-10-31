//! Network RDMA transport for cross-machine gradient exchange (Phase R1).
//!
//! This module brings in self-generated `ibverbs` bindings plus a C shim for the
//! static-inline verbs (`ibv_post_send`/`ibv_post_recv`/`ibv_poll_cq`) and opaque
//! field accessors. It is the foundation for the `NetworkRdma` gradient transport
//! backend: a replica rank on a remote machine computes gradient shards and streams
//! them to the primary rank over RoCE; the primary receives and reduces them.
//!
//! Design notes:
//! - Public-header bindings generate full public ABI structs; implementation
//!   structs (ibv_qp/ibv_context/ibv_mr/ibv_cq) are opaque — the shim reaches in.
//! - Endpoint exchange (QP number + 16-byte GID) is newline-delimited JSON over a
//!   TCP control channel, with explicit flush (serde_json::to_writer does not
//!   flush a TcpStream promptly and deadlocks a full-duplex handshake).
//! - Senders bound outstanding work requests to the send-queue depth (ENOSPC guard).

mod ffi {
    #![allow(
        non_snake_case,
        non_camel_case_types,
        dead_code,
        non_upper_case_globals,
        clippy::all
    )]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
use ffi::ibv_qp_state::{IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS};
use ffi::ibv_qp_type::IBV_QPT_RC;
use ffi::ibv_wc_status::IBV_WC_SUCCESS;
use ffi::ibv_wr_opcode::IBV_WR_SEND;
use ffi::*;

extern "C" {
    fn shim_poll_cq(cq: *mut ibv_cq, num_entries: i32, wc: *mut ibv_wc) -> i32;
    fn shim_post_send(
        qp: *mut ibv_qp,
        wr: *const ibv_send_wr,
        bad_wr: *mut *mut ibv_send_wr,
    ) -> i32;
    fn shim_post_recv(
        qp: *mut ibv_qp,
        wr: *const ibv_recv_wr,
        bad_wr: *mut *mut ibv_recv_wr,
    ) -> i32;
    fn shim_qp_num(qp: *mut ibv_qp) -> u32;
    fn shim_mr_lkey(mr: *mut ibv_mr) -> u32;
    fn shim_mr_rkey(mr: *mut ibv_mr) -> u32;
}

use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

/// RoCEv2 GID index (verified via perftest; commonly 3 for RoCEv2).
pub const DEFAULT_GID_INDEX: u32 = 3;
const PORT: u8 = 1;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Endpoint {
    pub qpn: u32,
    pub gid: [u8; 16],
}

impl Endpoint {
    /// Newline-delimited JSON (gid as a [u8;16] array) — matches the R0 spike
    /// framing so the two are interoperable.
    fn to_json(&self) -> String {
        let mut s = serde_json::to_string(self).expect("serialize endpoint");
        s.push('\n');
        s
    }

    fn from_line(line: &str) -> anyhow::Result<Self> {
        serde_json::from_str(line.trim()).map_err(|e| anyhow::anyhow!("parse endpoint: {e}"))
    }
}

/// A connected RDMA link: one QP between this host and a peer, plus a CQ.
pub struct RdmaLink {
    ctx: *mut ibv_context,
    pd: *mut ibv_pd,
    send_cq: *mut ibv_cq,
    recv_cq: *mut ibv_cq,
    qp: *mut ibv_qp,
    #[allow(dead_code)]
    device_name: String,
}

unsafe impl Send for RdmaLink {}

impl RdmaLink {
    /// Open a device and create a QP, then connect it to `peer` (endpoint
    /// exchange over TCP at `control_addr`). `is_listener` side accepts; the
    /// other side connects.
    pub fn connect(
        device_name: &str,
        control_addr: SocketAddr,
        is_listener: bool,
    ) -> anyhow::Result<Self> {
        let (link, _sock) = Self::connect_with_control(device_name, control_addr, is_listener)?;
        Ok(link)
    }

    /// Same as [`connect`], but keeps the TCP control socket open after the
    /// endpoint exchange and returns it. Callers use it for per-message
    /// metadata (gradient shard headers, weight-state skeletons).
    pub fn connect_with_control(
        device_name: &str,
        control_addr: SocketAddr,
        is_listener: bool,
    ) -> anyhow::Result<(Self, TcpStream)> {
        if is_listener {
            let listener = TcpListener::bind(control_addr)?;
            return Self::connect_with_bound_listener(device_name, listener);
        }
        let sock = Self::connect_retry(control_addr, 90)?;
        Self::connect_from_socket(device_name, sock, false)
    }

    /// TCP connect with retries: peers may still be doing one-time Vulkan
    /// setup (~10 min) before they reach the RDMA control bind.
    fn connect_retry(addr: SocketAddr, attempts: u32) -> anyhow::Result<TcpStream> {
        let mut last_err: Option<std::io::Error> = None;
        for i in 0..attempts {
            match TcpStream::connect(addr) {
                Ok(sock) => {
                    if i > 0 {
                        eprintln!("  [rdma] control connect succeeded on attempt {}", i + 1);
                    }
                    return Ok(sock);
                }
                Err(e) => {
                    if i % 15 == 0 {
                        eprintln!(
                            "  [rdma] waiting for peer control at {addr} (attempt {}/{}: {e})",
                            i + 1,
                            attempts
                        );
                    }
                    last_err = Some(e);
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
        Err(anyhow::anyhow!(
            "connect to {addr} failed after {attempts} attempts: {}",
            last_err.map(|e| e.to_string()).unwrap_or_default()
        ))
    }

    /// Listener variant using an already-bound control listener. Lets the
    /// caller complete other setup (e.g. signal a peer) before accepting.
    pub fn connect_with_bound_listener(
        device_name: &str,
        listener: TcpListener,
    ) -> anyhow::Result<(Self, TcpStream)> {
        let (mut sock, addr) = listener.accept()?;
        eprintln!("  [rdma] control accepted from {addr}");
        Self::connect_from_socket(device_name, sock, true)
    }

    fn connect_from_socket(
        device_name: &str,
        mut sock: TcpStream,
        wrote_first: bool,
    ) -> anyhow::Result<(Self, TcpStream)> {
        let ctx = Self::open_device(device_name)?;
        let pd = unsafe { ibv_alloc_pd(ctx) };
        anyhow::ensure!(!pd.is_null(), "ibv_alloc_pd");
        let send_cq =
            unsafe { ibv_create_cq(ctx, 4096, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        let recv_cq =
            unsafe { ibv_create_cq(ctx, 4096, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        anyhow::ensure!(!send_cq.is_null() && !recv_cq.is_null(), "ibv_create_cq");
        let qp = Self::make_qp(ctx, send_cq, recv_cq, pd)?;
        let qpn = unsafe { shim_qp_num(qp) };
        let gid = Self::query_gid(ctx)?;
        let local = Endpoint { qpn, gid };

        // Endpoint exchange over TCP. The listener side writes first.
        let remote = if wrote_first {
            sock.write_all(local.to_json().as_bytes())?;
            sock.flush()?;
            let mut line = String::new();
            BufReader::new(&mut sock).read_line(&mut line)?;
            Endpoint::from_line(&line)?
        } else {
            let mut line = String::new();
            BufReader::new(&mut sock).read_line(&mut line)?;
            let remote = Endpoint::from_line(&line)?;
            sock.write_all(local.to_json().as_bytes())?;
            sock.flush()?;
            remote
        };

        Self::connect_qp(qp, &remote)?;

        Ok((
            Self {
                ctx,
                pd,
                send_cq,
                recv_cq,
                qp,
                device_name: device_name.to_owned(),
            },
            sock,
        ))
    }

    fn open_device(name: &str) -> anyhow::Result<*mut ibv_context> {
        let mut n: i32 = 0;
        let list = unsafe { ibv_get_device_list(&mut n) };
        anyhow::ensure!(n > 0, "no RDMA devices");
        let mut ctx: *mut ibv_context = std::ptr::null_mut();
        for i in 0..n {
            let dev = unsafe { *list.offset(i as isize) };
            let cname = unsafe { std::ffi::CStr::from_ptr(ibv_get_device_name(dev)) };
            if cname.to_string_lossy() == name {
                ctx = unsafe { ibv_open_device(dev) };
                break;
            }
        }
        unsafe { ibv_free_device_list(list) };
        anyhow::ensure!(!ctx.is_null(), "device {name} not found");
        Ok(ctx)
    }

    fn query_gid(ctx: *mut ibv_context) -> anyhow::Result<[u8; 16]> {
        let mut gid: ibv_gid = unsafe { std::mem::zeroed() };
        unsafe {
            ibv_query_gid(ctx, PORT, DEFAULT_GID_INDEX as i32, &mut gid);
            Ok(gid.raw)
        }
    }

    fn make_qp(
        ctx: *mut ibv_context,
        send_cq: *mut ibv_cq,
        recv_cq: *mut ibv_cq,
        pd: *mut ibv_pd,
    ) -> anyhow::Result<*mut ibv_qp> {
        unsafe {
            let mut attr: ibv_qp_init_attr = std::mem::zeroed();
            attr.send_cq = send_cq;
            attr.recv_cq = recv_cq;
            attr.srq = std::ptr::null_mut();
            attr.cap.max_send_wr = 64;
            attr.cap.max_recv_wr = 4096;
            attr.cap.max_send_sge = 1;
            attr.cap.max_recv_sge = 1;
            attr.qp_type = IBV_QPT_RC;
            attr.sq_sig_all = 0;
            let qp = ibv_create_qp(pd, &mut attr);
            anyhow::ensure!(!qp.is_null(), "ibv_create_qp");
            Ok(qp)
        }
    }

    fn connect_qp(qp: *mut ibv_qp, remote: &Endpoint) -> anyhow::Result<()> {
        unsafe {
            let mut a: ibv_qp_attr = std::mem::zeroed();
            // INIT
            a.qp_state = IBV_QPS_INIT;
            a.port_num = PORT;
            a.pkey_index = 0;
            a.qp_access_flags = (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
                | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
                | ibv_access_flags::IBV_ACCESS_REMOTE_READ.0)
                as u32;
            Self::modify(
                qp,
                &mut a,
                (ibv_qp_attr_mask::IBV_QP_STATE.0
                    | ibv_qp_attr_mask::IBV_QP_PKEY_INDEX.0
                    | ibv_qp_attr_mask::IBV_QP_PORT.0
                    | ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS.0) as i32,
            )?;
            // RTR
            a.qp_state = IBV_QPS_RTR;
            a.path_mtu = IBV_MTU_1024;
            a.dest_qp_num = remote.qpn;
            a.rq_psn = 0;
            a.max_dest_rd_atomic = 16;
            a.min_rnr_timer = 12;
            a.ah_attr.dlid = 0;
            a.ah_attr.sl = 0;
            a.ah_attr.src_path_bits = 0;
            a.ah_attr.is_global = 1;
            a.ah_attr.port_num = PORT;
            a.ah_attr.grh.dgid.raw.copy_from_slice(&remote.gid);
            a.ah_attr.grh.sgid_index = DEFAULT_GID_INDEX as u8;
            a.ah_attr.grh.hop_limit = 255;
            Self::modify(
                qp,
                &mut a,
                (ibv_qp_attr_mask::IBV_QP_STATE.0
                    | ibv_qp_attr_mask::IBV_QP_AV.0
                    | ibv_qp_attr_mask::IBV_QP_PATH_MTU.0
                    | ibv_qp_attr_mask::IBV_QP_DEST_QPN.0
                    | ibv_qp_attr_mask::IBV_QP_RQ_PSN.0
                    | ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC.0
                    | ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER.0) as i32,
            )?;
            // RTS
            a.qp_state = IBV_QPS_RTS;
            a.sq_psn = 0;
            a.timeout = 14;
            a.retry_cnt = 7;
            a.rnr_retry = 7;
            a.max_rd_atomic = 16;
            Self::modify(
                qp,
                &mut a,
                (ibv_qp_attr_mask::IBV_QP_STATE.0
                    | ibv_qp_attr_mask::IBV_QP_SQ_PSN.0
                    | ibv_qp_attr_mask::IBV_QP_TIMEOUT.0
                    | ibv_qp_attr_mask::IBV_QP_RETRY_CNT.0
                    | ibv_qp_attr_mask::IBV_QP_RNR_RETRY.0
                    | ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC.0) as i32,
            )?;
            Ok(())
        }
    }

    #[inline]
    fn modify(qp: *mut ibv_qp, attr: &mut ibv_qp_attr, mask: i32) -> anyhow::Result<()> {
        let r = unsafe { ibv_modify_qp(qp, attr, mask) };
        anyhow::ensure!(r == 0, "ibv_modify_qp: {}", std::io::Error::last_os_error());
        Ok(())
    }

    /// Register a host buffer as an MR and return (mr, lkey, base address).
    pub fn register(&self, buf: &mut [u8]) -> anyhow::Result<RegisteredBuffer> {
        let mr = unsafe {
            ibv_reg_mr(
                self.pd,
                buf.as_mut_ptr() as *mut std::os::raw::c_void,
                buf.len(),
                (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
                    | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
                    | ibv_access_flags::IBV_ACCESS_REMOTE_READ.0) as i32,
            )
        };
        anyhow::ensure!(!mr.is_null(), "ibv_reg_mr");
        let lkey = unsafe { shim_mr_lkey(mr) };
        let rkey = unsafe { shim_mr_rkey(mr) };
        Ok(RegisteredBuffer {
            mr,
            lkey,
            rkey,
            base: buf.as_ptr() as u64,
            len: buf.len(),
        })
    }

    /// Post a single receive into `mr` with the given `wr_id`.
    pub fn post_receive_wr(&self, mr: &RegisteredBuffer, wr_id: u64) -> anyhow::Result<()> {
        unsafe {
            let mut sge: ibv_sge = std::mem::zeroed();
            sge.addr = mr.base;
            sge.length = mr.len as u32;
            sge.lkey = mr.lkey;
            let mut wr: ibv_recv_wr = std::mem::zeroed();
            wr.wr_id = wr_id;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;
            let mut bad = std::ptr::null_mut();
            if shim_post_recv(self.qp, &wr, &mut bad) != 0 {
                return Err(anyhow::anyhow!(
                    "post_recv: {}",
                    std::io::Error::last_os_error()
                ));
            }
        }
        Ok(())
    }

    /// Block (poll) until one receive completion is available; return it.
    pub fn poll_recv_one(&self) -> anyhow::Result<ibv_wc> {
        loop {
            let mut wcs: Vec<ibv_wc> = vec![unsafe { std::mem::zeroed::<ibv_wc>() }; 1];
            let n = unsafe { shim_poll_cq(self.recv_cq, 1, wcs.as_mut_ptr()) };
            if n < 0 {
                return Err(anyhow::anyhow!(
                    "poll_cq: {}",
                    std::io::Error::last_os_error()
                ));
            }
            if n == 1 {
                let wc = wcs[0];
                if wc.status != IBV_WC_SUCCESS {
                    return Err(anyhow::anyhow!("recv WC status {:?}", wc.status));
                }
                return Ok(wc);
            }
            std::thread::yield_now();
        }
    }

    /// Post a SEND from a byte range of a registered buffer.
    pub fn post_send_wr(
        &self,
        mr: &RegisteredBuffer,
        offset: usize,
        len: usize,
        wr_id: u64,
    ) -> anyhow::Result<()> {
        unsafe {
            let mut sge: ibv_sge = std::mem::zeroed();
            sge.addr = mr.base + offset as u64;
            sge.length = len as u32;
            sge.lkey = mr.lkey;
            let mut wr: ibv_send_wr = std::mem::zeroed();
            wr.wr_id = wr_id;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_SEND;
            wr.send_flags = ibv_send_flags::IBV_SEND_SIGNALED.0;
            let mut bad = std::ptr::null_mut();
            if shim_post_send(self.qp, &wr, &mut bad) != 0 {
                return Err(anyhow::anyhow!(
                    "post_send: {}",
                    std::io::Error::last_os_error()
                ));
            }
        }
        Ok(())
    }

    /// Post `count` receives of `mr.len()` bytes each onto the given MR.
    pub fn post_receives(&self, mr: &RegisteredBuffer, count: u64) -> anyhow::Result<()> {
        unsafe {
            let mut sge: ibv_sge = std::mem::zeroed();
            sge.addr = mr.base;
            sge.length = mr.len as u32;
            sge.lkey = mr.lkey;
            for i in 0..count {
                let mut wr: ibv_recv_wr = std::mem::zeroed();
                wr.wr_id = i;
                wr.sg_list = &mut sge;
                wr.num_sge = 1;
                let mut bad = std::ptr::null_mut();
                if shim_post_recv(self.qp, &wr, &mut bad) != 0 {
                    return Err(anyhow::anyhow!(
                        "post_recv: {}",
                        std::io::Error::last_os_error()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Poll receive completions; returns how many were completed.
    pub fn poll_recv(&self, max: usize) -> anyhow::Result<usize> {
        unsafe {
            let mut wcs: Vec<ibv_wc> = vec![std::mem::zeroed(); max];
            let n = shim_poll_cq(self.recv_cq, max as i32, wcs.as_mut_ptr());
            if n < 0 {
                return Err(anyhow::anyhow!(
                    "poll_cq: {}",
                    std::io::Error::last_os_error()
                ));
            }
            for wc in wcs.iter().take(n as usize) {
                if wc.status != IBV_WC_SUCCESS {
                    return Err(anyhow::anyhow!("recv WC status {:?}", wc.status));
                }
            }
            Ok(n as usize)
        }
    }

    /// Poll send completions; returns the completed wr_ids (all SUCCESS).
    pub fn poll_send_wrs(&self, max: usize) -> anyhow::Result<Vec<u64>> {
        unsafe {
            let mut wcs: Vec<ibv_wc> = vec![std::mem::zeroed(); max];
            let n = shim_poll_cq(self.send_cq, max as i32, wcs.as_mut_ptr());
            if n < 0 {
                return Err(anyhow::anyhow!(
                    "poll_cq: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let mut ids = Vec::with_capacity(n as usize);
            for wc in wcs.iter().take(n as usize) {
                if wc.status != IBV_WC_SUCCESS {
                    return Err(anyhow::anyhow!("send WC status {:?}", wc.status));
                }
                ids.push(wc.wr_id);
            }
            Ok(ids)
        }
    }

    /// Poll send completions; returns how many were completed.
    pub fn poll_send(&self, max: usize) -> anyhow::Result<usize> {
        unsafe {
            let mut wcs: Vec<ibv_wc> = vec![std::mem::zeroed(); max];
            let n = shim_poll_cq(self.send_cq, max as i32, wcs.as_mut_ptr());
            if n < 0 {
                return Err(anyhow::anyhow!(
                    "poll_cq: {}",
                    std::io::Error::last_os_error()
                ));
            }
            for wc in wcs.iter().take(n as usize) {
                if wc.status != IBV_WC_SUCCESS {
                    return Err(anyhow::anyhow!("send WC status {:?}", wc.status));
                }
            }
            Ok(n as usize)
        }
    }
}

impl Drop for RdmaLink {
    fn drop(&mut self) {
        unsafe {
            if !self.qp.is_null() {
                ibv_destroy_qp(self.qp);
            }
            if !self.send_cq.is_null() {
                ibv_destroy_cq(self.send_cq);
            }
            if !self.recv_cq.is_null() {
                ibv_destroy_cq(self.recv_cq);
            }
            if !self.pd.is_null() {
                ibv_dealloc_pd(self.pd);
            }
            if !self.ctx.is_null() {
                ibv_close_device(self.ctx);
            }
        }
    }
}

/// Receives a stream of gradient chunks over an RdmaLink.
///
/// The peer (replica rank) sends chunks of `chunk_len` f32 values in tensor
/// order. This type manages a RING of registered receive buffers. An RC QP
/// delivers completions in send order, and each ring slot re-posts with the
/// same wr_id (= its index), so `receive_into` drains exactly the next chunk.
pub struct NetworkGradientSource {
    link: RdmaLink,
    /// Ring of registered receive buffers; rings[i] uses wr_id i.
    rings: Vec<RegisteredBuffer>,
    /// Owning backing memory for each ring (keeps the MR valid).
    #[allow(dead_code)]
    ring_memory: Vec<Vec<u8>>,
    chunk_len: usize,
}

impl NetworkGradientSource {
    /// Connect (as listener) to a replica gradient sender.
    pub fn connect(
        device_name: &str,
        control_addr: std::net::SocketAddr,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(ring_count > 0, "ring_count must be positive");
        let (link, _sock) = RdmaLink::connect_with_control(device_name, control_addr, true)?;
        Self::finish(link, chunk_len, ring_count)
    }

    /// Listener connect that keeps the TCP control socket open for
    /// per-message metadata.
    pub fn connect_with_control(
        device_name: &str,
        control_addr: std::net::SocketAddr,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<(Self, TcpStream)> {
        anyhow::ensure!(ring_count > 0, "ring_count must be positive");
        let listener = TcpListener::bind(control_addr)?;
        Self::connect_with_bound_listener(device_name, listener, chunk_len, ring_count)
    }

    /// Listener connect on an already-bound control socket, keeping the TCP
    /// control socket open for per-message metadata.
    pub fn connect_with_bound_listener(
        device_name: &str,
        listener: TcpListener,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<(Self, TcpStream)> {
        anyhow::ensure!(ring_count > 0, "ring_count must be positive");
        let (link, sock) = RdmaLink::connect_with_bound_listener(device_name, listener)?;
        let source = Self::finish(link, chunk_len, ring_count)?;
        Ok((source, sock))
    }

    fn finish(link: RdmaLink, chunk_len: usize, ring_count: usize) -> anyhow::Result<Self> {
        let mut rings = Vec::with_capacity(ring_count);
        let mut ring_memory = Vec::with_capacity(ring_count);
        for _ in 0..ring_count {
            let mut buf = vec![0u8; chunk_len * std::mem::size_of::<f32>()];
            let mr = link.register(&mut buf)?;
            rings.push(mr);
            ring_memory.push(buf);
        }
        // Post the initial ring so the sender has buffer to run into.
        for (i, ring) in rings.iter().enumerate() {
            link.post_receive_wr(ring, i as u64)?;
        }
        Ok(Self {
            link,
            rings,
            ring_memory,
            chunk_len,
        })
    }

    /// Block until one message arrives, copy it into `out`, and re-post that
    /// ring slot. Uses the completion's byte_len so final (shorter) chunks of
    /// a stream are handled correctly. Returns byte count.
    pub fn receive_bytes_into(&mut self, out: &mut [u8]) -> anyhow::Result<usize> {
        let wc = self.link.poll_recv_one()?; // blocking-ish: returns next completion
        let ring_idx = wc.wr_id as usize;
        anyhow::ensure!(ring_idx < self.rings.len(), "unexpected wr_id {ring_idx}");
        let received = wc.byte_len as usize;
        anyhow::ensure!(
            received <= out.len(),
            "receive buffer too small: {} < {}",
            out.len(),
            received
        );
        let ring = &self.rings[ring_idx];
        // Copy the freshly-arrived chunk out of the ring MR.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ring.base as *const u8,
                out.as_mut_ptr() as *mut u8,
                received,
            );
        }
        // Re-post this ring slot (same wr_id) for the next chunk.
        self.link.post_receive_wr(ring, ring_idx as u64)?;
        Ok(received)
    }

    /// Block until one chunk arrives, copy it into `out`, and re-post that
    /// ring slot for the next chunk. Returns f32 count.
    pub fn receive_into(&mut self, out: &mut [f32]) -> anyhow::Result<usize> {
        let bytes = self.receive_bytes_into(bytemuck::cast_slice_mut(out))?;
        Ok(bytes / std::mem::size_of::<f32>())
    }
}

/// Sends a stream of chunks over an RdmaLink.
///
/// Mirrors [`NetworkGradientSource`] on the send side: a ring of registered
/// buffers; `send_chunk` picks the next slot, blocks if that slot still has
/// an in-flight send, copies the data in, and posts a signaled SEND. RC QPs
/// deliver messages in send order, so the receiver sees the exact chunk
/// sequence the sender posted.
pub struct NetworkSendChannel {
    link: RdmaLink,
    rings: Vec<RegisteredBuffer>,
    /// Owning backing memory for each ring (keeps the MR valid).
    #[allow(dead_code)]
    ring_memory: Vec<Vec<u8>>,
    chunk_len: usize,
    next_slot: usize,
    /// In-flight send wr_id per slot (None = free).
    outstanding: Vec<Option<u64>>,
}

impl NetworkSendChannel {
    pub fn connect(
        device_name: &str,
        control_addr: std::net::SocketAddr,
        is_listener: bool,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(chunk_len > 0, "chunk_len must be positive");
        anyhow::ensure!(ring_count > 0, "ring_count must be positive");
        let (link, _sock) = RdmaLink::connect_with_control(device_name, control_addr, is_listener)?;
        Self::finish(link, chunk_len, ring_count)
    }

    /// Connect that keeps the TCP control socket open for per-message
    /// metadata.
    pub fn connect_with_control(
        device_name: &str,
        control_addr: std::net::SocketAddr,
        is_listener: bool,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<(Self, TcpStream)> {
        anyhow::ensure!(chunk_len > 0, "chunk_len must be positive");
        anyhow::ensure!(ring_count > 0, "ring_count must be positive");
        let (link, sock) = RdmaLink::connect_with_control(device_name, control_addr, is_listener)?;
        let channel = Self::finish(link, chunk_len, ring_count)?;
        Ok((channel, sock))
    }

    fn finish(link: RdmaLink, chunk_len: usize, ring_count: usize) -> anyhow::Result<Self> {
        let mut rings = Vec::with_capacity(ring_count);
        let mut ring_memory = Vec::with_capacity(ring_count);
        for _ in 0..ring_count {
            let mut buf = vec![0u8; chunk_len * std::mem::size_of::<f32>()];
            let mr = link.register(&mut buf)?;
            rings.push(mr);
            ring_memory.push(buf);
        }
        Ok(Self {
            link,
            rings,
            ring_memory,
            chunk_len,
            next_slot: 0,
            outstanding: vec![None; ring_count],
        })
    }

    fn free_completed(&mut self) -> anyhow::Result<()> {
        for wr in self.link.poll_send_wrs(64)? {
            let i = wr as usize;
            if i < self.outstanding.len() {
                self.outstanding[i] = None;
            }
        }
        Ok(())
    }

    /// Send one chunk (<= chunk_len f32s). Blocks while the destination slot
    /// is still in flight; the receiver observes messages in send order.
    pub fn send_chunk(&mut self, data: &[f32]) -> anyhow::Result<()> {
        anyhow::ensure!(
            !data.is_empty() && data.len() <= self.chunk_len,
            "send chunk size {} out of range 1..={} f32s",
            data.len(),
            self.chunk_len
        );
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.rings.len();
        // Wait until this slot's previous send has completed.
        while self.outstanding[slot].is_some() {
            self.free_completed()?;
            if self.outstanding[slot].is_some() {
                std::thread::yield_now();
            }
        }
        let ring = &self.rings[slot];
        let bytes = data.len() * std::mem::size_of::<f32>();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ring.base as *mut u8, bytes);
        }
        self.link.post_send_wr(ring, 0, bytes, slot as u64)?;
        self.outstanding[slot] = Some(slot as u64);
        Ok(())
    }

    /// Wait until all in-flight sends have completed.
    pub fn flush(&mut self) -> anyhow::Result<()> {
        while self.outstanding.iter().any(|o| o.is_some()) {
            self.free_completed()?;
            std::thread::yield_now();
        }
        Ok(())
    }
}

/// A registered host memory region used for RDMA staging.
pub struct RegisteredBuffer {
    mr: *mut ibv_mr,
    pub lkey: u32,
    pub rkey: u32,
    pub base: u64,
    pub len: usize,
}

unsafe impl Send for RegisteredBuffer {}

impl Drop for RegisteredBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.mr.is_null() {
                ibv_dereg_mr(self.mr);
            }
        }
    }
}
