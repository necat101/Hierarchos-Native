// Build-only oracle adapter: remote RDMA is unavailable on Windows.
// The upstream single-device graph and all shader bytes remain untouched.
use std::net::{SocketAddr,TcpListener,TcpStream};
use anyhow::{bail,Result};
pub struct NetworkGradientSource;
pub struct NetworkSendChannel;
impl NetworkGradientSource {
 pub fn connect(_: &str,_:SocketAddr,_:usize,_:usize)->Result<Self>{bail!("oracle RDMA unavailable on Windows")}
 pub fn connect_with_control(_: &str,_:SocketAddr,_:usize,_:usize)->Result<(Self,TcpStream)>{bail!("oracle RDMA unavailable on Windows")}
 pub fn connect_with_bound_listener(_: &str,_:TcpListener,_:usize,_:usize)->Result<(Self,TcpStream)>{bail!("oracle RDMA unavailable on Windows")}
 pub fn receive_into(&mut self,_:&mut[f32])->Result<usize>{bail!("oracle RDMA unavailable on Windows")}
 pub fn receive_bytes_into(&mut self,_:&mut[u8])->Result<usize>{bail!("oracle RDMA unavailable on Windows")}
}
impl NetworkSendChannel {
 pub fn connect(_: &str,_:SocketAddr,_:bool,_:usize,_:usize)->Result<Self>{bail!("oracle RDMA unavailable on Windows")}
 pub fn connect_with_control(_: &str,_:SocketAddr,_:bool,_:usize,_:usize)->Result<(Self,TcpStream)>{bail!("oracle RDMA unavailable on Windows")}
 pub fn send_chunk(&mut self,_:&[f32])->Result<()>{bail!("oracle RDMA unavailable on Windows")}
 pub fn flush(&mut self)->Result<()>{bail!("oracle RDMA unavailable on Windows")}
}
