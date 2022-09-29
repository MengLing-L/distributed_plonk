use std::net::SocketAddr;

use serde::Deserialize;

#[derive(Clone, Deserialize)]
pub struct NetworkConfig {
    pub master: SocketAddr,
    pub slaves: Vec<SocketAddr>,
    pub peers: Vec<SocketAddr>,
}
