use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

mod protocol;
use protocol::*;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    log::info!("Starting Kiko Robot Server...");

    let state = Arc::new(RwLock::new(RobotState::default()));

    tokio::try_join!(
        protocol::udp_service(state.clone()),
        protocol::serial_service(state.clone()),
        protocol::http_service(state.clone()),
    )?;

    Ok(())
}
