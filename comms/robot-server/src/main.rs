use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

mod protocol;
use protocol::*;
use robot_server::config;
use robot_server::V2ControllerOwner;

const OWNER_SIBLING_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(1);

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    log::info!("Starting Kiko Robot Server...");

    let runtime_config = config::ServerArgs::parse_runtime()?;
    if runtime_config.legacy_http_camera_enabled() {
        log::warn!(
            "legacy HTTP/camera service explicitly enabled; its state is not V2 controller evidence"
        );
        let state = Arc::new(RwLock::new(RobotState::default()));
        tokio::try_join!(run_v2(runtime_config), async {
            protocol::http_service(state)
                .await
                .context("legacy HTTP/camera service stopped")
        },)?;
    } else {
        log::info!("legacy HTTP/camera service disabled; running the typed V2 owner only");
        run_v2(runtime_config).await?;
    }

    Ok(())
}

async fn run_v2(runtime_config: config::ServerRuntimeConfig) -> Result<()> {
    let (command_bind, controller) = runtime_config.into_v2();
    if let Some(controller) = controller {
        log::info!(
            "enabling V2 controller actor for claimed hardware profile {}",
            controller.hardware_profile_claim_id()
        );
        let owner = V2ControllerOwner::start(controller, command_bind)
            .await
            .context("configured V2 controller could not start; refusing a degraded server")?;
        owner
            .join(OWNER_SIBLING_SHUTDOWN_TIMEOUT)
            .await
            .context("V2 controller owner stopped")?;
    } else {
        log::warn!(
            "no controller authority supplied; V2 actuation is unavailable and status reports disconnected"
        );
        robot_server::unavailable_udp_service(command_bind)
            .await
            .context("unavailable V2 status service stopped")?;
    }

    Ok(())
}
