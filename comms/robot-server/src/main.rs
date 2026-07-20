use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::RwLock;

mod actuation_v2;
mod deadline;
mod protocol;
use protocol::*;
use robot_server::config;

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
    let command_bind = runtime_config.command_bind();
    if let Some(controller) = runtime_config.controller() {
        log::info!(
            "enabling V2 controller actor for claimed hardware profile {}",
            controller.hardware_profile_claim_id()
        );
        let started = actuation_v2::start_serial_actor(
            controller.clone(),
            Arc::new(actuation_v2::NoopActuationTelemetry),
        )
        .await;
        match started {
            Ok((actuation, actor)) => {
                tokio::try_join!(
                    async {
                        actuation_v2::udp_service(command_bind, actuation)
                            .await
                            .context("V2 command service stopped")
                    },
                    supervise_actuation_actor(actor),
                )?;
            }
            Err(source) => {
                return Err(source).context(
                    "configured V2 controller could not start; refusing a degraded server",
                );
            }
        }
    } else {
        log::warn!(
            "no controller authority supplied; V2 actuation is unavailable and status reports disconnected"
        );
        actuation_v2::unavailable_udp_service(command_bind)
            .await
            .context("unavailable V2 status service stopped")?;
    }

    Ok(())
}

async fn supervise_actuation_actor(
    actor: tokio::task::JoinHandle<std::result::Result<(), actuation_v2::ActuationActorError>>,
) -> Result<()> {
    match actor.await {
        Ok(Ok(())) => anyhow::bail!("V2 controller actor ended unexpectedly without an error"),
        Ok(Err(source)) => Err(source).context("V2 controller actor failed"),
        Err(source) => Err(source).context("V2 controller actor task failed"),
    }
}
