use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::UdpSocket;
use tokio::sync::RwLock;
use tokio_serial::SerialPortBuilderExt;
use warp::Filter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotCommand {
    pub left_speed: i8,
    pub right_speed: i8,
    pub timeout_ms: u16,
    pub sequence: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotTelemetry {
    pub left_actual: i8,
    pub right_actual: i8,
    pub battery_mv: u16,
    pub timestamp_ms: u32,
}

#[derive(Default)]
pub struct RobotState {
    pub last_command: Option<RobotCommand>,
    pub last_telemetry: Option<RobotTelemetry>,
    pub video_frame: Vec<u8>, // This is a placeholder rn, will add functionality next
    pub dashboard_addr: Option<std::net::SocketAddr>,
}

pub async fn udp_service(state: Arc<RwLock<RobotState>>) -> Result<()> {
    let socket = UdpSocket::bind("0.0.0.0:8080").await?;
    log::info!("UDP service listening on :8080");

    let mut buf = vec![0u8; 1024];

    loop {
        match socket.recv_from(&mut buf).await {
            Ok((len, addr)) => {
                println!("Received {} bytes from {}", len, addr);
                if let Ok(cmd) = bincode::deserialize::<RobotCommand>(&buf[..len]) {
                    println!("Received command: {:?} from {}", cmd, addr);
                    log::debug!("Received command: {:?} from {}", cmd, addr);

                    let mut state_guard = state.write().await;
                    state_guard.last_command = Some(cmd);
                    state_guard.dashboard_addr = Some(addr);

                    // Send immediate ACK with telemetry or default values
                    let telemetry = state_guard.last_telemetry.clone().unwrap_or(RobotTelemetry {
                        left_actual: 0,
                        right_actual: 0,
                        battery_mv: 0,
                        timestamp_ms: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u32,
                    });
                    
                    if let Ok(data) = bincode::serialize(&telemetry) {
                        drop(state_guard); // Release lock before await
                        socket.send_to(&data, addr).await?;
                    }
                }
            }
            Err(e) => log::error!("UDP error: {}", e),
        }
    }
}

pub async fn serial_service(state: Arc<RwLock<RobotState>>) -> Result<()> {
    println!("Starting serial service...");
    // this is where the stm32 port goes
    let port_names = vec!["/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyAMA0"];
    let mut port = None;

    for name in port_names {
        println!("Trying to open serial port: {}", name);
        if let Ok(p) = tokio_serial::new(name, 115200)
            .timeout(std::time::Duration::from_millis(10))
            .open_native_async()
        {
            println!("Serial port opened successfully: {}", name);
            log::info!("Serial port opened: {}", name);
            port = Some(p);
            break;
        } else {
            println!("Failed to open serial port: {}", name);
        }
    }

    let mut port = port.ok_or_else(|| anyhow::anyhow!("No serial port found"))?;
    println!("Serial service initialized, entering main loop...");

    let mut serial_buf = vec![0u8; 256];
    let mut rx_buffer = Vec::new();
    let mut last_command_time = tokio::time::Instant::now();

    loop {
        // freq 50Hz
        // parser based uart comms
        if last_command_time.elapsed() > tokio::time::Duration::from_millis(20) {
            if let Some(cmd) = state.read().await.last_command.clone() {
                let packet = format!(
                    "CMD,{},{},{}\n",
                    cmd.left_speed, cmd.right_speed, cmd.timeout_ms
                );

                println!("Sending to STM32: {}", packet.trim());
                use tokio::io::AsyncWriteExt;
                if let Err(e) = port.write_all(packet.as_bytes()).await {
                    println!("Serial write error: {}", e);
                    log::error!("Serial write error: {}", e);
                } else {
                    println!("Successfully sent to STM32");
                }
            } else {
                println!("No command to send to STM32");
            }
            last_command_time = tokio::time::Instant::now();
        }

        use tokio::io::AsyncReadExt;
        match port.read(&mut serial_buf).await {
            Ok(n) if n > 0 => {
                println!("Received {} bytes from STM32: {:?}", n, &serial_buf[..n]);
                rx_buffer.extend_from_slice(&serial_buf[..n]);

                while let Some(pos) = rx_buffer.iter().position(|&b| b == b'\n') {
                    let line = String::from_utf8_lossy(&rx_buffer[..pos]).to_string();
                    println!("STM32 response: {}", line);
                    rx_buffer.drain(..=pos);

                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 4 && parts[0] == "TEL" {
                        if let (Ok(left), Ok(right), Ok(battery)) = (
                            parts[1].parse::<i8>(),
                            parts[2].parse::<i8>(),
                            parts[3].parse::<u16>(),
                        ) {
                            let telemetry = RobotTelemetry {
                                left_actual: left,
                                right_actual: right,
                                battery_mv: battery,
                                timestamp_ms: last_command_time.elapsed().as_millis() as u32,
                            };

                            state.write().await.last_telemetry = Some(telemetry);
                        }
                    }
                }
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(e) => log::error!("Serial read error: {}", e),
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    }
}

pub async fn http_service(state: Arc<RwLock<RobotState>>) -> Result<()> {
    let state_filter = warp::any().map(move || state.clone());

    let status = warp::path("status")
        .and(state_filter.clone())
        .map(|state: Arc<RwLock<RobotState>>| {
            warp::reply::json(&serde_json::json!({
                "status": "running",
                "has_command": state.try_read().map(|s| s.last_command.is_some()).unwrap_or(false),
                "has_telemetry": state.try_read().map(|s| s.last_telemetry.is_some()).unwrap_or(false),
            }))
        });

    let video = warp::path("video").map(|| {
        warp::http::Response::builder()
            .status(200)
            .header("Content-Type", "text/plain")
            .body("Video streaming coming soon")
    });

    let routes = status.or(video);

    log::info!("HTTP service starting on :3030");
    warp::serve(routes).run(([0, 0, 0, 0], 3030)).await;

    Ok(())
}
