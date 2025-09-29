use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::UdpSocket;
use tokio::sync::{RwLock, broadcast};
use tokio_serial::SerialPortBuilderExt;
use warp::Filter;
use std::convert::Infallible;
use futures::StreamExt;
use bytes::Bytes;
use std::process::Stdio;
use tokio::process::Command;
use warp::hyper::Body;
use tokio_stream::wrappers::BroadcastStream;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotOdometry {
    pub left_ticks: i64,
    pub right_ticks: i64,
    pub left_velocity: i16,
    pub right_velocity: i16,
    pub timestamp_ms: u32,
}

#[derive(Default)]
pub struct RobotState {
    pub last_command: Option<RobotCommand>,
    pub last_telemetry: Option<RobotTelemetry>,
    pub last_odometry: Option<RobotOdometry>,
    pub video_frame: Vec<u8>,
    pub dashboard_addr: Option<std::net::SocketAddr>,
    pub video_tx: Option<broadcast::Sender<Bytes>>,
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
    // Try a set of common serial port paths for Linux and macOS so the server works cross-platform.
    // On macOS, USB CDC devices typically appear as /dev/cu.usb* or /dev/tty.usb*.
    let port_names = vec![
        // Typical Linux interfaces
        "/dev/ttyACM0",
        "/dev/ttyUSB0",
        "/dev/ttyAMA0",
        // Common macOS interfaces
        "/dev/cu.usbmodem1103",
        "/dev/tty.usbmodem1103",
        "/dev/cu.usbmodem",
        "/dev/tty.usbmodem",
        "/dev/cu.usbserial",
        "/dev/tty.usbserial",
    ];
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
    let mut last_command_time = tokio::time::Instant::now() - tokio::time::Duration::from_millis(25);

    loop {
        // freq 50Hz
        // parser based uart comms
        if last_command_time.elapsed() > tokio::time::Duration::from_millis(20) {
            let cmd_opt = state.read().await.last_command.clone();
            if let Some(cmd) = cmd_opt {
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
                println!("No command to send to STM32 (last_command is None)");
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
                    } else if parts.len() >= 6 && parts[0] == "ODO" {
                        if let (Ok(left_ticks), Ok(right_ticks), Ok(left_vel), Ok(right_vel), Ok(timestamp)) = (
                            parts[1].parse::<i64>(),
                            parts[2].parse::<i64>(),
                            parts[3].parse::<i16>(),
                            parts[4].parse::<i16>(),
                            parts[5].parse::<u32>(),
                        ) {
                            let odometry = RobotOdometry {
                                left_ticks,
                                right_ticks,
                                left_velocity: left_vel,
                                right_velocity: right_vel,
                                timestamp_ms: timestamp,
                            };

                            println!("Odometry: left_ticks={}, right_ticks={}, left_vel={}, right_vel={}", 
                                left_ticks, right_ticks, left_vel, right_vel);
                            state.write().await.last_odometry = Some(odometry);
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
    // Create video broadcast channel
    let (video_tx, _) = broadcast::channel::<Bytes>(4);
    
    // Store the sender in state
    {
        let mut state_guard = state.write().await;
        state_guard.video_tx = Some(video_tx.clone());
    }
    
    let state_filter = warp::any().map(move || state.clone());
    
    // Spawn GStreamer process for USB camera
    tokio::spawn(async move {
        println!("Starting camera stream...");
        
        // Try different camera devices
        let devices = vec!["/dev/video0", "/dev/video1", "/dev/video2"];
        let mut camera_found = false;
        
        for device in devices {
            // Check if device exists
            if std::path::Path::new(device).exists() {
                println!("Found camera at {}", device);
                camera_found = true;
                
                // Simple v4l2 capture and JPEG encoding
                let output = Command::new("ffmpeg")
                    .args(&[
                        "-f", "v4l2",
                        "-video_size", "640x480",
                        "-framerate", "30",
                        "-i", device,
                        "-f", "mjpeg",
                        "-q:v", "5",  // Quality 1-31 (lower is better)
                        "-"
                    ])
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn();
                
                if let Ok(mut child) = output {
                    if let Some(stdout) = child.stdout.take() {
                        let mut reader = tokio::io::BufReader::new(stdout);
                        let mut buffer = vec![0u8; 65536];
                        let mut jpeg_buffer = Vec::new();
                        
                        loop {
                            use tokio::io::AsyncReadExt;
                            match reader.read(&mut buffer).await {
                                Ok(0) => break, // EOF
                                Ok(n) => {
                                    jpeg_buffer.extend_from_slice(&buffer[..n]);
                                    
                                    // Look for JPEG markers
                                    while let Some(start) = find_jpeg_start(&jpeg_buffer) {
                                        if let Some(end) = find_jpeg_end(&jpeg_buffer[start..]) {
                                            let frame_end = start + end + 2;
                                            let frame = jpeg_buffer[start..frame_end].to_vec();
                                            
                                            // Broadcast the frame
                                            let _ = video_tx.send(Bytes::from(frame));
                                            
                                            // Remove processed data
                                            jpeg_buffer.drain(..frame_end);
                                        } else {
                                            break; // Wait for more data
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Camera read error: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                    
                    // Clean up
                    let _ = child.kill().await;
                }
                break;
            }
        }
        
        if !camera_found {
            eprintln!("No camera found. Video streaming disabled.");
        }
    });

    let status = warp::path("status")
        .and(state_filter.clone())
        .map(|state: Arc<RwLock<RobotState>>| {
            warp::reply::json(&serde_json::json!({
                "status": "running",
                "has_command": state.try_read().map(|s| s.last_command.is_some()).unwrap_or(false),
                "has_telemetry": state.try_read().map(|s| s.last_telemetry.is_some()).unwrap_or(false),
                "has_odometry": state.try_read().map(|s| s.last_odometry.is_some()).unwrap_or(false),
            }))
        });

    // MJPEG stream endpoint
    let video = warp::path("video.mjpeg")
        .and(state_filter.clone())
        .and_then(|state: Arc<RwLock<RobotState>>| async move {
            let rx = {
                let state_guard = state.read().await;
                state_guard.video_tx.as_ref().map(|tx| tx.subscribe())
            };
            
            if let Some(rx) = rx {
                let stream = BroadcastStream::new(rx);
                let body_stream = stream
                    .filter_map(|result| async move {
                        result.ok().map(|frame| {
                            let mut data = Vec::new();
                            data.extend_from_slice(b"--frame\r\n");
                            data.extend_from_slice(b"Content-Type: image/jpeg\r\n");
                            data.extend_from_slice(b"Content-Length: ");
                            data.extend_from_slice(frame.len().to_string().as_bytes());
                            data.extend_from_slice(b"\r\n\r\n");
                            data.extend_from_slice(&frame);
                            data.extend_from_slice(b"\r\n");
                            
                            Ok::<_, std::convert::Infallible>(Bytes::from(data))
                        })
                    });
                
                let response = warp::http::Response::builder()
                    .header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    .header("Cache-Control", "no-cache")
                    .body(Body::wrap_stream(body_stream))
                    .unwrap();
                
                Ok::<_, std::convert::Infallible>(response)
            } else {
                let response = warp::http::Response::builder()
                    .status(503)
                    .body(Body::from("Video stream not available"))
                    .unwrap();
                Ok(response)
            }
        });

    let debug = warp::path("debug")
        .and(state_filter.clone())
        .and_then(|state: Arc<RwLock<RobotState>>| async move {
            let s = state.read().await;
            Ok::<_, Infallible>(warp::reply::json(&serde_json::json!({
                "last_command": s.last_command,
                "last_telemetry": s.last_telemetry,
                "last_odometry": s.last_odometry,
            })))
        });

    let odometry = warp::path("odometry")
        .and(state_filter.clone())
        .and_then(|state: Arc<RwLock<RobotState>>| async move {
            let s = state.read().await;
            if let Some(odo) = &s.last_odometry {
                Ok::<_, Infallible>(Box::new(warp::reply::json(odo)) as Box<dyn warp::Reply>)
            } else {
                Ok(Box::new(warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({
                        "error": "No odometry data available"
                    })),
                    warp::http::StatusCode::SERVICE_UNAVAILABLE
                )) as Box<dyn warp::Reply>)
            }
        });

    let routes = status.or(video).or(debug).or(odometry);

    log::info!("HTTP service starting on :3030");
    warp::serve(routes).run(([0, 0, 0, 0], 3030)).await;

    Ok(())
}

// Helper functions for JPEG parsing
fn find_jpeg_start(data: &[u8]) -> Option<usize> {
    data.windows(2)
        .position(|window| window[0] == 0xFF && window[1] == 0xD8)
}

fn find_jpeg_end(data: &[u8]) -> Option<usize> {
    data.windows(2)
        .position(|window| window[0] == 0xFF && window[1] == 0xD9)
}
