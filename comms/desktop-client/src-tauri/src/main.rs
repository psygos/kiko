#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use serde::{Deserialize, Serialize};
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{Manager, State};
use log::{info, warn, error, debug};

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

#[derive(Debug, Clone, Serialize)]
pub struct TelemetryUpdate {
    pub telemetry: RobotTelemetry,
    pub latency: u32,
    pub left_command: i8,
    pub right_command: i8,
}

struct CommandStream {
    socket: UdpSocket,
    sequence: u32,
    server_addr: String,
    last_telemetry: Option<RobotTelemetry>,
    left_speed: i8,
    right_speed: i8,
}

impl CommandStream {
    fn new(server_addr: String) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Creating new command stream to {}", server_addr);
        
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        info!("UDP socket bound successfully");
        
        socket.set_read_timeout(Some(Duration::from_millis(50)))?;
        socket.set_nonblocking(false)?;
        debug!("Socket timeouts configured");
        
        // Test connection by sending a ping command
        let test_cmd = RobotCommand {
            left_speed: 0,
            right_speed: 0,
            timeout_ms: 150,
            sequence: 0,
        };
        
        let test_packet = bincode::serialize(&test_cmd)?;
        socket.send_to(&test_packet, &server_addr)?;
        info!("Test packet sent to {}", server_addr);
        
        // Try to receive response to verify connection
        let mut buf = [0u8; 1024];
        match socket.recv_from(&mut buf) {
            Ok((len, addr)) => {
                info!("Received {} bytes from {} - connection verified", len, addr);
            }
            Err(e) => {
                warn!("No immediate response from server: {} - continuing anyway", e);
            }
        }
        
        Ok(CommandStream {
            socket,
            sequence: 0,
            server_addr,
            last_telemetry: None,
            left_speed: 0,
            right_speed: 0,
        })
    }
    
    fn send_command(&mut self) -> Result<Option<TelemetryUpdate>, Box<dyn std::error::Error>> {
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF;
        
        let cmd = RobotCommand {
            left_speed: self.left_speed,
            right_speed: self.right_speed,
            timeout_ms: 150, // 150ms timeout for safety
            sequence: self.sequence,
        };
        
        let start_time = Instant::now();
        
        // Send command
        let packet = bincode::serialize(&cmd)?;
        let bytes_sent = self.socket.send_to(&packet, &self.server_addr)?;
        debug!("Sent {} bytes (seq: {}, L: {}, R: {}) to {}", 
               bytes_sent, self.sequence, self.left_speed, self.right_speed, self.server_addr);
        
        // Try to receive telemetry (non-blocking with timeout)
        let mut buf = [0u8; 1024];
        match self.socket.recv_from(&mut buf) {
            Ok((len, addr)) => {
                let telemetry: RobotTelemetry = bincode::deserialize(&buf[..len])?;
                let latency = start_time.elapsed().as_millis() as u32;
                
                debug!("Received telemetry from {}: L: {}, R: {}, Battery: {}mV, Latency: {}ms", 
                       addr, telemetry.left_actual, telemetry.right_actual, 
                       telemetry.battery_mv, latency);
                
                self.last_telemetry = Some(telemetry.clone());
                
                Ok(Some(TelemetryUpdate {
                    telemetry,
                    latency,
                    left_command: self.left_speed,
                    right_command: self.right_speed,
                }))
            }
            Err(e) => {
                if self.sequence % 50 == 0 { // Log every 50th missed response to avoid spam
                    warn!("No telemetry response (seq: {}): {}", self.sequence, e);
                }
                Ok(None)
            }
        }
    }
    
    fn set_speeds(&mut self, left: i8, right: i8) {
        let old_left = self.left_speed;
        let old_right = self.right_speed;
        
        self.left_speed = left.clamp(-100, 100);
        self.right_speed = right.clamp(-100, 100);
        
        if old_left != self.left_speed || old_right != self.right_speed {
            info!("Speed changed: L: {} -> {}, R: {} -> {}", 
                  old_left, self.left_speed, old_right, self.right_speed);
        }
    }
}

type StreamState = Arc<Mutex<Option<CommandStream>>>;

#[tauri::command]
async fn get_connection_status(state: State<'_, StreamState>) -> Result<bool, String> {
    let state_guard = state.lock().unwrap();
    Ok(state_guard.is_some())
}

#[tauri::command]
async fn connect(
    state: State<'_, StreamState>,
    address: String,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    // Disconnect existing if any
    {
        *state.lock().unwrap() = None;
    }
    
    info!("Attempting to connect to: {}", address);
    
    // Parse address (add port if not specified)
    let server_addr = if address.contains(':') {
        address.clone()
    } else {
        format!("{}:8080", address)
    };
    
    info!("Parsed server address: {}", server_addr);
    
    // Create new connection
    let stream = CommandStream::new(server_addr.clone())
        .map_err(|e| {
            error!("Failed to create command stream: {}", e);
            format!("Connection failed: {}", e)
        })?;
    
    info!("Command stream created successfully");
    
    *state.lock().unwrap() = Some(stream);
    
    // Start command streaming thread
    let state_clone = state.inner().clone();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(40)); // 25Hz
            
            let should_continue = {
                let mut state_guard = state_clone.lock().unwrap();
                if let Some(stream) = state_guard.as_mut() {
                    match stream.send_command() {
                        Ok(Some(update)) => {
                            // Emit telemetry update to frontend
                            if let Err(e) = app_handle.emit_all("telemetry-update", &update) {
                                warn!("Failed to emit telemetry update: {}", e);
                            }
                            true
                        }
                        Ok(None) => true, // No telemetry received, but command sent
                        Err(e) => {
                            error!("Command stream error: {}", e);
                            if let Err(emit_err) = app_handle.emit_all("connection-error", e.to_string()) {
                                error!("Failed to emit connection error: {}", emit_err);
                            }
                            false
                        }
                    }
                } else {
                    false // Not connected
                }
            };
            
            if !should_continue {
                break;
            }
        }
        
        // Emit disconnection event
        info!("Command stream thread terminated - emitting connection-lost event");
        if let Err(e) = app_handle.emit_all("connection-lost", ()) {
            error!("Failed to emit connection-lost event: {}", e);
        }
    });
    
    Ok(format!("Connected to {}", server_addr))
}

#[tauri::command]
async fn disconnect(state: State<'_, StreamState>) -> Result<(), String> {
    info!("Disconnecting from robot server");
    *state.lock().unwrap() = None;
    info!("Disconnected successfully");
    Ok(())
}

#[tauri::command]
async fn set_motor_speeds(
    state: State<'_, StreamState>,
    left: i8,
    right: i8,
) -> Result<(), String> {
    let mut state_guard = state.lock().unwrap();
    if let Some(stream) = state_guard.as_mut() {
        stream.set_speeds(left, right);
        Ok(())
    } else {
        warn!("Attempted to set motor speeds while not connected");
        Err("Not connected".to_string())
    }
}

#[tauri::command]
async fn emergency_stop(state: State<'_, StreamState>) -> Result<(), String> {
    warn!("EMERGENCY STOP ACTIVATED");
    let mut state_guard = state.lock().unwrap();
    if let Some(stream) = state_guard.as_mut() {
        stream.set_speeds(0, 0);
        // Send immediate stop command
        match stream.send_command() {
            Ok(_) => info!("Emergency stop command sent successfully"),
            Err(e) => error!("Failed to send emergency stop: {}", e),
        }
        Ok(())
    } else {
        warn!("Emergency stop called while not connected");
        Err("Not connected".to_string())
    }
}

fn main() {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    
    info!("Starting Robot Control Client v{}", env!("CARGO_PKG_VERSION"));
    
    let stream_state: StreamState = Arc::new(Mutex::new(None));
    
    tauri::Builder::default()
        .manage(stream_state)
        .invoke_handler(tauri::generate_handler![
            connect,
            disconnect,
            set_motor_speeds,
            emergency_stop,
            get_connection_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}