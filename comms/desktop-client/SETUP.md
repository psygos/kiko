# Robot Control Client Setup

## Complete Setup Instructions

### 1. Server Setup (On Jetson/Robot)

First, update your robot server dependencies in `comms/robot-server/Cargo.toml`:

```toml
[dependencies]
tokio = { version = "1.35", features = ["full"] }
tokio-serial = "5.4"
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"
warp = "0.3"
bytes = "1.5"
log = "0.4"
env_logger = "0.10"
serde_json = "1.0"
tokio-stream = { version = "0.1", features = ["sync"] }
futures = "0.3"
```

Install ffmpeg for camera streaming:
```bash
sudo apt-get install ffmpeg
```

Run the server:
```bash
cd comms/robot-server
RUST_LOG=info cargo run
```

### 2. Client Setup (Desktop)

Install dependencies:
```bash
cd comms/desktop-client
npm install
```

Run in development mode:
```bash
npm run dev
```

### 3. Connection Setup

1. Click "Connect" in the client
2. Enter your robot's IP address (default: `10.42.200.50`)
3. The client will connect on port `8080` for commands
4. Video stream will start automatically from port `3030`

### 4. Controls

- **Movement**: Arrow keys or WASD
- **Speed Boost**: Hold Shift while moving
- **Emergency Stop**: Spacebar
- **Speed Control**: Use the slider in the control panel

### 5. Network Requirements

Ensure these ports are open between client and robot:
- **UDP 8080**: Command/telemetry streaming
- **TCP 3030**: HTTP video streaming

### 6. Troubleshooting

**No Video**: Check that ffmpeg is installed and camera device exists at `/dev/video0-2`

**Connection Failed**: Verify robot server is running on port 8080

**High Latency**: Check network connection quality between client and robot

**Controls Not Working**: Ensure the application window has focus for keyboard input

### 7. Architecture Overview

```
Desktop Client (Tauri)
├── Frontend: HTML/CSS/JavaScript UI
├── Backend: Rust UDP streaming
└── Video: MJPEG over HTTP

Robot Server (Rust)
├── UDP: Commands & telemetry (25Hz)
├── Serial: STM32 communication  
└── HTTP: Camera streaming
```

### 8. Safety Features

- 150ms command timeout (robot stops if no commands received)
- Emergency stop on spacebar
- Connection monitoring with automatic disconnect
- Speed limits and input validation
