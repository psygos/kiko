# Robot Control Desktop Client

This is the Tauri desktop application for controlling kiko with live video streaming(mjpeg) , command stream(udp socket) and telemetry.

## Features

- **Real-time Control**: 25Hz UDP command streaming with keyboard controls
- **Live Video**: MJPEG video streaming from robot camera (TODO: webRTC)
- **Telemetry Display**: Real-time battery, latency, and movement feedback (Much of this is not active yet)
- **Safety Features**: Emergency stop and connection timeout protection

## Controls

- **Movement**: Arrow keys or WASD
- **Speed Boost**: Hold Shift while moving
- **Emergency Stop**: Spacebar
- **Speed Control**: Slider in control panel

## Setup

1. Install dependencies:
```bash
npm install
```

2. Run in development:
```bash
npm run dev
```

3. Build for production:
```bash
npm run build
```

## Connection

- **UDP**: Port 8080 for commands and telemetry
- **HTTP**: Port 3030 for video streaming

The current default robot address is `10.42.200.50` but the next update will have UI to specify IP of choice.

