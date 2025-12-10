// Safely obtain Tauri helpers (works both inside & outside Tauri, e.g. normal browser)
let invoke, listen;
try {
    if (window.__TAURI__ && window.__TAURI__.tauri && window.__TAURI__.event) {
        invoke = window.__TAURI__.tauri.invoke;
        listen = window.__TAURI__.event.listen;
    } else {
        console.warn('Tauri global not found – falling back to dynamic import');
        // Dynamically import @tauri-apps/api when available (in dev bundles)
        import('@tauri-apps/api/tauri').then(mod => {
            invoke = mod.invoke;
            console.log('Loaded invoke via dynamic import');
        }).catch(err => console.error('Failed to import tauri invoke:', err));
        import('@tauri-apps/api/event').then(mod => {
            listen = mod.listen;
            console.log('Loaded listen via dynamic import');
        }).catch(err => console.error('Failed to import tauri listen:', err));
    }
} catch (e) {
    console.error('Error while acquiring Tauri APIs:', e);
}
// Ensure functions exist to avoid runtime crashes until imports resolve
invoke = invoke || (() => Promise.reject('Tauri invoke not ready'));
listen = listen || (() => Promise.reject('Tauri listen not ready'));

// Control state
let connected = false;
// Connection settings with persistence
let host = localStorage.getItem('kiko_host') || '10.42.200.50';
let udpPort = parseInt(localStorage.getItem('kiko_udp_port') || '8080', 10);
let httpPort = parseInt(localStorage.getItem('kiko_http_port') || '3030', 10);
let robotAddress = `${host}:${udpPort}`;
let currentLeft = 0;
let currentRight = 0;
let baseSpeed = 50;
let sequenceNum = 0;

// Key states for smooth control
const keys = {
    ArrowUp: false,
    ArrowDown: false,
    ArrowLeft: false,
    ArrowRight: false,
    w: false,
    s: false,
    a: false,
    d: false,
    Shift: false
};

// UI Elements
const videoStream = document.getElementById('videoStream');
const videoOffline = document.getElementById('videoOffline');
const statusIndicator = document.getElementById('statusIndicator');
const connectionText = document.getElementById('connectionText');
const directionArrow = document.getElementById('directionArrow');
const speedDisplay = document.getElementById('speedDisplay');
const connectBtn = document.getElementById('connectBtn');
const hostInput = document.getElementById('hostInput');
const udpPortInput = document.getElementById('udpPortInput');
const httpPortInput = document.getElementById('httpPortInput');
// -------------------- DEBUG LOGGING SETUP --------------------
const debugDiv = document.getElementById('debugLog');
if (debugDiv) {
    const hdr = document.createElement('div');
    hdr.textContent = '--- Debug panel ready ---';
    debugDiv.appendChild(hdr);
}
const originalLog = console.log.bind(console);
function pushDebug(...args) {
    const time = new Date().toLocaleTimeString();
    const msg = args.map(a => (typeof a === 'object' ? JSON.stringify(a) : a)).join(' ');
    const line = `[${time}] ${msg}`;
    if (debugDiv) {
        const el = document.createElement('div');
        el.textContent = line;
        debugDiv.prepend(el);
        if (debugDiv.childElementCount > 300) {
            debugDiv.removeChild(debugDiv.lastChild);
        }
    }
    originalLog(...args);
}
console.log = (...args) => pushDebug(...args);
console.error = (...args) => pushDebug(...args);
window.addEventListener('error', (e) => pushDebug('Unhandled error:', e.message || e));
// -------------------- END DEBUG LOGGING --------------------

// Keyboard event handlers
document.addEventListener('keydown', (e) => {
    if (e.key in keys) {
        keys[e.key] = true;
        updateMotorSpeeds();
    }
});

document.addEventListener('keyup', (e) => {
    if (e.key in keys) {
        keys[e.key] = false;
        updateMotorSpeeds();
    }
});

// Prevent arrow keys from scrolling the page
window.addEventListener('keydown', (e) => {
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {
        e.preventDefault();
    }
});

// Calculate motor speeds from keyboard input
function updateMotorSpeeds() {
    if (!connected) return;
    
    const speed = keys.Shift ? Math.min(100, Math.floor(baseSpeed * 1.5)) : baseSpeed;
    
    let left = 0;
    let right = 0;
    
    const forward = keys.ArrowUp || keys.w;
    const backward = keys.ArrowDown || keys.s;
    const turnLeft = keys.ArrowLeft || keys.a;
    const turnRight = keys.ArrowRight || keys.d;
    
    if (forward && !backward) {
        left = speed;
        right = speed;
        
        if (turnLeft && !turnRight) {
            left = Math.floor(speed * 0.3);
        } else if (turnRight && !turnLeft) {
            right = Math.floor(speed * 0.3);
        }
    } else if (backward && !forward) {
        left = -speed;
        right = -speed;
        
        if (turnLeft && !turnRight) {
            left = -Math.floor(speed * 0.3);
        } else if (turnRight && !turnLeft) {
            right = -Math.floor(speed * 0.3);
        }
    } else if (turnLeft && !turnRight) {
        left = -Math.floor(speed / 2);
        right = Math.floor(speed / 2);
    } else if (turnRight && !turnLeft) {
        left = Math.floor(speed / 2);
        right = -Math.floor(speed / 2);
    }
    
    // Send to backend
    setMotorSpeeds(left, right);
}

// Send motor speeds to backend
async function setMotorSpeeds(left, right) {
    try {
        await invoke('set_motor_speeds', { left, right });
        currentLeft = left;
        currentRight = right;
        
        // Log significant speed changes for debugging
        if (Math.abs(left) > 50 || Math.abs(right) > 50) {
            console.log('High speed command:', { left, right });
        }
    } catch (err) {
        console.error('Failed to set motor speeds:', err);
        // If not connected, try to reconnect
        if (String(err).includes('Not connected')) {
            setConnected(false);
        }
    }
}

// Speed is now controlled by the interactive speed bar in the HTML

// Direction visualization
function updateDirection(left, right) {
    const leftElem = document.getElementById('leftSpeed');
    const rightElem = document.getElementById('rightSpeed');
    leftElem.textContent = left;
    rightElem.textContent = right;
    
    // Determine direction for arrow
    if (left === 0 && right === 0) {
        directionArrow.className = 'direction-arrow stopped';
    } else if (left > 0 && right > 0) {
        if (Math.abs(left - right) < 10) {
            directionArrow.className = 'direction-arrow up';
        } else if (left > right) {
            directionArrow.className = 'direction-arrow right';
        } else {
            directionArrow.className = 'direction-arrow left';
        }
    } else if (left < 0 && right < 0) {
        directionArrow.className = 'direction-arrow down';
    } else if (left < 0 && right > 0) {
        directionArrow.className = 'direction-arrow left';
    } else if (left > 0 && right < 0) {
        directionArrow.className = 'direction-arrow right';
    }
}

// Connection management
async function connect() {
    try {
        // Build address from inputs and persist
        host = hostInput?.value?.trim() || host;
        udpPort = parseInt(udpPortInput?.value || String(udpPort), 10) || udpPort;
        httpPort = parseInt(httpPortInput?.value || String(httpPort), 10) || httpPort;
        localStorage.setItem('kiko_host', host);
        localStorage.setItem('kiko_udp_port', String(udpPort));
        localStorage.setItem('kiko_http_port', String(httpPort));
        robotAddress = `${host}:${udpPort}`;

        console.log('Attempting to connect to:', robotAddress, 'http:', httpPort);

        // Quick sanity check – ping HTTP status endpoint
        try {
            const statusUrl = `http://${host}:${httpPort}/status`;
            console.log('Pinging status endpoint:', statusUrl);
            const resp = await fetch(statusUrl).catch(e => { throw e; });
            console.log('Status response:', resp.status, resp.ok);
        } catch (pingErr) {
            console.error('Status ping failed (this is okay if server runs on another subnet):', pingErr);
        }

        const result = await invoke('connect', { address: robotAddress, http_port: httpPort });
        console.log('Connection result:', result);
        
        // Verify connection status
        const isConnected = await invoke('get_connection_status');
        console.log('Connection status verified:', isConnected);
        
        if (isConnected) {
            setConnected(true);
            console.log('Successfully connected to robot');
        } else {
            throw new Error('Connection status check failed');
        }
    } catch (err) {
        console.error('Connection failed:', err);
        alert('Connection failed: ' + err);
        setConnected(false);
    }
}

async function disconnect() {
    try {
        await invoke('disconnect');
        setConnected(false);
    } catch (err) {
        console.error('Disconnect error:', err);
    }
}

function setConnected(isConnected) {
    connected = isConnected;
    
    if (isConnected) {
        statusIndicator.classList.remove('offline');
        connectBtn.textContent = 'Disconnect';
        connectBtn.classList.add('connected');
        
        // Start video stream
        videoStream.src = `http://${host}:${httpPort}/video.mjpeg`;
        videoStream.style.display = 'block';
        videoOffline.style.display = 'none';
        
        // Start odometry polling
        startOdometryPolling();
    } else {
        statusIndicator.classList.add('offline');
        connectBtn.textContent = 'Connect';
        connectBtn.classList.remove('connected');
        
        // Stop video stream
        videoStream.src = '';
        videoStream.style.display = 'none';
        videoOffline.style.display = 'flex';
        
        // Stop odometry polling
        stopOdometryPolling();
        
        // Reset displays
        updateDirection(0, 0);
        updateOdometryDisplay(null);
        currentLeft = 0;
        currentRight = 0;
        
        // Clear key states
        Object.keys(keys).forEach(key => keys[key] = false);
    }
}

// Handle connection button
connectBtn.addEventListener('click', () => {
    if (connected) {
        disconnect();
    } else {
        connect();
    }
});

// Emergency stop on spacebar
document.addEventListener('keydown', async (e) => {
    if (e.key === ' ' && connected) {
        e.preventDefault();
        try {
            await invoke('emergency_stop');
            // Clear all key states
            Object.keys(keys).forEach(key => keys[key] = false);
            currentLeft = 0;
            currentRight = 0;
            updateDirection(0, 0);
        } catch (err) {
            console.error('Emergency stop failed:', err);
        }
    }
});

// Listen for telemetry updates from backend
let telemetryListener = null;
let connectionLostListener = null;
let connectionErrorListener = null;
let odometryInterval = null;

async function setupEventListeners() {
    // Telemetry updates
    telemetryListener = await listen('telemetry-update', (event) => {
        const update = event.payload;
        console.log('Telemetry update:', update);
        
        // Update telemetry displays
        document.getElementById('latency').textContent = update.latency + 'ms';
        document.getElementById('sequence').textContent = ++sequenceNum;
        document.getElementById('battery').textContent = 
            (update.telemetry.battery_mv / 1000).toFixed(1) + 'V';
        
        // Update direction based on commanded values
        updateDirection(update.left_command, update.right_command);
    });
    
    // Connection lost
    connectionLostListener = await listen('connection-lost', () => {
        console.error('Connection lost event received');
        setConnected(false);
        alert('Connection to robot lost');
    });
    
    // Connection errors
    connectionErrorListener = await listen('connection-error', (event) => {
        console.error('Connection error:', event.payload);
        alert('Connection error: ' + event.payload);
    });
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard loaded – JS initialised');
    // Populate connection inputs and wire persistence
    if (hostInput) hostInput.value = host;
    if (udpPortInput) udpPortInput.value = isFinite(udpPort) ? udpPort : 8080;
    if (httpPortInput) httpPortInput.value = isFinite(httpPort) ? httpPort : 3030;

    hostInput?.addEventListener('change', () => {
        host = hostInput.value.trim();
        localStorage.setItem('kiko_host', host);
    });
    udpPortInput?.addEventListener('change', () => {
        const v = parseInt(udpPortInput.value, 10);
        udpPort = isFinite(v) && v > 0 ? v : 8080;
        udpPortInput.value = udpPort;
        localStorage.setItem('kiko_udp_port', String(udpPort));
    });
    httpPortInput?.addEventListener('change', () => {
        const v = parseInt(httpPortInput.value, 10);
        httpPort = isFinite(v) && v > 0 ? v : 3030;
        httpPortInput.value = httpPort;
        localStorage.setItem('kiko_http_port', String(httpPort));
        if (connected) {
            videoStream.src = `http://${host}:${httpPort}/video.mjpeg`;
        }
    });

    setupEventListeners();
    setConnected(false);
    updateDirection(0, 0);
    
    // Handle window focus/blur to reset key states
    window.addEventListener('blur', () => {
        Object.keys(keys).forEach(key => keys[key] = false);
        if (connected) {
            setMotorSpeeds(0, 0);
        }
    });
    // Responsive canvas sizing
    resizeOdometryCanvas();
    window.addEventListener('resize', resizeOdometryCanvas);
});

// Handle video feed selection
document.querySelectorAll('.feed-option').forEach(option => {
    option.addEventListener('click', (e) => {
        if (e.target.classList.contains('disabled')) return;
        
        document.querySelectorAll('.feed-option').forEach(o => 
            o.classList.remove('active'));
        e.target.classList.add('active');
        
        // In future, this would switch camera feeds
        console.log('Selected feed:', e.target.dataset.feed);
    });
});

// Odometry polling functions
function startOdometryPolling() {
    console.log('Starting odometry polling');
    stopOdometryPolling(); // Clear any existing interval
    
    odometryInterval = setInterval(async () => {
        try {
            const odometry = await invoke('get_odometry');
            updateOdometryDisplay(odometry);
        } catch (err) {
            console.error('Failed to get odometry:', err);
        }
    }, 100); // 10Hz polling for smoother updates
}

function stopOdometryPolling() {
    if (odometryInterval) {
        clearInterval(odometryInterval);
        odometryInterval = null;
        console.log('Stopped odometry polling');
    }
}

// ==================== REAL-TIME PATH VISUALIZATION SYSTEM ====================

class RobotPathVisualizer {
    constructor(canvasId, infoId, placeholderId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.infoDiv = document.getElementById(infoId);
        this.placeholder = document.getElementById(placeholderId);
        
        // Robot physical parameters
        this.WHEEL_BASE = 0.15; // meters (distance between wheels)
        this.TICKS_PER_METER = 1000; // encoder ticks per meter (configurable)
        
        // Robot state
        this.position = {x: 0, y: 0}; // meters
        this.heading = 0; // radians
        this.totalDistance = 0; // meters
        this.lastOdometry = null;
        this.isInitialized = false;
        
        // Path history
        this.pathHistory = [];
        this.maxPathPoints = 2000; // Limit memory usage
        this.showTrail = true;
        
        // Visualization parameters
        this.scale = 50; // pixels per meter
        this.centerOffset = {x: 209, y: 126}; // Canvas center
        this.panOffset = {x: 0, y: 0}; // Pan offset
        this.minScale = 10;
        this.maxScale = 500;
        
        // Interaction state
        this.isDragging = false;
        this.lastMousePos = {x: 0, y: 0};
        
        // Animation
        this.lastUpdateTime = 0;
        this.targetPosition = {x: 0, y: 0};
        this.targetHeading = 0;
        
        this.setupEventListeners();
        this.startRenderLoop();
        
        console.log('🤖 Path Visualizer initialized');
    }
    
    setupEventListeners() {
        // Mouse interactions
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this.handleWheel(e));
        
        // Control buttons
        document.getElementById('resetPath').addEventListener('click', () => this.resetPath());
        document.getElementById('centerView').addEventListener('click', () => this.centerView());
        document.getElementById('toggleTrail').addEventListener('click', () => this.toggleTrail());
        
        // Prevent context menu
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }
    
    // ==================== ODOMETRY CALCULATIONS ====================
    
    updateOdometry(odometry) {
        if (!odometry) {
            this.showPlaceholder(true);
            return;
        }
        
        this.showPlaceholder(false);
        
        if (!this.isInitialized) {
            this.initializeOdometry(odometry);
            return;
        }
        
        const dt = (odometry.timestamp_ms - this.lastOdometry.timestamp_ms) / 1000.0; // seconds
        if (dt <= 0 || dt > 1.0) { // Skip invalid time deltas
            this.lastOdometry = odometry;
            return;
        }
        
        // Calculate wheel distances (delta)
        const leftDelta = (odometry.left_ticks - this.lastOdometry.left_ticks) / this.TICKS_PER_METER;
        const rightDelta = (odometry.right_ticks - this.lastOdometry.right_ticks) / this.TICKS_PER_METER;
        
        // Differential drive kinematics
        const distance = (leftDelta + rightDelta) / 2.0; // forward distance
        const deltaHeading = (rightDelta - leftDelta) / this.WHEEL_BASE; // heading change
        
        // Update robot state with smooth interpolation
        const newX = this.position.x + distance * Math.cos(this.heading + deltaHeading / 2.0);
        const newY = this.position.y + distance * Math.sin(this.heading + deltaHeading / 2.0);
        const newHeading = this.heading + deltaHeading;
        
        // Smooth interpolation for animation
        this.targetPosition = {x: newX, y: newY};
        this.targetHeading = newHeading;
        
        this.totalDistance += Math.abs(distance);
        
        // Add to path history (with decimation for performance)
        if (this.pathHistory.length === 0 || 
            Math.abs(newX - this.pathHistory[this.pathHistory.length - 1].x) > 0.01 ||
            Math.abs(newY - this.pathHistory[this.pathHistory.length - 1].y) > 0.01) {
            
            this.pathHistory.push({
                x: newX, 
                y: newY, 
                timestamp: odometry.timestamp_ms,
                heading: newHeading
            });
            
            // Manage memory
            if (this.pathHistory.length > this.maxPathPoints) {
                this.pathHistory.shift();
            }
        }
        
        this.lastOdometry = odometry;
        this.updateInfoDisplay(odometry, dt);
    }
    
    initializeOdometry(odometry) {
        this.lastOdometry = odometry;
        this.position = {x: 0, y: 0};
        this.heading = 0;
        this.totalDistance = 0;
        this.pathHistory = [{x: 0, y: 0, timestamp: odometry.timestamp_ms, heading: 0}];
        this.targetPosition = {x: 0, y: 0};
        this.targetHeading = 0;
        this.isInitialized = true;
        console.log('🎯 Odometry initialized');
    }
    
    // ==================== VISUALIZATION ====================
    
    startRenderLoop() {
        const animate = (currentTime) => {
            this.render(currentTime);
            requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
    }
    
    render(currentTime) {
        const dt = currentTime - this.lastUpdateTime;
        this.lastUpdateTime = currentTime;
        
        // Smooth interpolation (60fps)
        const alpha = Math.min(dt / 16.67, 1.0); // 16.67ms = 60fps
        this.position.x += (this.targetPosition.x - this.position.x) * alpha * 0.3;
        this.position.y += (this.targetPosition.y - this.position.y) * alpha * 0.3;
        this.heading += this.normalizeAngle(this.targetHeading - this.heading) * alpha * 0.3;
        
        this.clearCanvas();
        this.drawGrid();
        this.drawPath();
        this.drawRobot();
        this.drawOrigin();
    }
    
    clearCanvas() {
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    drawGrid() {
        const ctx = this.ctx;
        ctx.strokeStyle = '#f0f0f0';
        ctx.lineWidth = 1;
        
        const gridSize = this.scale; // 1 meter grid
        const startX = (-this.panOffset.x % gridSize);
        const startY = (-this.panOffset.y % gridSize);
        
        // Vertical lines
        for (let x = startX; x < this.canvas.width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.canvas.height);
            ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = startY; y < this.canvas.height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.canvas.width, y);
            ctx.stroke();
        }
    }
    
    drawPath() {
        if (!this.showTrail || this.pathHistory.length < 2) return;
        
        const ctx = this.ctx;
        ctx.strokeStyle = '#2196f3';
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // Draw path with gradient opacity
        ctx.beginPath();
        for (let i = 1; i < this.pathHistory.length; i++) {
            const point = this.pathHistory[i];
            const screenPos = this.worldToScreen(point.x, point.y);
            
            if (i === 1) {
                ctx.moveTo(screenPos.x, screenPos.y);
            } else {
                ctx.lineTo(screenPos.x, screenPos.y);
            }
        }
        ctx.stroke();
        
        // Draw path points (recent ones more opaque)
        for (let i = Math.max(0, this.pathHistory.length - 50); i < this.pathHistory.length; i++) {
            const point = this.pathHistory[i];
            const screenPos = this.worldToScreen(point.x, point.y);
            const age = (this.pathHistory.length - i) / 50.0;
            
            ctx.fillStyle = `rgba(33, 150, 243, ${0.8 - age * 0.6})`;
            ctx.beginPath();
            ctx.arc(screenPos.x, screenPos.y, 2, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    
    drawRobot() {
        const screenPos = this.worldToScreen(this.position.x, this.position.y);
        const ctx = this.ctx;
        
        ctx.save();
        ctx.translate(screenPos.x, screenPos.y);
        ctx.rotate(this.heading);
        
        // Robot body (rectangle)
        const width = this.scale * 0.08; // 8cm width
        const height = this.scale * 0.12; // 12cm length
        
        ctx.fillStyle = '#4caf50';
        ctx.strokeStyle = '#2e7d32';
        ctx.lineWidth = 2;
        ctx.fillRect(-width/2, -height/2, width, height);
        ctx.strokeRect(-width/2, -height/2, width, height);
        
        // Direction indicator (triangle)
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.moveTo(height/3, 0);
        ctx.lineTo(-height/6, -width/4);
        ctx.lineTo(-height/6, width/4);
        ctx.closePath();
        ctx.fill();
        
        // Wheels
        ctx.fillStyle = '#333';
        const wheelWidth = this.scale * 0.02;
        const wheelHeight = this.scale * 0.04;
        const wheelOffset = this.scale * this.WHEEL_BASE / 2;
        
        // Left wheel
        ctx.fillRect(-wheelWidth/2, wheelOffset - wheelHeight/2, wheelWidth, wheelHeight);
        // Right wheel
        ctx.fillRect(-wheelWidth/2, -wheelOffset - wheelHeight/2, wheelWidth, wheelHeight);
        
        ctx.restore();
    }
    
    drawOrigin() {
        const originScreen = this.worldToScreen(0, 0);
        const ctx = this.ctx;
        
        // Origin cross
        ctx.strokeStyle = '#f44336';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(originScreen.x - 8, originScreen.y);
        ctx.lineTo(originScreen.x + 8, originScreen.y);
        ctx.moveTo(originScreen.x, originScreen.y - 8);
        ctx.lineTo(originScreen.x, originScreen.y + 8);
        ctx.stroke();
        
        // Origin label
        ctx.fillStyle = '#f44336';
        ctx.font = '10px Consolas';
        ctx.fillText('(0,0)', originScreen.x + 12, originScreen.y - 4);
    }
    
    // ==================== COORDINATE TRANSFORMS ====================
    
    worldToScreen(worldX, worldY) {
        return {
            x: (worldX * this.scale) + this.centerOffset.x + this.panOffset.x,
            y: (-worldY * this.scale) + this.centerOffset.y + this.panOffset.y // Flip Y
        };
    }
    
    screenToWorld(screenX, screenY) {
        return {
            x: (screenX - this.centerOffset.x - this.panOffset.x) / this.scale,
            y: -((screenY - this.centerOffset.y - this.panOffset.y) / this.scale) // Flip Y
        };
    }
    
    // ==================== INTERACTION HANDLERS ====================
    
    handleMouseDown(e) {
        this.isDragging = true;
        this.lastMousePos = {x: e.offsetX, y: e.offsetY};
        this.canvas.style.cursor = 'grabbing';
    }
    
    handleMouseMove(e) {
        if (this.isDragging) {
            const dx = e.offsetX - this.lastMousePos.x;
            const dy = e.offsetY - this.lastMousePos.y;
            this.panOffset.x += dx;
            this.panOffset.y += dy;
            this.lastMousePos = {x: e.offsetX, y: e.offsetY};
        }
    }
    
    handleMouseUp(e) {
        this.isDragging = false;
        this.canvas.style.cursor = 'grab';
    }
    
    handleWheel(e) {
        e.preventDefault();
        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * zoomFactor));
        
        // Zoom towards mouse position
        const mouseWorld = this.screenToWorld(e.offsetX, e.offsetY);
        this.scale = newScale;
        const newMouseScreen = this.worldToScreen(mouseWorld.x, mouseWorld.y);
        
        this.panOffset.x += e.offsetX - newMouseScreen.x;
        this.panOffset.y += e.offsetY - newMouseScreen.y;
    }
    
    // ==================== CONTROL FUNCTIONS ====================
    
    resetPath() {
        this.position = {x: 0, y: 0};
        this.heading = 0;
        this.totalDistance = 0;
        this.pathHistory = [{x: 0, y: 0, timestamp: Date.now(), heading: 0}];
        this.targetPosition = {x: 0, y: 0};
        this.targetHeading = 0;
        this.centerView();
        console.log('🔄 Path reset');
    }
    
    centerView() {
        this.panOffset = {x: 0, y: 0};
        this.scale = 50;
        console.log('🎯 View centered');
    }
    
    toggleTrail() {
        this.showTrail = !this.showTrail;
        const btn = document.getElementById('toggleTrail');
        btn.style.background = this.showTrail ? '#777' : '#333';
        console.log(`👁️ Trail ${this.showTrail ? 'enabled' : 'disabled'}`);
    }
    
    // ==================== UI UPDATES ====================
    
    updateInfoDisplay(odometry, dt) {
        const speed = this.calculateSpeed(odometry, dt);
        const headingDeg = (this.heading * 180 / Math.PI) % 360;
        
        this.infoDiv.innerHTML = `
            <div>Position: (${this.position.x.toFixed(2)}, ${this.position.y.toFixed(2)}) m</div>
            <div>Heading: ${headingDeg.toFixed(1)}°</div>
            <div>Speed: ${speed.toFixed(2)} m/s</div>
            <div>Distance: ${this.totalDistance.toFixed(2)} m</div>
        `;
    }
    
    calculateSpeed(odometry, dt) {
        if (!this.lastOdometry || dt <= 0) return 0;
        
        const leftDelta = (odometry.left_ticks - this.lastOdometry.left_ticks) / this.TICKS_PER_METER;
        const rightDelta = (odometry.right_ticks - this.lastOdometry.right_ticks) / this.TICKS_PER_METER;
        const distance = (leftDelta + rightDelta) / 2.0;
        
        return Math.abs(distance / dt);
    }
    
    showPlaceholder(show) {
        this.placeholder.style.display = show ? 'block' : 'none';
        this.infoDiv.style.display = show ? 'none' : 'block';
    }
    
    // ==================== UTILITY FUNCTIONS ====================
    
    normalizeAngle(angle) {
        while (angle > Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }
}

// Global path visualizer instance
let pathVisualizer = null;

function updateOdometryDisplay(odometry) {
    if (!pathVisualizer) {
        pathVisualizer = new RobotPathVisualizer('odometryCanvas', 'odometryInfo', 'odometryPlaceholder');
        const c = document.getElementById('odometryCanvas');
        if (c) {
            pathVisualizer.centerOffset = { x: c.width / 2, y: c.height / 2 };
        }
    }
    pathVisualizer.updateOdometry(odometry);
}

// Cleanup on window close
window.addEventListener('beforeunload', () => {
    if (connected) {
        disconnect();
    }
    stopOdometryPolling();
    if (telemetryListener) telemetryListener();
    if (connectionLostListener) connectionLostListener();
    if (connectionErrorListener) connectionErrorListener();
});

// Resize odometry canvas to its container and keep visualizer centered
function resizeOdometryCanvas() {
    const canvas = document.getElementById('odometryCanvas');
    if (!canvas) return;
    const container = canvas.parentElement;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const newWidth = Math.max(200, Math.floor(rect.width));
    const newHeight = Math.max(150, Math.floor(rect.height));
    if (canvas.width !== newWidth || canvas.height !== newHeight) {
        canvas.width = newWidth;
        canvas.height = newHeight;
        if (pathVisualizer) {
            pathVisualizer.centerOffset = { x: newWidth / 2, y: newHeight / 2 };
        }
    }
}