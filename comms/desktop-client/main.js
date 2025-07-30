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
let robotAddress = '10.42.200.50'; // Default address
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
        if (err.includes('Not connected')) {
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
        // Prompt for address if needed
        const input = prompt('Enter robot address:', robotAddress);
        if (input) {
            robotAddress = input;
        }
        
        console.log('Attempting to connect to:', robotAddress);

        // Quick sanity check – ping HTTP status endpoint
        try {
            const statusUrl = `http://${robotAddress.split(':')[0]}:3030/status`;
            console.log('Pinging status endpoint:', statusUrl);
            const resp = await fetch(statusUrl).catch(e => { throw e; });
            console.log('Status response:', resp.status, resp.ok);
        } catch (pingErr) {
            console.error('Status ping failed (this is okay if server runs on another subnet):', pingErr);
        }

        const result = await invoke('connect', { address: robotAddress });
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
        connectionText.textContent = 'Online';
        connectBtn.textContent = 'Disconnect';
        connectBtn.classList.add('connected');
        
        // Start video stream
        videoStream.src = `http://${robotAddress.split(':')[0]}:3030/video.mjpeg`;
        videoStream.style.display = 'block';
        videoOffline.style.display = 'none';
    } else {
        statusIndicator.classList.add('offline');
        connectionText.textContent = 'Offline';
        connectBtn.textContent = 'Connect';
        connectBtn.classList.remove('connected');
        
        // Stop video stream
        videoStream.src = '';
        videoStream.style.display = 'none';
        videoOffline.style.display = 'flex';
        
        // Reset displays
        updateDirection(0, 0);
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

// Cleanup on window close
window.addEventListener('beforeunload', () => {
    if (connected) {
        disconnect();
    }
    if (telemetryListener) telemetryListener();
    if (connectionLostListener) connectionLostListener();
    if (connectionErrorListener) connectionErrorListener();
});