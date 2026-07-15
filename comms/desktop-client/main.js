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
let activeStreamGenerationDecimal = null;
let desiredMotorPwm = { left: 0, right: 0 };
let submittedDesiredMotorPwm = desiredMotorPwm;
let motorPwmDrain = null;
let acknowledgedStop = null;
let motorLeaseRefreshTimer = null;
let motorLeaseRefreshGeneration = 0;
let connectionTransitionInProgress = false;
const pendingConnectionFailures = new Map();
// Connection settings with persistence
const DEFAULT_HOST = '10.42.200.50';
const DEFAULT_UDP_PORT = 8080;
const DEFAULT_HTTP_PORT = 3030;

function parsePort(rawValue, fieldName) {
    const value = String(rawValue).trim();
    if (!/^\d+$/.test(value)) {
        throw new Error(`${fieldName} must be an integer from 1 to 65535`);
    }
    const port = Number(value);
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
        throw new Error(`${fieldName} must be an integer from 1 to 65535`);
    }
    return port;
}

function readStoredSetting(key) {
    try {
        return localStorage.getItem(key);
    } catch (error) {
        console.warn(`Unable to read persisted setting ${key}:`, error);
        return null;
    }
}

function storeSetting(key, value) {
    try {
        localStorage.setItem(key, value);
    } catch (error) {
        console.warn(`Unable to persist setting ${key}:`, error);
    }
}

function loadStoredPort(key, fallback, fieldName) {
    const stored = readStoredSetting(key);
    if (stored === null) return fallback;
    try {
        return parsePort(stored, fieldName);
    } catch (error) {
        console.warn(`Ignoring invalid stored ${fieldName}:`, error);
        return fallback;
    }
}

function requireHost(rawValue) {
    const value = String(rawValue).trim();
    if (!value) throw new Error('Host must not be empty');
    if (/\s/.test(value) || value.includes('://') || /[/?#]/.test(value)) {
        throw new Error('Host must be a hostname or IP address without a URL scheme, path, query, or fragment');
    }
    const startsBracket = value.startsWith('[');
    const endsBracket = value.endsWith(']');
    if (startsBracket !== endsBracket || /[\[\]]/.test(value.slice(1, -1))) {
        throw new Error('IPv6 host brackets must form one outer pair');
    }
    if (startsBracket && !value.slice(1, -1).includes(':')) {
        throw new Error('Brackets are accepted only around an IPv6 address');
    }
    return value;
}

function loadStoredHost() {
    const stored = readStoredSetting('kiko_host');
    if (stored === null) return DEFAULT_HOST;
    try {
        return requireHost(stored);
    } catch (error) {
        console.warn('Ignoring invalid stored host:', error);
        return DEFAULT_HOST;
    }
}

function udpAddress(hostValue, port) {
    const unbracketed = hostValue.startsWith('[') && hostValue.endsWith(']')
        ? hostValue.slice(1, -1)
        : hostValue;
    return unbracketed.includes(':')
        ? `[${unbracketed}]:${port}`
        : `${unbracketed}:${port}`;
}

function httpOrigin(hostValue, port) {
    return `http://${udpAddress(hostValue, port)}`;
}

let host = loadStoredHost();
let udpPort = loadStoredPort('kiko_udp_port', DEFAULT_UDP_PORT, 'UDP port');
let httpPort = loadStoredPort('kiko_http_port', DEFAULT_HTTP_PORT, 'HTTP port');
let robotAddress = udpAddress(host, udpPort);
let basePwmPercent = 50;

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
const directionArrow = document.getElementById('directionArrow');
const connectBtn = document.getElementById('connectBtn');
connectBtn.disabled = true;
const hostInput = document.getElementById('hostInput');
const udpPortInput = document.getElementById('udpPortInput');
const httpPortInput = document.getElementById('httpPortInput');
const connectionSettingInputs = [hostInput, udpPortInput, httpPortInput].filter(Boolean);

function updateConnectionSettingAvailability() {
    const disabled = connected || connectionTransitionInProgress;
    connectionSettingInputs.forEach(input => {
        input.disabled = disabled;
    });
}
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
        updateMotorPwm();
    }
});

document.addEventListener('keyup', (e) => {
    if (e.key in keys) {
        keys[e.key] = false;
        updateMotorPwm();
    }
});

// Prevent arrow keys from scrolling the page
window.addEventListener('keydown', (e) => {
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {
        e.preventDefault();
    }
});

function updateMotorPwm() {
    if (!connected) return;
    
    const boostedPwmPercentCappedAtDomainMaximum = Math.min(
        100,
        Math.floor(basePwmPercent * 1.5)
    );
    const pwmMagnitude = keys.Shift
        ? boostedPwmPercentCappedAtDomainMaximum
        : basePwmPercent;
    
    let left = 0;
    let right = 0;
    
    const forward = keys.ArrowUp || keys.w;
    const backward = keys.ArrowDown || keys.s;
    const turnLeft = keys.ArrowLeft || keys.a;
    const turnRight = keys.ArrowRight || keys.d;
    
    if (forward && !backward) {
        left = pwmMagnitude;
        right = pwmMagnitude;
        
        if (turnLeft && !turnRight) {
            left = Math.floor(pwmMagnitude * 0.3);
        } else if (turnRight && !turnLeft) {
            right = Math.floor(pwmMagnitude * 0.3);
        }
    } else if (backward && !forward) {
        left = -pwmMagnitude;
        right = -pwmMagnitude;
        
        if (turnLeft && !turnRight) {
            left = -Math.floor(pwmMagnitude * 0.3);
        } else if (turnRight && !turnLeft) {
            right = -Math.floor(pwmMagnitude * 0.3);
        }
    } else if (turnLeft && !turnRight) {
        left = -Math.floor(pwmMagnitude / 2);
        right = Math.floor(pwmMagnitude / 2);
    } else if (turnRight && !turnLeft) {
        left = Math.floor(pwmMagnitude / 2);
        right = -Math.floor(pwmMagnitude / 2);
    }
    
    // Send to backend
    queueMotorPwm(left, right);
}

function sameMotorPwm(first, second) {
    return first.left === second.left && first.right === second.right;
}

function stopIsInProgressForGeneration(streamGenerationDecimal) {
    return streamGenerationDecimal !== null &&
        acknowledgedStop?.streamGenerationDecimal === streamGenerationDecimal;
}

function queueMotorPwm(left, right, forceRefresh = false) {
    if (connected && activeStreamGenerationDecimal === null) {
        console.error('Connected state has no stream generation');
        alert('Motor command stream is internally inconsistent and has been closed');
        setConnected(false);
        return Promise.resolve();
    }
    const next = stopIsInProgressForGeneration(activeStreamGenerationDecimal)
        ? { left: 0, right: 0 }
        : { left, right };
    if (forceRefresh || !sameMotorPwm(desiredMotorPwm, next)) {
        desiredMotorPwm = next;
    }

    if (motorPwmDrain === null && connected && submittedDesiredMotorPwm !== desiredMotorPwm) {
        const generation = activeStreamGenerationDecimal;
        const drain = (async () => {
            while (
                connected &&
                generation === activeStreamGenerationDecimal &&
                submittedDesiredMotorPwm !== desiredMotorPwm
            ) {
                const requested = desiredMotorPwm;
                await invoke('set_motor_pwm', {
                    left: requested.left,
                    right: requested.right,
                    stream_generation_decimal: generation
                });
                if (!connected || generation !== activeStreamGenerationDecimal) return;
                submittedDesiredMotorPwm = requested;
                if (Math.abs(requested.left) > 50 || Math.abs(requested.right) > 50) {
                    console.log('High PWM command:', requested);
                }
            }
        })();
        motorPwmDrain = drain;
        drain.then(
            () => {
                if (motorPwmDrain === drain) motorPwmDrain = null;
                resumeCurrentMotorPwmDrain();
            },
            err => {
                if (motorPwmDrain === drain) motorPwmDrain = null;
                if (generation !== activeStreamGenerationDecimal) {
                    console.log(
                        `Ignored desired-PWM failure from superseded stream generation ${generation}:`,
                        err
                    );
                    resumeCurrentMotorPwmDrain();
                    return;
                }
                console.error('Failed to update desired motor PWM:', err);
                if (!stopIsInProgressForGeneration(generation)) {
                    alert('Motor PWM update failed: ' + err);
                }
                setConnected(false);
            }
        );
    }

    return motorPwmDrain ?? Promise.resolve();
}

function resumeCurrentMotorPwmDrain() {
    if (connected && submittedDesiredMotorPwm !== desiredMotorPwm) {
        queueMotorPwm(desiredMotorPwm.left, desiredMotorPwm.right);
    }
}

function requestAcknowledgedZeroPwm() {
    if (!connected || activeStreamGenerationDecimal === null) {
        return Promise.reject(new Error('Not connected'));
    }
    const streamGenerationDecimal = activeStreamGenerationDecimal;
    if (stopIsInProgressForGeneration(streamGenerationDecimal)) {
        return acknowledgedStop.promise;
    }

    Object.keys(keys).forEach(key => keys[key] = false);
    const request = (async () => {
        await queueMotorPwm(0, 0);
        if (!connected || activeStreamGenerationDecimal !== streamGenerationDecimal) {
            throw new Error(
                `Zero-PWM stop for superseded stream generation ${streamGenerationDecimal} was not sent to the replacement stream`
            );
        }
        await invoke('stop_motors', { stream_generation_decimal: streamGenerationDecimal });
        if (activeStreamGenerationDecimal === streamGenerationDecimal) {
            updateDirection(0, 0);
        }
    })();
    const stop = { streamGenerationDecimal, promise: request };
    acknowledgedStop = stop;
    request.then(
        () => {
            if (acknowledgedStop === stop) acknowledgedStop = null;
        },
        () => {
            if (acknowledgedStop === stop) acknowledgedStop = null;
        }
    );
    return request;
}

function resetMotorPwmState() {
    const zero = { left: 0, right: 0 };
    desiredMotorPwm = zero;
    submittedDesiredMotorPwm = zero;
}

function startMotorLeaseRefresh() {
    stopMotorLeaseRefresh();
    const refreshGeneration = motorLeaseRefreshGeneration;

    const refresh = async () => {
        if (!connected || refreshGeneration !== motorLeaseRefreshGeneration) return;
        if (
            !stopIsInProgressForGeneration(activeStreamGenerationDecimal) &&
            (desiredMotorPwm.left !== 0 || desiredMotorPwm.right !== 0)
        ) {
            try {
                await queueMotorPwm(desiredMotorPwm.left, desiredMotorPwm.right, true);
            } catch (_) {
                // The queue's single rejection handler reports the exact failure.
            }
        }
        if (connected && refreshGeneration === motorLeaseRefreshGeneration) {
            motorLeaseRefreshTimer = setTimeout(refresh, 50);
        }
    };

    motorLeaseRefreshTimer = setTimeout(refresh, 50);
}

function stopMotorLeaseRefresh() {
    motorLeaseRefreshGeneration += 1;
    if (motorLeaseRefreshTimer !== null) {
        clearTimeout(motorLeaseRefreshTimer);
        motorLeaseRefreshTimer = null;
    }
}

function reportStopFailure(context, err, streamGenerationDecimal) {
    if (streamGenerationDecimal !== activeStreamGenerationDecimal) {
        console.log(
            `Ignored ${context.toLowerCase()} from superseded stream generation ${streamGenerationDecimal}:`,
            err
        );
        return;
    }
    console.error(`${context}:`, err);
    alert(`${context}: ${err}`);
    setConnected(false);
}

function rejectMovementWhileStopping(event) {
    if (stopIsInProgressForGeneration(activeStreamGenerationDecimal) && event.key in keys) {
        event.preventDefault();
        event.stopImmediatePropagation();
        return true;
    }
    return false;
}

// Ignore movement input until an in-flight acknowledged stop has completed.
document.addEventListener('keydown', event => {
    if (rejectMovementWhileStopping(event)) {
        keys[event.key] = false;
    }
}, true);

document.addEventListener('keyup', event => {
    if (rejectMovementWhileStopping(event)) {
        keys[event.key] = false;
    }
}, true);

// Direction visualization
function updateDirection(left, right) {
    const leftElem = document.getElementById('leftPwm');
    const rightElem = document.getElementById('rightPwm');
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
    if (connectionTransitionInProgress) return;
    connectionTransitionInProgress = true;
    connectBtn.disabled = true;
    updateConnectionSettingAvailability();
    try {
        // Build address from inputs and persist
        host = requireHost(hostInput?.value ?? host);
        udpPort = parsePort(udpPortInput?.value ?? udpPort, 'UDP port');
        httpPort = parsePort(httpPortInput?.value ?? httpPort, 'HTTP port');
        storeSetting('kiko_host', host);
        storeSetting('kiko_udp_port', String(udpPort));
        storeSetting('kiko_http_port', String(httpPort));
        robotAddress = udpAddress(host, udpPort);

        console.log('Attempting to connect to:', robotAddress, 'http:', httpPort);

        const connection = await invoke('connect', { address: robotAddress, http_port: httpPort });
        if (
            !connection ||
            connection.server_addr !== robotAddress ||
            !/^\d+$/.test(connection.stream_generation_decimal)
        ) {
            throw new Error('Connection response does not contain the expected address and stream generation');
        }
        const earlyFailure = pendingConnectionFailures.get(connection.stream_generation_decimal);
        pendingConnectionFailures.delete(connection.stream_generation_decimal);
        if (earlyFailure !== undefined) {
            throw new Error(`Command stream failed during connection: ${earlyFailure}`);
        }
        activeStreamGenerationDecimal = connection.stream_generation_decimal;
        resetMotorPwmState();
        setConnected(true);
        console.log('Successfully connected to robot stream generation:', activeStreamGenerationDecimal);
    } catch (err) {
        console.error('Connection failed:', err);
        alert('Connection failed: ' + err);
        setConnected(false);
    } finally {
        connectionTransitionInProgress = false;
        connectBtn.disabled = false;
        updateConnectionSettingAvailability();
    }
}

async function disconnect() {
    if (connectionTransitionInProgress) return;
    connectionTransitionInProgress = true;
    connectBtn.disabled = true;
    updateConnectionSettingAvailability();
    const generation = activeStreamGenerationDecimal;
    try {
        if (generation !== null) {
            await invoke('disconnect', { stream_generation_decimal: generation });
        }
    } catch (err) {
        console.error('Disconnect error:', err);
        alert('Disconnected locally with warning: ' + err);
    } finally {
        setConnected(false);
        connectionTransitionInProgress = false;
        connectBtn.disabled = false;
        updateConnectionSettingAvailability();
    }
}

function setConnected(isConnected) {
    connected = isConnected;
    
    if (isConnected) {
        statusIndicator.classList.remove('offline');
        connectBtn.textContent = 'Disconnect';
        connectBtn.classList.add('connected');
        startMotorLeaseRefresh();
        
        // Start video stream
        videoStream.src = `${httpOrigin(host, httpPort)}/video.mjpeg`;
        videoStream.style.display = 'block';
        videoOffline.style.display = 'none';
        
        // Start odometry polling
        startOdometryPolling();
    } else {
        stopMotorLeaseRefresh();
        activeStreamGenerationDecimal = null;
        acknowledgedStop = null;
        resetMotorPwmState();
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
        
        // Clear key states
        Object.keys(keys).forEach(key => keys[key] = false);
    }
    updateConnectionSettingAvailability();
}

// Handle connection button
connectBtn.addEventListener('click', () => {
    if (connected) {
        disconnect();
    } else {
        connect();
    }
});

// Send a zero-PWM stop command on spacebar.
document.addEventListener('keydown', async (e) => {
    if (e.key === ' ' && connected) {
        e.preventDefault();
        const streamGenerationDecimal = activeStreamGenerationDecimal;
        try {
            await requestAcknowledgedZeroPwm();
        } catch (err) {
            reportStopFailure('Zero-PWM stop failed', err, streamGenerationDecimal);
        }
    }
});

let commandAcknowledgementListener = null;
let connectionLostListener = null;
let connectionErrorListener = null;
let odometryPollTimer = null;
let odometryPollGeneration = 0;

async function setupEventListeners() {
    commandAcknowledgementListener = await listen('command-acknowledgement', (event) => {
        const update = event.payload;
        if (
            !update ||
            typeof update.stream_generation_decimal !== 'string' ||
            !/^\d+$/.test(update.stream_generation_decimal)
        ) {
            console.error('Rejected malformed command acknowledgement event:', update);
            if (connected) {
                alert('The command stream emitted an invalid acknowledgement and has been closed');
                setConnected(false);
            }
            return;
        }
        if (update.stream_generation_decimal !== activeStreamGenerationDecimal) {
            console.log('Ignored acknowledgement from a superseded command stream:', update);
            return;
        }
        console.log('Command acknowledgement:', update);
        
        document.getElementById('latency').textContent = update.round_trip_latency_ms + 'ms';
        document.getElementById('sequence').textContent = update.accepted_sequence;
        
        updateDirection(
            update.commanded_left_pwm_percent,
            update.commanded_right_pwm_percent
        );
    });
    
    // Connection lost
    connectionLostListener = await listen('connection-lost', (event) => {
        const failure = event.payload;
        if (!recordConnectionFailure(failure)) return;
        if (failure.stream_generation_decimal === activeStreamGenerationDecimal) {
            console.error('Connection lost event received:', failure.message);
            setConnected(false);
        }
    });
    
    // Connection errors
    connectionErrorListener = await listen('connection-error', (event) => {
        const failure = event.payload;
        if (!recordConnectionFailure(failure)) return;
        if (failure.stream_generation_decimal === activeStreamGenerationDecimal) {
            console.error('Connection error:', failure.message);
            alert('Connection error: ' + failure.message);
        }
    });
}

function recordConnectionFailure(failure) {
    if (
        !failure ||
        typeof failure.message !== 'string' ||
        typeof failure.stream_generation_decimal !== 'string' ||
        !/^\d+$/.test(failure.stream_generation_decimal)
    ) {
        console.error('Rejected malformed connection failure event:', failure);
        if (connected) {
            alert('The command stream emitted an invalid failure event and has been closed');
            setConnected(false);
        }
        return false;
    }
    pendingConnectionFailures.set(failure.stream_generation_decimal, failure.message);
    while (pendingConnectionFailures.size > 16) {
        pendingConnectionFailures.delete(pendingConnectionFailures.keys().next().value);
    }
    return true;
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard loaded – JS initialised');
    // Populate connection inputs and wire persistence
    if (hostInput) hostInput.value = host;
    if (udpPortInput) udpPortInput.value = udpPort;
    if (httpPortInput) httpPortInput.value = httpPort;

    hostInput?.addEventListener('change', () => {
        try {
            host = requireHost(hostInput.value);
            hostInput.setCustomValidity('');
            storeSetting('kiko_host', host);
        } catch (error) {
            hostInput.setCustomValidity(error.message);
            hostInput.reportValidity();
            hostInput.value = host;
        }
    });
    udpPortInput?.addEventListener('change', () => {
        try {
            udpPort = parsePort(udpPortInput.value, 'UDP port');
            udpPortInput.setCustomValidity('');
            storeSetting('kiko_udp_port', String(udpPort));
        } catch (error) {
            udpPortInput.setCustomValidity(error.message);
            udpPortInput.reportValidity();
            udpPortInput.value = udpPort;
        }
    });
    httpPortInput?.addEventListener('change', () => {
        try {
            httpPort = parsePort(httpPortInput.value, 'HTTP port');
            httpPortInput.setCustomValidity('');
            storeSetting('kiko_http_port', String(httpPort));
            if (connected) {
                videoStream.src = `${httpOrigin(host, httpPort)}/video.mjpeg`;
            }
        } catch (error) {
            httpPortInput.setCustomValidity(error.message);
            httpPortInput.reportValidity();
            httpPortInput.value = httpPort;
        }
    });

    const pwmInput = document.getElementById('pwmInput');
    const pwmText = document.getElementById('pwmText');
    pwmInput?.addEventListener('input', () => {
        const requested = Number(pwmInput.value);
        if (!Number.isInteger(requested) || requested < 10 || requested > 100 || requested % 10 !== 0) {
            console.error('Rejected out-of-domain PWM slider value:', pwmInput.value);
            pwmInput.value = String(basePwmPercent);
            return;
        }
        basePwmPercent = requested;
        pwmText.textContent = `${basePwmPercent}%`;
        updateMotorPwm();
    });

    setupEventListeners()
        .then(() => {
            connectBtn.disabled = false;
        })
        .catch(error => {
            console.error('Failed to register application event listeners:', error);
            alert('Control event listeners could not be registered; connection controls remain disabled: ' + error);
        });
    setConnected(false);
    updateDirection(0, 0);
    
    // Handle window focus/blur to reset key states
    window.addEventListener('blur', () => {
        Object.keys(keys).forEach(key => keys[key] = false);
        if (connected) {
            const streamGenerationDecimal = activeStreamGenerationDecimal;
            requestAcknowledgedZeroPwm().catch(err => {
                reportStopFailure(
                    'Focus-loss zero-PWM stop failed',
                    err,
                    streamGenerationDecimal
                );
            });
        }
    });
});

// Odometry polling functions
function startOdometryPolling() {
    console.log('Starting odometry polling');
    stopOdometryPolling();
    const generation = odometryPollGeneration;
    const streamGenerationDecimal = activeStreamGenerationDecimal;

    const poll = async () => {
        try {
            const odometry = await invoke('get_odometry', {
                stream_generation_decimal: streamGenerationDecimal
            });
            if (connected && generation === odometryPollGeneration) {
                updateOdometryDisplay(odometry);
            }
        } catch (err) {
            if (connected && generation === odometryPollGeneration) {
                console.error('Failed to get odometry:', err);
                showOdometryUnavailable('Odometry request failed', String(err));
            }
        }

        if (connected && generation === odometryPollGeneration) {
            odometryPollTimer = setTimeout(poll, 100);
        }
    };
    poll();
}

function stopOdometryPolling() {
    odometryPollGeneration += 1;
    if (odometryPollTimer) {
        clearTimeout(odometryPollTimer);
        odometryPollTimer = null;
        console.log('Stopped odometry polling');
    }
}

function updateOdometryDisplay(odometry) {
    if (!odometry) {
        showOdometryUnavailable('Waiting for odometry data', '');
        return;
    }

    document.getElementById('leftEncoderTicks').textContent =
        odometry.left_estimated_extended_ticks_wrapping_i64;
    document.getElementById('rightEncoderTicks').textContent =
        odometry.right_estimated_extended_ticks_wrapping_i64;
    document.getElementById('leftSampleDeltaTicks').textContent =
        odometry.left_sample_delta_ticks_modulo_i16;
    document.getElementById('rightSampleDeltaTicks').textContent =
        odometry.right_sample_delta_ticks_modulo_i16;
    document.getElementById('controllerUptimeMs').textContent =
        odometry.controller_uptime_ms_wrapping;
    document.getElementById('odometryServerReceiveAgeMs').textContent =
        odometry.server_receive_age_ms_decimal;
    document.getElementById('odometryValues').hidden = false;
    const placeholder = document.getElementById('odometryPlaceholder');
    placeholder.hidden = true;
    placeholder.title = '';
}

function showOdometryUnavailable(message, detail) {
    document.getElementById('odometryValues').hidden = true;
    const placeholder = document.getElementById('odometryPlaceholder');
    placeholder.hidden = false;
    placeholder.textContent = message;
    placeholder.title = detail;
}

// Cleanup on window close
window.addEventListener('beforeunload', () => {
    if (connected) {
        disconnect();
    }
    stopOdometryPolling();
    if (commandAcknowledgementListener) commandAcknowledgementListener();
    if (connectionLostListener) connectionLostListener();
    if (connectionErrorListener) connectionErrorListener();
});
