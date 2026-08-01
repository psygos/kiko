#!/usr/bin/env python3
"""Kiko face-follow: head looks at + follows the closest person; eyes express.

Single owner of OAK RGB, STS head bus, and KEP2 eye serial for the run.

Behavior contract (operator-specified):
- the head's powered rest pose is always the natural pose;
- tracking applies bounded yaw/curl offsets around natural;
- person lost -> ease back to natural;
- any fault or exit -> return to natural while retaining the established
  four-axis torque hold (the mechanism must never be allowed to collapse).

Stage flags: --no-head (eyes+camera only), --no-eyes (head+camera only).
"""

import argparse
import json
import math
import os
import signal
import struct
import sys
import threading
import time

import cv2
import numpy as np
import serial

from compliant_head import (
    CompliantHeadController, CompliantHeadPolicy, FOLLOWING, RECOVERING,
    RELEASE_DWELL, YIELDING,
)

# ----------------------------------------------------------------------------
# Configuration (operator-reviewed values; signs are flippable via config file)
# ----------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "head_device": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00",
    "eye_device": "/dev/serial/by-id/usb-kiko_kiko-eyes-kep2_98c47919804f9f1aaacfd5fa0a20bf74-if00",
    "eye_uid_hex": "98c47919804f9f1aaacfd5fa0a20bf74",
    "oak_mxid": "19443010F1B43A2E00",
    "camera_width": 640,
    "camera_height": 400,
    "camera_fps": 10,
    "natural_ticks": [2155, 2545, 2943, 2876],  # bow, curl, yaw, roll
    "admission_window_ticks": 420,
    "torque_limit_permille": [650, 550, 400, 400],
    "return_speed_ticks_s": 50,
    "track_speed_min": 30,
    "track_speed_max": 135,
    "return_step_ticks": 2,   # goal advance per 20 Hz cycle => 40 t/s
    "track_step_ticks": 6,    # goal advance per 20 Hz cycle => <=120 t/s
    "goal_deadband_ticks": 3,
    "offset_ease": 0.30,      # exponential ease toward desired offsets
    "bow_limit_ticks": 45,
    "curl_limit_ticks": 90,
    "yaw_limit_ticks": 260,
    "roll_limit_ticks": 70,
    "yaw_ticks_per_rad": 750.0,   # slightly theatrical aim
    "pitch_ticks_per_rad": 330.0,
    "curl_pitch_share": 0.75,
    "bow_pitch_share": 0.15,   # bow fights gravity; keep its duty low (overtemp)
    "derate_temp_raw": 48,     # pitch joints rest at natural above this
    "derate_clear_temp_raw": 42,
    "yaw_sign": 1,                # flip to -1 if head turns away from person
    "curl_sign": 1,               # flip to -1 if head pitches away vertically
    "roll_sign": 1,               # aesthetic tilt direction
    "eye_x_sign": 1,              # flip if eyes mirror instead of follow
    "eye_y_sign": 1,
    "temp_abort_raw": 58,
    "preengage_temp_limit_raw": 55,
    "volt_min_raw": 90,
    "volt_max_raw": 135,
    "head_offset_up_m": 0.18,    # head center above camera origin (operator)
    "head_offset_back_m": 0.15,  # head center behind camera origin (operator)
    "face_width_m": 0.16,        # assumed physical face width for ranging
    "goal_divergence_abort_ticks": 120,
    "person_lost_grace_s": 2.0,
    "greet_cooldown_s": 6.0,
    "sleepy_after_idle_s": 60.0,
    "fx_fallback": 465.0,
}

BOW, CURL, YAW, ROLL = 0, 1, 2, 3
SERVO_IDS = [1, 2, 3, 4]
JOINT_NAMES = ["bow", "curl", "yaw", "roll"]

# ----------------------------------------------------------------------------
# STS servo bus (protocol facts from crates/kiko-head-protocol)
# ----------------------------------------------------------------------------

STS_TORQUE_REG = 40
STS_GOAL_REG = 42
STS_TORQUE_LIMIT_REG = 48
STS_TELEMETRY_REG = 56
STS_TELEMETRY_BYTES = 15


class StsError(Exception):
    pass


class StsBus:
    def __init__(self, device):
        self.port = serial.Serial(
            device, baudrate=1000000, bytesize=8, parity="N", stopbits=1,
            timeout=0.1, write_timeout=0.1, exclusive=True,
        )
        self.port.dtr = False
        self.port.rts = True
        self.lock = threading.Lock()

    @staticmethod
    def _checksum(payload):
        return (~sum(payload)) & 0xFF

    def _transact(self, servo_id, instruction, params, expect_params,
                  expect_response=True):
        frame = bytes([servo_id, len(params) + 2, instruction]) + bytes(params)
        wire = b"\xff\xff" + frame + bytes([self._checksum(frame)])
        with self.lock:
            self.port.reset_input_buffer()
            self.port.write(wire)
            self.port.flush()
            if not expect_response:
                # Servos run response level 0: only READ/PING answer. A write
                # gets host-write completion evidence only, per the Rust owner.
                time.sleep(0.002)
                return b""
            want = 6 + expect_params
            raw = self.port.read(want)
        if len(raw) < 6:
            raise StsError(f"servo {servo_id}: short response {raw.hex()}")
        if raw[0] != 0xFF or raw[1] != 0xFF:
            raise StsError(f"servo {servo_id}: bad header {raw.hex()}")
        if raw[2] != servo_id:
            raise StsError(f"servo {servo_id}: wrong id {raw[2]}")
        if len(raw) != raw[3] + 4:
            raise StsError(f"servo {servo_id}: length mismatch {raw.hex()}")
        if self._checksum(raw[2:-1]) != raw[-1]:
            raise StsError(f"servo {servo_id}: checksum mismatch {raw.hex()}")
        if raw[4] != 0:
            raise StsError(f"servo {servo_id}: device status 0x{raw[4]:02x}")
        return raw[5:-1]

    def read_telemetry(self, servo_id):
        p = self._transact(servo_id, 0x02, [STS_TELEMETRY_REG, STS_TELEMETRY_BYTES],
                           STS_TELEMETRY_BYTES)
        return {
            "position": p[0] | (p[1] << 8),
            "speed_raw": p[2] | (p[3] << 8),
            "load_raw": p[4] | (p[5] << 8),
            "voltage_raw": p[6],
            "temperature_raw": p[7],
            "moving": p[10] != 0,
            "current_raw": p[13] | (p[14] << 8),
        }

    def read_position_redundant(self, servo_id, tolerance=10):
        a = self.read_telemetry(servo_id)
        b = self.read_telemetry(servo_id)
        if abs(a["position"] - b["position"]) > tolerance:
            raise StsError(
                f"servo {servo_id}: redundant read disagrees "
                f"{a['position']} vs {b['position']}")
        return b

    def write_goal(self, servo_id, position, speed):
        position = max(0, min(4095, int(position)))
        speed = max(1, min(32766, int(speed)))
        self._transact(servo_id, 0x03,
                       [STS_GOAL_REG, position & 0xFF, position >> 8, 0, 0,
                        speed & 0xFF, speed >> 8], 0, expect_response=False)

    def write_torque_limit(self, servo_id, permille):
        permille = max(1, min(1000, int(permille)))
        self._transact(servo_id, 0x03,
                       [STS_TORQUE_LIMIT_REG, permille & 0xFF, permille >> 8],
                       0, expect_response=False)

    def write_torque_switch(self, servo_id, enabled):
        self._transact(servo_id, 0x03, [STS_TORQUE_REG, 1 if enabled else 0],
                       0, expect_response=False)

    def close(self):
        try:
            self.port.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# KEP2 eye client (protocol facts from crates/kiko-eye-protocol)
# ----------------------------------------------------------------------------

KEP2_MAGIC = b"KE"
KEP2_VERSION = 2
K_IDENTITY_QUERY, K_IDENTITY_REPORT = 1, 2
K_ACQUIRE, K_ACQUIRE_RESULT = 3, 4
K_APPLY, K_RELEASE, K_INTENT_RESULT = 5, 6, 7
EXPR = {"neutral": 0, "curious": 1, "greet": 2, "concerned": 3, "sleepy": 4}


def crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            mask = -(crc & 1) & 0xFFFFFFFF
            crc = (crc >> 1) ^ (0x82F63B78 & mask)
    return (~crc) & 0xFFFFFFFF


def cobs_encode(data):
    out = bytearray()
    idx = 0
    while True:
        block = data[idx:idx + 254]
        zero = block.find(b"\x00")
        if zero == -1:
            out.append(len(block) + 1)
            out += block
            idx += len(block)
            if len(block) < 254:
                break
        else:
            out.append(zero + 1)
            out += block[:zero]
            idx += zero + 1
            if idx == len(data):
                out.append(1)
                break
    if len(data) == 0:
        out.append(1)
    out.append(0)
    return bytes(out)


def cobs_decode(data):
    out = bytearray()
    idx = 0
    while idx < len(data):
        code = data[idx]
        if code == 0:
            raise ValueError("zero inside COBS record")
        idx += 1
        count = code - 1
        if idx + count > len(data):
            raise ValueError("COBS overrun")
        chunk = data[idx:idx + count]
        if b"\x00" in chunk:
            raise ValueError("zero inside COBS block")
        out += chunk
        idx += count
        if code != 0xFF and idx < len(data):
            out.append(0)
    return bytes(out)


class Kep2Error(Exception):
    pass


class Kep2Eyes:
    def __init__(self, device, expected_uid_hex):
        self.port = serial.Serial(device, baudrate=115200, timeout=0.25,
                                  write_timeout=0.25, exclusive=True)
        self.expected_uid = bytes.fromhex(expected_uid_hex)
        self.boot_id = None
        self.epoch = None
        self.sequence = 0
        self.lock = threading.Lock()
        self.nonce = (int(time.time()) & 0xFFFFFFFF) | 0x100000000

    def _write_frame(self, kind, payload):
        raw = bytearray(KEP2_MAGIC)
        raw.append(KEP2_VERSION)
        raw.append(kind)
        raw += struct.pack("<H", len(payload))
        raw += b"\x00\x00"
        raw += payload
        raw += struct.pack("<I", crc32c(raw))
        self.port.write(cobs_encode(bytes(raw)))
        self.port.flush()

    def _read_frame(self, deadline_s=0.6):
        buf = bytearray()
        deadline = time.monotonic() + deadline_s
        while time.monotonic() < deadline:
            byte = self.port.read(1)
            if not byte:
                continue
            if byte == b"\x00":
                if not buf:
                    continue
                try:
                    raw = cobs_decode(bytes(buf))
                except ValueError:
                    buf.clear()
                    continue
                finally:
                    buf.clear()
                if len(raw) < 12 or raw[:2] != KEP2_MAGIC or raw[2] != KEP2_VERSION:
                    continue
                length = struct.unpack("<H", raw[4:6])[0]
                if len(raw) != 8 + length + 4:
                    continue
                if struct.unpack("<I", raw[-4:])[0] != crc32c(raw[:-4]):
                    continue
                return raw[3], raw[8:8 + length]
            else:
                buf += byte
        raise Kep2Error("no frame before deadline")

    def start_session(self):
        with self.lock:
            self.nonce += 1
            self._write_frame(K_IDENTITY_QUERY, struct.pack("<Q", self.nonce))
            kind, payload = self._read_frame()
            if kind != K_IDENTITY_REPORT or len(payload) != 76:
                raise Kep2Error(f"unexpected identity reply kind={kind}")
            (nonce,) = struct.unpack_from("<Q", payload, 0)
            uid = payload[8:24]
            boot_id = struct.unpack_from("<Q", payload, 56)[0]
            caps = struct.unpack_from("<I", payload, 72)[0]
            if nonce != self.nonce:
                raise Kep2Error("identity nonce mismatch")
            if uid != self.expected_uid:
                raise Kep2Error(f"eye UID mismatch: {uid.hex()}")
            self.boot_id = boot_id
            self.epoch = (int(time.time()) & 0x7FFFFFFF) or 1
            self.nonce += 1
            self._write_frame(K_ACQUIRE, struct.pack(
                "<QIQ", self.boot_id, self.epoch, self.nonce))
            kind, payload = self._read_frame()
            if kind != K_ACQUIRE_RESULT or len(payload) != 32:
                raise Kep2Error(f"unexpected acquire reply kind={kind}")
            result = payload[20]
            if result != 0:
                raise Kep2Error(f"acquire not granted: code={result}")
            self.sequence = 0
            return {"boot_id": boot_id, "capabilities": caps}

    def apply(self, gaze_x=0, gaze_y=0, lid=80, pupil=550, brightness=500,
              expression="neutral", blink=False, color=(80, 180, 200),
              lease_ms=400):
        with self.lock:
            payload = struct.pack(
                "<QIIHhhHHHBB3sB",
                self.boot_id, self.epoch, self.sequence, lease_ms,
                int(max(-1000, min(1000, gaze_x))),
                int(max(-1000, min(1000, gaze_y))),
                int(max(0, min(1000, lid))),
                int(max(0, min(1000, pupil))),
                int(max(0, min(1000, brightness))),
                EXPR[expression], 1 if blink else 0,
                bytes(color), 0)
            self._write_frame(K_APPLY, payload)
            kind, reply = self._read_frame()
            if kind != K_INTENT_RESULT:
                raise Kep2Error(f"unexpected intent reply kind={kind}")
            code = reply[16]
            self.sequence += 1
            if code not in (0, 1):
                raise Kep2Error(f"intent rejected: code={code}")

    def release(self, reason=1):
        with self.lock:
            if self.boot_id is None:
                return
            payload = struct.pack("<QIIB3x", self.boot_id, self.epoch,
                                  self.sequence, reason)
            try:
                self._write_frame(K_RELEASE, payload)
                self._read_frame(deadline_s=0.3)
            except Exception:
                pass

    def close(self):
        try:
            self.port.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Camera + detection thread
# ----------------------------------------------------------------------------

class CameraThread(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.latest = None          # (monotonic_ts, faces, frame_shape)
        self.latest_frame = None    # last BGR frame for snapshots
        self.fx = cfg["fx_fallback"]
        self.cx = cfg["camera_width"] / 2.0
        self.cy = cfg["camera_height"] / 2.0
        self.ready = threading.Event()
        self.dead = threading.Event()
        self.stop_flag = threading.Event()
        self.error = None
        cascade_dir = cv2.data.haarcascades
        self.frontal = cv2.CascadeClassifier(
            cascade_dir + "haarcascade_frontalface_default.xml")
        self.profile = cv2.CascadeClassifier(
            cascade_dir + "haarcascade_profileface.xml")

    def run(self):
        import depthai as dai
        try:
            info = dai.DeviceInfo(self.cfg["oak_mxid"])
            with dai.Device(info, dai.UsbSpeed.SUPER) as device:
                try:
                    calib = device.readCalibration2()
                    intr = calib.getCameraIntrinsics(
                        dai.CameraBoardSocket.CAM_A,
                        self.cfg["camera_width"], self.cfg["camera_height"])
                    self.fx = float(intr[0][0])
                    self.cx = float(intr[0][2])
                    self.cy = float(intr[1][2])
                except Exception:
                    pass
                pipeline = dai.Pipeline(device)
                cam = pipeline.create(dai.node.Camera).build(
                    dai.CameraBoardSocket.CAM_A)
                out = cam.requestOutput(
                    (self.cfg["camera_width"], self.cfg["camera_height"]),
                    dai.ImgFrame.Type.BGR888i, fps=self.cfg["camera_fps"])
                queue = out.createOutputQueue(maxSize=4, blocking=False)
                pipeline.start()
                print(f"camera_ready usb={device.getUsbSpeed().name} "
                      f"fx={self.fx:.1f} cx={self.cx:.1f} cy={self.cy:.1f}",
                      flush=True)
                self.ready.set()
                while not self.stop_flag.is_set():
                    msg = queue.tryGet()
                    if msg is None:
                        time.sleep(0.005)
                        continue
                    frame = msg.getCvFrame()
                    faces = self._detect(frame)
                    self.latest = (time.monotonic(), faces, frame.shape)
                    self.latest_frame = frame
        except Exception as exc:  # camera loss is a typed fault upstream
            self.error = repr(exc)
            self.dead.set()
            self.ready.set()

    def _detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = list(self.frontal.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=6, minSize=(30, 30)))
        if not faces:
            faces = list(self.profile.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30)))
            if not faces:
                flipped = cv2.flip(gray, 1)
                mirrored = self.profile.detectMultiScale(
                    flipped, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
                width = gray.shape[1]
                faces = [(width - x - w, y, w, h) for (x, y, w, h) in mirrored]
        return [tuple(int(v) for v in f) for f in faces]


# ----------------------------------------------------------------------------
# Head controller
# ----------------------------------------------------------------------------

class HeadController:
    def __init__(self, bus, cfg):
        self.bus = bus
        self.cfg = cfg
        self.natural = list(cfg["natural_ticks"])
        self.goal = None                       # last commanded goals
        self.expression_goal = None            # independently slewed character goal
        self.offsets = [0.0, 0.0, 0.0, 0.0]    # bow, curl, yaw, roll (ticks)
        self.limits = [cfg["bow_limit_ticks"], cfg["curl_limit_ticks"],
                       cfg["yaw_limit_ticks"], cfg["roll_limit_ticks"]]
        self.state = "INIT"
        self.fault = None
        self.thermal_derate = False
        self.compliant_policy = CompliantHeadPolicy.parse(
            cfg["compliant_hold"], cfg["torque_limit_permille"])
        self.compliance = None
        self.compliance_probe_due = 0.0

    def _confirm_temperature(self, joint, servo_id, telemetry, limit, stage):
        if telemetry["temperature_raw"] <= limit:
            return telemetry
        temperatures = [telemetry["temperature_raw"]]
        for _ in range(2):
            time.sleep(0.100)
            telemetry = self.bus.read_telemetry(servo_id)
            temperatures.append(telemetry["temperature_raw"])
            if telemetry["temperature_raw"] <= limit:
                print(
                    f"telemetry_temperature_transient stage={stage} "
                    f"joint={JOINT_NAMES[joint]} "
                    f"samples={'->'.join(map(str, temperatures))}",
                    flush=True,
                )
                return telemetry
        raise StsError(
            f"{JOINT_NAMES[joint]} confirmed {stage} overtemp "
            f"{'->'.join(map(str, temperatures))}")

    def admit_and_engage(self):
        poses = []
        for joint, servo_id in enumerate(SERVO_IDS):
            t = self.bus.read_position_redundant(servo_id)
            verified_position = t["position"]
            t = self._confirm_temperature(
                joint, servo_id, t,
                self.cfg["preengage_temp_limit_raw"], "preengage")
            poses.append(verified_position)
            window = self.cfg["admission_window_ticks"]
            if abs(verified_position - self.natural[joint]) > window:
                raise StsError(
                    f"{JOINT_NAMES[joint]} pose {verified_position} outside "
                    f"natural±{window}")
            if not (self.cfg["volt_min_raw"] <= t["voltage_raw"]
                    <= self.cfg["volt_max_raw"]):
                raise StsError(f"{JOINT_NAMES[joint]} voltage out of range: "
                               f"{t['voltage_raw']}")
            print(f"admit joint={JOINT_NAMES[joint]} pos={verified_position} "
                  f"temp={t['temperature_raw']} volt={t['voltage_raw']}",
                  flush=True)
        # Engage: torque limit, goal = present pose (zero jump), torque on.
        for joint, servo_id in enumerate(SERVO_IDS):
            self.bus.write_torque_limit(
                servo_id, self.cfg["torque_limit_permille"][joint])
            self.bus.write_goal(servo_id, poses[joint],
                                self.cfg["return_speed_ticks_s"])
            self.bus.write_torque_switch(servo_id, True)
        self.goal = list(poses)
        self.expression_goal = list(poses)
        self.state = "RETURNING"
        print(f"engaged_at={poses}", flush=True)

    def _target_ticks(self):
        return [self.natural[j] + int(round(self.offsets[j])) for j in range(4)]

    def _write_toward(self, desired, step, speed):
        deadband = self.cfg["goal_deadband_ticks"]
        for joint, servo_id in enumerate(SERVO_IDS):
            want = desired[joint]
            have = self.goal[joint]
            if abs(want - have) < deadband:
                continue
            move = max(-step, min(step, want - have))
            self.goal[joint] = have + move
            self.bus.write_goal(servo_id, self.goal[joint], speed)

    def step(self, desired_offsets, now=None):
        """One 20 Hz control step toward natural + bounded 4-DOF offsets."""
        now = time.monotonic() if now is None else now
        desired = [max(-self.limits[j], min(self.limits[j], desired_offsets[j]))
                   for j in range(4)]
        if self.state == "RETURNING":
            desired = [0.0, 0.0, 0.0, 0.0]
        # Ease offsets toward desired (exponential, per-cycle step clamp), so
        # motion starts and ends softly; the goal advance rate stays at or
        # below the servo speed so the shaft keeps up with the goal.
        step = (self.cfg["return_step_ticks"] if self.state == "RETURNING"
                else self.cfg["track_step_ticks"])
        ease = self.cfg["offset_ease"]
        for i in range(4):
            delta = (desired[i] - self.offsets[i]) * ease
            self.offsets[i] += max(-step, min(step, delta))
        target = self._target_ticks()
        for joint in range(4):
            have = self.expression_goal[joint]
            move = max(-step, min(step, target[joint] - have))
            self.expression_goal[joint] = have + move
        speed_base = self.cfg["return_speed_ticks_s"]
        # The incremental goal trajectory rate-limits real motion; the servo
        # speed register only needs to keep the shaft up with the goal. During
        # touch arbitration the compliant target is the only physical writer.
        if self.compliance is None or self.compliance.state == FOLLOWING:
            speed = (speed_base if self.state == "RETURNING"
                     else self.cfg["track_speed_max"])
            self._write_toward(self.expression_goal, step, speed)
        if self.state == "RETURNING":
            err = max(abs(self.goal[j] - self.natural[j]) for j in range(4))
            if err <= 2:
                self.state = "TRACKING"
                self.expression_goal = list(self.natural)
                self.compliance = CompliantHeadController(
                    self.compliant_policy, tuple(self.goal), now)
                print("head_at_natural tracking_enabled", flush=True)

        if (self.state == "TRACKING" and self.compliance is not None and
                now + 1e-9 >= self.compliance.next_service_due):
            self._service_compliance(now)

    def _read_safe_observation(self):
        started = time.monotonic()
        telemetry = []
        hottest = 0
        for joint, servo_id in enumerate(SERVO_IDS):
            t = self.bus.read_telemetry(servo_id)
            if t["temperature_raw"] >= self.cfg["temp_abort_raw"]:
                t = self._confirm_temperature(
                    joint, servo_id, t,
                    self.cfg["temp_abort_raw"] - 1, "energized")
                # Confirmation occupied more than the admitted observation
                # span. Freeze output and acquire a complete fresh set next
                # control slot rather than mixing timestamps.
                return None
            hottest = max(hottest, t["temperature_raw"])
            if not (self.cfg["volt_min_raw"] <= t["voltage_raw"]
                    <= self.cfg["volt_max_raw"]):
                raise StsError(f"{JOINT_NAMES[joint]} voltage "
                               f"{t['voltage_raw']}")
            if self.goal is not None:
                divergence = abs(t["position"] - self.goal[joint])
                if divergence > self.cfg["goal_divergence_abort_ticks"]:
                    raise StsError(
                        f"{JOINT_NAMES[joint]} diverged {divergence} ticks "
                        f"(pos={t['position']} goal={self.goal[joint]})")
            telemetry.append(t)
        span = time.monotonic() - started
        self._update_thermal_derate(hottest)
        return telemetry, span

    def _update_thermal_derate(self, hottest):
        if not self.thermal_derate and hottest >= self.cfg["derate_temp_raw"]:
            self.thermal_derate = True
            print(f"thermal_derate_on hottest={hottest} "
                  f"(pitch joints resting)", flush=True)
        elif self.thermal_derate and hottest <= self.cfg["derate_clear_temp_raw"]:
            self.thermal_derate = False
            print(f"thermal_derate_off hottest={hottest}", flush=True)

    def _service_compliance(self, now):
        observed = self._read_safe_observation()
        if observed is None:
            # Preserve the last verified target while fresh temperature
            # confirmation consumed this control slot.
            self.compliance.next_service_due = time.monotonic()
            return
        telemetry, span = observed
        observed_command = tuple(self.goal)
        step = self.compliance.service(
            now,
            observed_command,
            tuple(item["position"] for item in telemetry),
            tuple(bool(item["moving"]) for item in telemetry),
            span,
        )
        desired = list(step.target_ticks)
        for joint, servo_id in enumerate(SERVO_IDS):
            if desired[joint] != self.goal[joint]:
                self.goal[joint] = desired[joint]
                self.bus.write_goal(
                    servo_id, desired[joint], self.cfg["track_speed_max"])
        if step.event is not None:
            print(f"compliant event={step.event} state={step.state} "
                  f"target={list(step.target_ticks)} "
                  f"residual={list(step.residual_error_ticks)} "
                  f"baseline={list(self.compliance.baseline_error)}", flush=True)
        # A bounded, rate-limited near-threshold trace makes attended tuning
        # observable without turning ordinary encoder noise into a touch or
        # flooding the long-running owner log.
        near_contact = any(
            abs(error) * 2 >= threshold
            for error, threshold in zip(
                step.residual_error_ticks,
                self.compliant_policy.contact_entry_error_ticks))
        if (self.compliance.contact_armed and near_contact and
                now >= self.compliance_probe_due):
            print(
                f"compliant probe residual={list(step.residual_error_ticks)} "
                f"positions={[item['position'] for item in telemetry]} "
                f"goal={list(observed_command)} "
                f"baseline={list(self.compliance.baseline_error)} "
                f"moving={[bool(item['moving']) for item in telemetry]}",
                flush=True,
            )
            self.compliance_probe_due = now + 0.5

    def telemetry_check(self):
        self._read_safe_observation()

    def park_and_release(self):
        """Return to natural at bounded speed and release I/O, retaining torque."""
        try:
            print("park_begin", flush=True)
            self.compliance = None
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                pending = False
                for joint, servo_id in enumerate(SERVO_IDS):
                    want = self.natural[joint]
                    have = self.goal[joint] if self.goal else want
                    if want != have:
                        pending = True
                        move = max(-self.cfg["return_step_ticks"],
                                   min(self.cfg["return_step_ticks"],
                                       want - have))
                        self.goal[joint] = have + move
                        self.bus.write_goal(servo_id, self.goal[joint],
                                            self.cfg["return_speed_ticks_s"])
                if not pending:
                    break
                time.sleep(0.1)
            time.sleep(1.0)  # let motion settle before ownership release
        except Exception as exc:
            print(f"park_error {exc!r} (torque remains enabled)", flush=True)
        # OPERATOR HARD RULE (2026-07-21): the mechanism must NEVER lose
        # tension — a torque-off collapse reached bow=1733. Every exit path
        # parks at natural and leaves all four servos holding.
        print("park_complete torque_hold=ENABLED tension_preserved=true",
              flush=True)
        return True


# ----------------------------------------------------------------------------
# Character engine: non-repetitive expressive behavior for head + eyes
# ----------------------------------------------------------------------------
#
# Design: a baseline mood layer (tracking gaze, drifting color, breathing,
# blinks) plus a library of short parameterized "acts" (tilts, double-takes,
# wiggles, lean-ins, sparkles, look-arounds...). Every act instance is rebuilt
# from randomized parameters, scheduled with per-act cooldowns, novelty
# suppression against recent history, and randomized gaps — so performances
# never repeat exactly. All head output remains bounded offsets that the
# HeadController clamps, slews, and thermally derates.

import random


def _hsv_to_rgb(h, s, v):
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t),
               (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return (int(r * 255), int(g * 255), int(b * 255))


def _ease(a, b, u):
    return a + (b - a) * (1 - math.cos(math.pi * max(0.0, min(1.0, u)))) / 2


class ActPerformance:
    """One running act: per-channel keyframes with cosine easing.

    Channels: yaw, pitch, roll (head ticks, additive), gx, gy (gaze bias),
    lid_add, pupil_add, bright_mul, colorw (0..1 blend toward color).
    Every head channel must start and end at 0 so acts blend seamlessly.
    """

    def __init__(self, name, duration, keys, color=None, expression=None,
                 blinks=(), uses_pitch=False):
        self.name = name
        self.duration = duration
        self.keys = keys
        self.color = color
        self.expression = expression
        self.blinks = sorted(blinks)
        self.uses_pitch = uses_pitch
        self.fired_blinks = 0

    def sample(self, frac):
        out = {}
        for channel, points in self.keys.items():
            previous = points[0]
            value = points[-1][1]
            for point in points[1:]:
                if frac <= point[0]:
                    span = point[0] - previous[0]
                    u = 1.0 if span <= 0 else (frac - previous[0]) / span
                    value = _ease(previous[1], point[1], u)
                    break
                previous = point
            out[channel] = value
        return out

    def take_blink(self, frac):
        if self.fired_blinks < len(self.blinks) and frac >= self.blinks[self.fired_blinks]:
            self.fired_blinks += 1
            return True
        return False


def _keys_pulse(peak, rise=0.3, fall=0.7):
    return [(0.0, 0.0), (rise, peak), (fall, peak), (1.0, 0.0)]


def _build_curious_tilt(rng, ctx):
    side = rng.choice((-1, 1))
    amp = rng.uniform(55, 140) * side
    hold = rng.uniform(0.35, 0.55)
    keys = {"roll": [(0.0, 0.0), (0.30, amp), (0.30 + hold, amp * rng.uniform(0.85, 1.0)), (1.0, 0.0)],
            "yaw": _keys_pulse(rng.uniform(25, 55) * side),
            "pupil_add": _keys_pulse(rng.uniform(80, 180))}
    if rng.random() < 0.4:
        keys["gy"] = _keys_pulse(rng.uniform(60, 140) * side)
    return ActPerformance("curious_tilt", rng.uniform(2.2, 3.6), keys,
                          expression="curious")


def _build_double_take(rng, ctx):
    side = rng.choice((-1, 1))
    away = rng.uniform(120, 230) * side
    return ActPerformance(
        "double_take", rng.uniform(1.3, 1.9),
        {"yaw": [(0.0, 0.0), (0.22, away), (0.42, away * 0.9), (0.75, -away * 0.12), (1.0, 0.0)],
         "gx": [(0.0, 0.0), (0.18, away * 2.2), (0.45, away * 2.0), (0.7, 0.0), (1.0, 0.0)]},
        blinks=(0.72,))


def _build_excited_wiggle(rng, ctx):
    amp = rng.uniform(42, 78)
    cycles = rng.choice((2, 3, 3, 4))
    keys_roll = [(0.0, 0.0)]
    for i in range(cycles * 2):
        keys_roll.append(((i + 1) / (cycles * 2 + 1), amp * (1 if i % 2 == 0 else -1)))
    keys_roll.append((1.0, 0.0))
    return ActPerformance(
        "excited_wiggle", rng.uniform(1.6, 2.6),
        {"roll": keys_roll, "pitch": _keys_pulse(rng.uniform(18, 32), 0.25, 0.75),
         "bright_mul": _keys_pulse(0.35),
         "pupil_add": _keys_pulse(rng.uniform(150, 250))},
        color=_hsv_to_rgb(rng.uniform(0.02, 0.12), 0.9, 1.0),
        expression="greet")


def _build_lean_in(rng, ctx):
    return ActPerformance(
        "lean_in", rng.uniform(2.8, 4.4),
        {"pitch": [(0.0, 0.0), (0.35, rng.uniform(48, 95)), (0.75, rng.uniform(40, 85)), (1.0, 0.0)],
         "roll": _keys_pulse(rng.uniform(16, 34) * rng.choice((-1, 1)), 0.3, 0.75),
         "pupil_add": _keys_pulse(rng.uniform(120, 220), 0.35, 0.75),
         "gy": _keys_pulse(rng.uniform(-60, 60))},
        expression="curious", uses_pitch=True)


def _build_nod(rng, ctx):
    dip = rng.uniform(36, 68)
    return ActPerformance(
        "nod", rng.uniform(1.6, 2.3),
        {"pitch": [(0.0, 0.0), (0.22, -dip * 0.85), (0.45, -dip), (0.68, dip * 0.4),
                   (0.85, dip * 0.15), (1.0, 0.0)]},
        uses_pitch=True)


def _build_soft_nod(rng, ctx):
    dip = rng.uniform(24, 42)
    bounces = rng.choice((2, 2, 3))
    keys = [(0.0, 0.0)]
    for i in range(bounces):
        base = (i + 0.5) / (bounces + 0.5)
        keys.append((base - 0.08, -dip * rng.uniform(0.85, 1.0)))
        keys.append((base + 0.08, dip * 0.18))
    keys.append((1.0, 0.0))
    return ActPerformance(
        "soft_nod", rng.uniform(2.4, 3.4),
        {"pitch": sorted(keys), "bright_mul": _keys_pulse(0.18)},
        expression="greet", uses_pitch=True)


def _build_happy_squint(rng, ctx):
    return ActPerformance(
        "happy_squint", rng.uniform(1.6, 2.6),
        {"lid_add": _keys_pulse(rng.uniform(330, 450), 0.25, 0.75),
         "roll": _keys_pulse(rng.uniform(22, 44) * rng.choice((-1, 1)), 0.25, 0.75),
         "bright_mul": _keys_pulse(0.35, 0.2, 0.8),
         "pupil_add": _keys_pulse(-rng.uniform(60, 120))},
        color=_hsv_to_rgb(rng.uniform(0.09, 0.15), 0.85, 1.0),
        expression="greet")


def _build_puppy_eyes(rng, ctx):
    sway = rng.uniform(14, 26) * rng.choice((-1, 1))
    return ActPerformance(
        "puppy_eyes", rng.uniform(2.6, 3.8),
        {"pupil_add": _keys_pulse(rng.uniform(300, 400), 0.25, 0.8),
         "gy": _keys_pulse(-rng.uniform(80, 140), 0.3, 0.75),
         "roll": [(0.0, 0.0), (0.3, sway), (0.6, -sway * 0.7), (1.0, 0.0)],
         "bright_mul": _keys_pulse(0.2, 0.3, 0.8)},
        color=_hsv_to_rgb(rng.uniform(0.86, 0.95), 0.45, 1.0),
        expression="curious")


def _build_shy_dip(rng, ctx):
    side = rng.choice((-1, 1))
    return ActPerformance(
        "shy_dip", rng.uniform(1.8, 2.6),
        {"gy": [(0.0, 0.0), (0.3, -rng.uniform(280, 420)), (0.7, -rng.uniform(200, 350)), (1.0, 0.0)],
         "roll": _keys_pulse(rng.uniform(30, 56) * side),
         "pitch": _keys_pulse(-rng.uniform(18, 34)),
         "lid_add": _keys_pulse(rng.uniform(180, 300))},
        color=_hsv_to_rgb(rng.uniform(0.9, 0.98), 0.5, 0.9))


def _build_sparkle(rng, ctx):
    flickers = [(0.0, 0.0)]
    n = rng.randint(4, 7)
    for i in range(n):
        flickers.append(((i + 1) / (n + 1), 0.45 if i % 2 == 0 else 0.05))
    flickers.append((1.0, 0.0))
    return ActPerformance(
        "sparkle", rng.uniform(1.0, 1.7),
        {"bright_mul": flickers, "pupil_add": _keys_pulse(rng.uniform(80, 160))},
        color=_hsv_to_rgb(rng.uniform(0.45, 0.62), 0.55, 1.0))


def _build_blink_flourish(rng, ctx):
    return ActPerformance(
        "blink_flourish", rng.uniform(0.9, 1.3),
        {"roll": _keys_pulse(rng.uniform(16, 28) * rng.choice((-1, 1)))},
        blinks=(0.15, 0.5) if rng.random() < 0.6 else (0.3,))


def _build_look_around(rng, ctx):
    n = rng.randint(2, 4)
    gx_keys, yaw_keys = [(0.0, 0.0)], [(0.0, 0.0)]
    for i in range(n):
        frac = (i + 1) / (n + 1)
        gaze = rng.uniform(-700, 700)
        gx_keys.append((frac, gaze))
        yaw_keys.append((frac, gaze * rng.uniform(0.25, 0.45)))
    gx_keys.append((1.0, 0.0))
    yaw_keys.append((1.0, 0.0))
    return ActPerformance(
        "look_around", rng.uniform(3.2, 5.5),
        {"gx": gx_keys, "yaw": yaw_keys,
         "gy": _keys_pulse(rng.uniform(-150, 220))})


def _build_perk_up(rng, ctx):
    return ActPerformance(
        "perk_up", rng.uniform(1.4, 2.2),
        {"roll": _keys_pulse(rng.uniform(48, 88) * rng.choice((-1, 1)), 0.2, 0.75),
         "pitch": _keys_pulse(rng.uniform(24, 42), 0.2, 0.8),
         "bright_mul": _keys_pulse(0.4, 0.2, 0.8),
         "pupil_add": _keys_pulse(-120)},
        expression="curious")


def _build_daydream(rng, ctx):
    phase = rng.uniform(0, math.pi)
    n = 6
    gx = [(i / n, 300 * math.sin(phase + i * 1.1)) for i in range(n + 1)]
    gy = [(i / n, 180 * math.sin(phase * 0.7 + i * 0.8)) for i in range(n + 1)]
    gx[0] = (0.0, 0.0); gx[-1] = (1.0, 0.0)
    gy[0] = (0.0, 0.0); gy[-1] = (1.0, 0.0)
    return ActPerformance(
        "daydream", rng.uniform(4.0, 6.5),
        {"gx": gx, "gy": gy, "bright_mul": _keys_pulse(-0.25, 0.3, 0.7)})


def _build_stretch(rng, ctx):
    return ActPerformance(
        "stretch", rng.uniform(2.6, 3.6),
        {"pitch": [(0.0, 0.0), (0.35, -rng.uniform(45, 70)), (0.7, rng.uniform(40, 68)), (1.0, 0.0)],
         "yaw": _keys_pulse(rng.uniform(-70, 70), 0.3, 0.7),
         "lid_add": _keys_pulse(rng.uniform(250, 400), 0.3, 0.7)},
        uses_pitch=True)


def _build_sweep_scan(rng, ctx):
    side = rng.choice((-1, 1))
    arc = rng.uniform(150, 280) * side
    return ActPerformance(
        "sweep_scan", rng.uniform(3.4, 5.2),
        {"yaw": [(0.0, 0.0), (0.3, arc), (0.55, arc * 0.8), (0.8, -arc * 0.35), (1.0, 0.0)],
         "roll": [(0.0, 0.0), (0.3, -arc * 0.22), (0.8, arc * 0.14), (1.0, 0.0)],
         "gx": [(0.0, 0.0), (0.2, arc * 1.6), (0.55, arc * 1.2), (0.85, -arc * 0.5), (1.0, 0.0)]},
        expression="curious")


def _build_head_bob(rng, ctx):
    amp = rng.uniform(26, 44)
    cycles = rng.choice((2, 3))
    pitch_keys, roll_keys = [(0.0, 0.0)], [(0.0, 0.0)]
    for i in range(cycles * 2):
        frac = (i + 1) / (cycles * 2 + 1)
        pitch_keys.append((frac, amp * (1 if i % 2 == 0 else -0.55)))
        roll_keys.append((frac, amp * 0.5 * (1 if i % 2 else -1)))
    pitch_keys.append((1.0, 0.0))
    roll_keys.append((1.0, 0.0))
    return ActPerformance(
        "head_bob", rng.uniform(1.8, 2.8),
        {"pitch": pitch_keys, "roll": roll_keys,
         "bright_mul": _keys_pulse(0.25)}, uses_pitch=True)


def _build_sneeze(rng, ctx):
    up = rng.uniform(40, 62)
    down = -rng.uniform(70, 95)
    shiver = rng.uniform(6, 10)
    kick = rng.uniform(14, 26) * rng.choice((-1, 1))
    return ActPerformance(
        "sneeze", rng.uniform(2.4, 3.2),
        {"pitch": [(0.0, 0.0), (0.18, up * 0.5), (0.34, up * 0.85), (0.46, up),
                   (0.56, down), (0.68, down * 0.55), (0.80, -shiver),
                   (0.90, shiver * 0.6), (1.0, 0.0)],
         "roll": [(0.0, 0.0), (0.22, shiver), (0.30, -shiver), (0.38, shiver),
                  (0.46, -shiver * 0.5), (0.56, kick), (0.75, 0.0), (1.0, 0.0)],
         "gy": [(0.0, 0.0), (0.45, 300), (0.56, -450), (0.8, 0.0), (1.0, 0.0)],
         "lid_add": [(0.0, 0.0), (0.4, 350), (0.52, 900), (0.66, 500),
                     (0.85, 0.0), (1.0, 0.0)],
         "bright_mul": [(0.0, 0.0), (0.45, 0.25), (0.56, 0.6), (0.7, 0.0),
                        (1.0, 0.0)]},
        color=(255, 240, 210), blinks=(0.55,), expression="concerned",
        uses_pitch=True)


def _build_dance(rng, ctx):
    style = rng.randrange(3)
    beats = rng.choice((4, 6))
    base_hue = rng.random()
    yaw, pitch, roll = [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]
    gx, hue = [(0.0, 0.0)], [(0.0, base_hue)]
    for i in range(beats):
        f = (i + 1) / (beats + 1)
        s = 1 if i % 2 == 0 else -1
        if style == 0:    # sway: broad yaw + counter-roll
            yaw.append((f, s * rng.uniform(70, 130)))
            roll.append((f, -s * rng.uniform(35, 60)))
            pitch.append((f, rng.uniform(8, 18)))
        elif style == 1:  # bop: pitch bounce with roll accents
            pitch.append((f, rng.uniform(26, 42) * (1 if i % 2 else -0.7)))
            roll.append((f, s * rng.uniform(18, 34)))
            yaw.append((f, s * rng.uniform(15, 35)))
        else:             # big sweep: wide yaw leans
            yaw.append((f, s * rng.uniform(140, 220)))
            roll.append((f, s * rng.uniform(40, 70)))
            pitch.append((f, rng.uniform(-10, 16)))
        gx.append((f, s * rng.uniform(300, 600)))
        hue.append((f, (base_hue + (i + 1) * 0.17) % 1.0))
    for keys in (yaw, pitch, roll, gx):
        keys.append((1.0, 0.0))
    hue.append((1.0, (base_hue + (beats + 1) * 0.17) % 1.0))
    return ActPerformance(
        "dance", rng.uniform(4.2, 6.8),
        {"yaw": yaw, "pitch": pitch, "roll": roll, "gx": gx, "hue": hue,
         "bright_mul": _keys_pulse(0.3, 0.15, 0.85)},
        expression="greet", uses_pitch=True)


ACT_LIBRARY = [
    # (name, builder, states, weight, cooldown_s)
    ("curious_tilt", _build_curious_tilt, ("TRACK",), 3.0, 9.0),
    ("double_take", _build_double_take, ("TRACK",), 2.0, 16.0),
    ("excited_wiggle", _build_excited_wiggle, ("TRACK",), 1.6, 14.0),
    ("lean_in", _build_lean_in, ("TRACK",), 2.0, 15.0),
    ("nod", _build_nod, ("TRACK",), 2.2, 11.0),
    ("soft_nod", _build_soft_nod, ("TRACK",), 2.4, 12.0),
    ("happy_squint", _build_happy_squint, ("TRACK",), 2.0, 12.0),
    ("puppy_eyes", _build_puppy_eyes, ("TRACK",), 1.6, 18.0),
    ("shy_dip", _build_shy_dip, ("TRACK",), 0.9, 25.0),
    ("sparkle", _build_sparkle, ("TRACK", "IDLE"), 1.6, 10.0),
    ("blink_flourish", _build_blink_flourish, ("TRACK", "IDLE"), 1.8, 8.0),
    ("look_around", _build_look_around, ("IDLE",), 3.0, 7.0),
    ("perk_up", _build_perk_up, ("IDLE",), 1.6, 13.0),
    ("daydream", _build_daydream, ("IDLE",), 2.0, 12.0),
    ("stretch", _build_stretch, ("IDLE",), 0.8, 40.0),
    ("sweep_scan", _build_sweep_scan, ("TRACK", "IDLE"), 1.8, 16.0),
    ("head_bob", _build_head_bob, ("TRACK",), 1.8, 13.0),
    ("sneeze", _build_sneeze, ("TRACK", "IDLE"), 0.7, 45.0),
    ("dance", _build_dance, ("TRACK", "IDLE"), 1.3, 30.0),
]


class CharacterEngine:
    """Mood state machine + act scheduler. Returns (eye_intent, desired4)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = random.SystemRandom()
        self.mode = "IDLE"
        self.mode_since = time.monotonic()
        self.last_person_at = time.monotonic()
        self.greet_until = 0.0
        self.search_until = 0.0
        self.next_blink = time.monotonic() + 4.0
        self.act = None
        self.act_started = 0.0
        self.next_act_at = time.monotonic() + self.rng.uniform(2.0, 5.0)
        self.last_run = {}
        self.history = []
        self.saccade = (0.0, 0.0)
        self.saccade_until = 0.0
        self.hue_phase = self.rng.uniform(0.0, 6.28)
        self.still_ref = 0.0
        self.still_since = time.monotonic()
        self.prev_proximity = 0.0
        self.greet_style = 0

    # -- mode machine ---------------------------------------------------------
    def _enter(self, mode):
        if mode != self.mode:
            print(f"eyes_mode {self.mode}->{mode}", flush=True)
            self.mode = mode
            self.mode_since = time.monotonic()

    def _update_mode(self, person, now):
        if person:
            if self.mode in ("IDLE", "SLEEPY", "SEARCH", "LOST"):
                if now - self.last_person_at > self.cfg["greet_cooldown_s"]:
                    self._enter("GREET")
                    self.greet_until = now + self.rng.uniform(0.9, 1.4)
                    self.greet_style = self.rng.randrange(3)
                else:
                    self._enter("TRACK")
            self.last_person_at = now
        else:
            if self.mode in ("TRACK", "GREET"):
                self._enter("LOST")
            elif self.mode == "LOST" and now - self.mode_since > 0.7:
                self._enter("SEARCH")
                self.search_until = now + self.rng.uniform(2.2, 3.8)
            elif self.mode == "SEARCH" and now > self.search_until:
                self._enter("IDLE")
            elif self.mode == "IDLE" and (now - self.last_person_at
                                          > self.cfg["sleepy_after_idle_s"]):
                self._enter("SLEEPY")
        if self.mode == "GREET" and now >= self.greet_until:
            self._enter("TRACK")

    # -- act scheduling -------------------------------------------------------
    def _maybe_start_act(self, now, derate, proximity):
        if self.act is not None or self.mode not in ("TRACK", "IDLE"):
            return
        burst = (proximity - self.prev_proximity) > 0.30
        if now < self.next_act_at and not burst:
            return
        pool = []
        for name, builder, states, weight, cooldown in ACT_LIBRARY:
            if self.mode not in states:
                continue
            if now - self.last_run.get(name, -1e9) < cooldown:
                continue
            w = weight
            w *= 0.25 ** self.history[-4:].count(name)  # novelty suppression
            if burst and name == "excited_wiggle":
                w *= 8.0
            if now - self.still_since > 3.0 and name in ("curious_tilt", "lean_in", "nod"):
                w *= 2.2
            pool.append((name, builder, w))
        if not pool:
            self.next_act_at = now + self.rng.uniform(1.5, 4.0)
            return
        total = sum(w for _, _, w in pool)
        pick = self.rng.uniform(0, total)
        for name, builder, w in pool:
            pick -= w
            if pick <= 0:
                act = builder(self.rng, {})
                if derate and act.uses_pitch:
                    self.next_act_at = now + self.rng.uniform(1.0, 3.0)
                    return
                self.act = act
                self.act_started = now
                self.last_run[name] = now
                self.history.append(name)
                del self.history[:-8]
                print(f"act {name} dur={act.duration:.1f}s", flush=True)
                return

    def _act_sample(self, now):
        if self.act is None:
            return {}, False
        frac = (now - self.act_started) / self.act.duration
        if frac >= 1.0:
            gap = (self.rng.uniform(3.0, 9.0) if self.mode == "TRACK"
                   else self.rng.uniform(4.0, 12.0))
            self.next_act_at = now + gap
            self.act = None
            return {}, False
        return self.act.sample(frac), self.act.take_blink(frac)

    # -- main tick ------------------------------------------------------------
    def compute(self, now, person, bearings, residual, proximity, derate):
        cfg = self.cfg
        self._update_mode(person, now)

        if person and abs(bearings[0] - self.still_ref) > 0.035:
            self.still_ref = bearings[0]
            self.still_since = now
        self._maybe_start_act(now, derate, proximity)
        act_values, act_blink = self._act_sample(now)
        act = self.act  # may be None after sampling completes it
        self.prev_proximity = proximity

        # Baseline blink schedule (Poisson-ish), suppressed while an act blinks.
        blink = act_blink
        if now >= self.next_blink:
            blink = True
            self.next_blink = now + min(12.0, max(2.0, self.rng.expovariate(1 / 4.5)))

        # Micro-saccades refresh at random intervals.
        if now >= self.saccade_until:
            self.saccade = (self.rng.gauss(0, 26), self.rng.gauss(0, 18))
            self.saccade_until = now + self.rng.uniform(0.18, 0.55)

        # Slowly drifting base color, warmed by proximity.
        hue = 0.52 + 0.10 * math.sin(now * 0.045 + self.hue_phase)
        warm = proximity * 0.85
        hue = hue * (1 - warm) + 0.07 * warm
        base_color = _hsv_to_rgb(hue, 0.75, 0.72 + 0.28 * proximity)

        gx_act = act_values.get("gx", 0.0)
        gy_act = act_values.get("gy", 0.0)
        lid_add = act_values.get("lid_add", 0.0)
        pupil_add = act_values.get("pupil_add", 0.0)
        bright_mul = 1.0 + act_values.get("bright_mul", 0.0)

        if self.mode in ("GREET", "TRACK"):
            gx = residual[0] * 2100 * cfg["eye_x_sign"] + gx_act + self.saccade[0]
            gy = residual[1] * 2100 * cfg["eye_y_sign"] + gy_act + self.saccade[1]
            if self.mode == "GREET":
                styles = [
                    dict(expression="greet", color=(255, 180, 70), brightness=880,
                         pupil=740, blink=(now - self.mode_since) < 0.5),
                    dict(expression="greet", color=_hsv_to_rgb(0.33, 0.8, 1.0),
                         brightness=820, pupil=680, blink=blink),
                    dict(expression="curious", color=(255, 120, 150),
                         brightness=760, pupil=800, blink=blink),
                ]
                style = styles[self.greet_style]
                intent = dict(gaze_x=gx, gaze_y=gy, lid=40 + lid_add,
                              pupil=style["pupil"], brightness=style["brightness"],
                              expression=style["expression"], blink=style["blink"],
                              color=style["color"])
            else:
                expression = "curious" if (act and act.expression == "curious") else (
                    act.expression if act and act.expression else
                    ("curious" if now - self.still_since > 2.0 else "neutral"))
                color = base_color
                if act and act.color:
                    color = act.color
                if "hue" in act_values:
                    color = _hsv_to_rgb(act_values["hue"], 0.9, 1.0)
                intent = dict(
                    gaze_x=gx, gaze_y=gy, lid=60 + lid_add,
                    pupil=int(430 + proximity * 400 + pupil_add),
                    brightness=int(720 * bright_mul), expression=expression,
                    blink=blink, color=color)
        elif self.mode == "LOST":
            intent = dict(gaze_x=self.saccade[0] * 3, gaze_y=gy_act, lid=140,
                          pupil=520, brightness=600, expression="concerned",
                          blink=False, color=(200, 120, 180))
        elif self.mode == "SEARCH":
            phase = (now - self.mode_since) * 2 * math.pi * self.rng.choice((0.3, 0.35, 0.42))
            intent = dict(gaze_x=650 * math.sin(phase), gaze_y=-80 + gy_act,
                          lid=100, pupil=560, brightness=650,
                          expression="curious", blink=blink,
                          color=(150, 150, 210))
        elif self.mode == "SLEEPY":
            breathe = 0.5 + 0.5 * math.sin((now - self.mode_since) * 0.5)
            intent = dict(gaze_x=0, gaze_y=-250, lid=620, pupil=400,
                          brightness=int(150 + 60 * breathe),
                          expression="sleepy", blink=False, color=(25, 60, 130))
        else:  # IDLE
            t = now - self.mode_since
            breathe = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 0.1)
            intent = dict(
                gaze_x=200 * math.sin(t * 0.31) + gx_act + self.saccade[0] * 0.6,
                gaze_y=120 * math.sin(t * 0.21 + 1.3) + gy_act,
                lid=90 + lid_add, pupil=int(550 + pupil_add),
                brightness=int((300 + 120 * breathe) * bright_mul),
                expression=(act.expression if act and act.expression else "neutral"),
                blink=blink,
                color=(_hsv_to_rgb(act_values["hue"], 0.9, 1.0)
                       if "hue" in act_values
                       else (act.color if act and act.color else base_color)))

        # Head: tracking aim + act overlay, all as bounded offsets.
        breathing = 6.0 * math.sin(now * 2 * math.pi * 0.22)
        if person:
            yaw_aim = cfg["yaw_sign"] * cfg["yaw_ticks_per_rad"] * bearings[0]
            pitch_aim = cfg["pitch_ticks_per_rad"] * bearings[1]
        else:
            yaw_aim, pitch_aim = 0.0, 0.0
        mode_pitch, mode_yaw, mode_roll = 0.0, 0.0, 0.0
        if self.mode == "GREET":
            u = min(1.0, (now - self.mode_since) / max(0.4, self.greet_until - self.mode_since))
            pulse = math.sin(math.pi * u)
            side = 1 if self.greet_style % 2 == 0 else -1
            mode_roll = side * 95.0 * pulse
            mode_pitch = 34.0 * pulse
        elif self.mode == "SEARCH":
            sweep_phase = (now - self.mode_since) * 2 * math.pi * 0.3
            mode_yaw = 140.0 * math.sin(sweep_phase)
            mode_roll = 30.0 * math.sin(sweep_phase * 0.5)
        pitch_total = pitch_aim + act_values.get("pitch", 0.0) + mode_pitch
        if derate:
            pitch_total = 0.0
        curl_t = cfg["curl_sign"] * cfg["curl_pitch_share"] * pitch_total
        bow_t = -cfg["curl_sign"] * cfg["bow_pitch_share"] * pitch_total
        yaw_t = yaw_aim + act_values.get("yaw", 0.0) + mode_yaw
        roll_t = cfg["roll_sign"] * (act_values.get("roll", 0.0) + mode_roll
                                     + 34.0 * proximity) + breathing
        desired4 = [bow_t, curl_t, yaw_t, roll_t]
        return intent, desired4

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
        print(f"config_loaded {path}", flush=True)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "config.json"))
    parser.add_argument("--no-head", action="store_true")
    parser.add_argument("--no-camera", action="store_true",
                        help="idle character mode: head+eyes only, zero USB "
                             "camera traffic (for benchmark coexistence)")
    parser.add_argument("--no-eyes", action="store_true")
    parser.add_argument("--axis-dance", action="store_true",
                        help="announce firmware gaze directions for operator "
                             "axis confirmation before the main loop")
    parser.add_argument("--snapshot", action="store_true",
                        help="save annotated frames to /tmp/follow-snap.jpg")
    parser.add_argument("--duration-s", type=float, default=1800.0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.no_camera:
        # Performer mode: no person detection exists, so never doze off.
        cfg["sleepy_after_idle_s"] = float("inf")

    stop = threading.Event()

    def on_signal(signum, _frame):
        print(f"signal {signum} -> shutdown", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    camera = None
    if not args.no_camera:
        camera = CameraThread(cfg)
        camera.start()
        camera.ready.wait(timeout=30)
        if camera.dead.is_set():
            print(f"fatal camera_failed {camera.error}", flush=True)
            return 2

    eyes = None
    if not args.no_eyes:
        eyes = Kep2Eyes(cfg["eye_device"], cfg["eye_uid_hex"])
        session = eyes.start_session()
        print(f"eyes_session boot_id={session['boot_id']} "
              f"capabilities={session['capabilities']}", flush=True)

    head = None
    bus = None
    if not args.no_head:
        bus = StsBus(cfg["head_device"])
        head = HeadController(bus, cfg)
        head.admit_and_engage()

    if eyes is not None and args.axis_dance:
        dance = [
            ("RED    = gaze_x=-1000 (protocol LEFT)", -1000, 0, (255, 0, 0)),
            ("BLUE   = gaze_x=+1000 (protocol RIGHT)", 1000, 0, (0, 80, 255)),
            ("GREEN  = gaze_y=+1000 (protocol +Y)", 0, 1000, (0, 255, 80)),
            ("YELLOW = gaze_y=-1000 (protocol -Y)", 0, -1000, (255, 220, 0)),
            ("WHITE  = center", 0, 0, (255, 255, 255)),
        ]
        for label, gx, gy, color in dance:
            print(f"axis_dance {label}", flush=True)
            end = time.monotonic() + 2.5
            while time.monotonic() < end and not stop.is_set():
                eyes.apply(gaze_x=gx, gaze_y=gy, lid=40, pupil=650,
                           brightness=900, expression="neutral",
                           color=color, lease_ms=400)
                time.sleep(0.1)

    engine = CharacterEngine(cfg)
    yaw_rad_est = 0.0
    pitch_rad_est = 0.0
    smoothing = {"x": None, "y": None, "w": None}
    last_seen = 0.0
    last_frame_ts = None
    candidate_prev = None
    switch_streak = 0
    eye_failures = 0
    started = time.monotonic()
    tick = 0
    exit_code = 0

    try:
        while not stop.is_set():
            loop_t0 = time.monotonic()
            if loop_t0 - started > args.duration_s:
                print("duration_reached", flush=True)
                break
            if camera is not None and camera.dead.is_set():
                raise RuntimeError(f"camera died: {camera.error}")

            person = False
            residual = (0.0, 0.0)
            proximity = 0.0
            latest = camera.latest if camera is not None else None
            if latest is not None:
                ts, faces, _shape = latest
                if faces and (loop_t0 - ts) < 1.0 and ts != last_frame_ts:
                    last_frame_ts = ts
                    selected = None
                    tracking = smoothing["x"] is not None
                    if tracking:
                        near = [f for f in faces
                                if abs(f[0] + f[2] / 2 - smoothing["x"]) < 140
                                and abs(f[1] + f[3] / 2 - smoothing["y"]) < 140]
                        if near:
                            selected = max(near, key=lambda f: f[2])
                        bigger = [f for f in faces
                                  if f[2] > 1.5 * smoothing["w"]
                                  and f not in near]
                        if bigger and selected is not None:
                            switch_streak += 1
                            if switch_streak >= 5:
                                # a much closer person persisted: switch target
                                selected = max(bigger, key=lambda f: f[2])
                                smoothing = {"x": None, "y": None, "w": None}
                                switch_streak = 0
                                print("track_switch closer_person", flush=True)
                        else:
                            switch_streak = 0
                        if selected is None:
                            candidate_prev = None
                    else:
                        best = max(faces, key=lambda f: f[2])
                        if candidate_prev is not None and (
                                abs(best[0] - candidate_prev[0]) < 90
                                and abs(best[1] - candidate_prev[1]) < 90):
                            selected = best  # confirmed on consecutive frames
                        candidate_prev = best
                    if selected is not None:
                        x, y, w, h = selected
                        fx_c = x + w / 2.0
                        fy_c = y + h / 2.0
                        a = 0.45
                        for key, value in (("x", fx_c), ("y", fy_c), ("w", w)):
                            prev = smoothing[key]
                            smoothing[key] = value if prev is None else (
                                a * value + (1 - a) * prev)
                        last_seen = loop_t0
                if smoothing["x"] is not None and (
                        loop_t0 - last_seen) < cfg["person_lost_grace_s"]:
                    person = True
                    # Range from face width, then express the person's position
                    # in the head frame (head sits above/behind the camera).
                    dist = camera.fx * cfg["face_width_m"] / max(smoothing["w"], 8)
                    dist = max(0.3, min(4.0, dist))
                    px = (smoothing["x"] - camera.cx) / camera.fx * dist
                    py_up = (camera.cy - smoothing["y"]) / camera.fx * dist
                    z_head = dist + cfg["head_offset_back_m"]
                    y_head = py_up - cfg["head_offset_up_m"]
                    bearings = (math.atan2(px, z_head),
                                math.atan2(y_head, math.hypot(px, z_head)))
                    proximity = max(0.0, min(1.0, (1.6 - dist) / 1.3))
                    residual = (bearings[0] - yaw_rad_est,
                                bearings[1] - pitch_rad_est)
                elif (loop_t0 - last_seen) >= cfg["person_lost_grace_s"]:
                    smoothing = {"x": None, "y": None, "w": None}

            # Head at 10 Hz (every 2nd tick), eyes every tick (20 Hz).
            derate = head.thermal_derate if head is not None else False
            intent, desired4 = engine.compute(
                loop_t0, person, bearings if person else (0.0, 0.0),
                residual, proximity, derate)

            if head is not None and head.state != "INIT":
                head.step(desired4, loop_t0)
                yaw_rad_est = (head.offsets[YAW]
                               / (cfg["yaw_sign"] * cfg["yaw_ticks_per_rad"]))
                pitch_rad_est = (head.offsets[CURL]
                                 / (cfg["curl_sign"]
                                    * cfg["pitch_ticks_per_rad"]
                                    * cfg["curl_pitch_share"]))
                if tick % 20 == 0 and head.compliance is None:
                    head.telemetry_check()

            if eyes is not None:
                if head is not None and head.compliance is not None:
                    pet_state = head.compliance.state
                    if pet_state in (YIELDING, RELEASE_DWELL):
                        # Held contact gets a sustained, unmistakable warm
                        # squint. Head compliance still owns all movement;
                        # this is only its eye-level acknowledgement.
                        intent.update(
                            lid=330, pupil=720, brightness=900,
                            expression="greet", blink=False,
                            color=(255, 125, 175),
                        )
                    elif pet_state == RECOVERING:
                        intent.update(
                            lid=180, pupil=650, brightness=820,
                            expression="greet", color=(255, 175, 105),
                        )
                try:
                    eyes.apply(**intent)
                    eye_failures = 0
                except Kep2Error as exc:
                    eye_failures += 1
                    if eye_failures in (3, 10):
                        print(f"eyes_error {exc!r}; reacquiring", flush=True)
                        try:
                            eyes.start_session()
                        except Kep2Error as exc2:
                            print(f"eyes_reacquire_failed {exc2!r}", flush=True)
                    if eye_failures > 40:
                        print("eyes_disabled_after_failures", flush=True)
                        eyes.close()
                        eyes = None

            if args.snapshot and camera is not None and tick % 100 == 50:
                frame = camera.latest_frame
                latest_det = camera.latest
                if frame is not None:
                    annotated = frame.copy()
                    if latest_det is not None:
                        for (fx0, fy0, fw, fh) in latest_det[1]:
                            cv2.rectangle(annotated, (fx0, fy0),
                                          (fx0 + fw, fy0 + fh), (0, 255, 0), 2)
                    cv2.imwrite("/tmp/follow-snap.jpg", annotated)

            if tick % 100 == 0:
                state = head.state if head else "no-head"
                compliant = (head.compliance.state if head and head.compliance
                             else "inactive")
                print(f"status t={loop_t0 - started:6.1f}s head={state} "
                      f"compliant={compliant} "
                      f"eyes={engine.mode} person={person} "
                      f"prox={proximity:.2f}", flush=True)
            tick += 1
            elapsed = time.monotonic() - loop_t0
            time.sleep(max(0.0, 0.05 - elapsed))
    except Exception as exc:
        print(f"fault {exc!r}", flush=True)
        exit_code = 1
    finally:
        stop.set()
        if camera is not None:
            camera.stop_flag.set()
        if eyes is not None:
            try:
                eyes.apply(gaze_x=0, gaze_y=0, lid=100, pupil=550,
                           brightness=350, expression="neutral",
                           color=(60, 160, 190))
            except Exception:
                pass
            eyes.release(reason=1)
            eyes.close()
        if head is not None:
            head.park_and_release()  # parks at natural; torque stays ON
        if bus is not None:
            bus.close()
        if camera is not None:
            camera.join(timeout=6)
        print("shutdown_complete", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
