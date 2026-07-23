//! Same-process OAK/visual/IMU owner for attended base commissioning.
//!
//! The source opens the exact launch-bound OAK graph once, constructs the
//! tracker from retained runtime/model bytes, transforms correction-safe
//! camera increments into the declared base frame, and calibrates native OAK
//! gyroscope samples before yielding an aligned observation. No external
//! injector can manufacture these values or their static stream binding.

use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use kiko_expression_core::StreamEpochId;
use oak_sys::{
    CloseError as OakCloseError, ConnectionError as OakConnectionError, Device,
    ImageError as OakImageError, ImageFrame as OakImageFrame, ImuError as OakImuError,
    StreamId as OakStreamId,
};

use super::nano_base_commissioning::CommissioningExternalSignal;
use super::nano_base_commissioning_bootstrap::{
    AdmittedCommissioningObservationStream, CommissioningAlignedObservation,
    CommissioningObservationEvent, CommissioningObservationSource, CommissioningSamplingRequest,
    PreparedNanoBaseCommissioning,
};
use super::{
    NanoAccessoryComponentHealth, NanoAccessoryFaultWaitError, NanoAccessoryFrameStats,
    NanoAccessoryFrameSubmitOutcome, NanoAccessoryHealthPeriod, NanoAccessoryHealthPeriodError,
    NanoAccessoryHealthStatusError, NanoAccessoryTerminalFault, NanoAccessoryWorker,
    NanoAccessoryWorkerConfig, NanoAccessoryWorkerConfigError, NanoAccessoryWorkerExit,
    NanoAccessoryWorkerJoinError, NanoAccessoryWorkerStartError,
};
use crate::dataset::{Calibration, CameraIntrinsics};
use crate::{
    DownscaleFactor, KeyframePolicy, KeypointLimit, LightGlue, LmConfig, LocalBaConfig,
    LoopSubsystemConfig, OakImuAngularVelocity, PairingInputError, PairingWindowNs, Pose64,
    Pose64Error, RansacConfig, RectifiedStereo, SensorId, SlamTracker, StereoPairer, SuperPoint,
    TrackerConfig, TrackerError, TrackerInitError, TriangulationConfig, VisualIncrement,
    oak_to_frame, pin_ort_runtime_from_memory,
};

const OAK_POLL_TIMEOUT_MS: u32 = 10;
const OAK_IDLE_SLEEP: Duration = Duration::from_micros(500);
const OAK_BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(15);
const TRACKER_WARMUP_TIMEOUT: Duration = Duration::from_secs(30);
const ACCESSORY_HEALTH_PERIOD: Duration = Duration::from_secs(1);
const PAIRING_WINDOW_NS: u64 = 5_000_000;

/// Exact connected stream plus its sole live OAK owner.
#[must_use = "the OAK owner must be explicitly closed and its result inspected"]
pub struct PreparedCommissioningLiveObservation {
    pub stream: AdmittedCommissioningObservationStream,
    pub source: NanoCommissioningLiveObservationSource,
}

/// Open the exact launch-bound OAK and warm its visual/IMU source before any
/// attended attestation is consumed or STM32 motion authority is acquired.
pub fn prepare_commissioning_live_observation(
    prepared: &PreparedNanoBaseCommissioning,
    clock_origin: Instant,
    running: Arc<AtomicBool>,
    clock_epoch: super::nano_base_commissioning_bootstrap::CommissioningClockEpoch,
    accessory_stream_epoch: StreamEpochId,
) -> Result<PreparedCommissioningLiveObservation, CommissioningLiveOpenError> {
    if !running.load(Ordering::Acquire) {
        return Err(CommissioningLiveOpenError::before_oak(
            CommissioningLiveOpenPrimaryError::Interrupted,
        ));
    }

    let live_graph = prepared.live_graph();
    let inference = live_graph.inference();
    let inputs = prepared.loaded_inputs();
    let runtime = pin_ort_runtime_from_memory(inputs.onnx_runtime_library.bytes())
        .map_err(CommissioningLiveOpenPrimaryError::Inference)
        .map_err(CommissioningLiveOpenError::before_oak)?;
    let superpoint_left = SuperPoint::new_from_memory_with_backend(
        inputs.superpoint_model.bytes(),
        runtime,
        inference.superpoint_backend().runtime(),
    )
    .map_err(CommissioningLiveOpenPrimaryError::Inference)
    .map_err(CommissioningLiveOpenError::before_oak)?;
    let superpoint_right = SuperPoint::new_from_memory_with_backend(
        inputs.superpoint_model.bytes(),
        runtime,
        inference.superpoint_backend().runtime(),
    )
    .map_err(CommissioningLiveOpenPrimaryError::Inference)
    .map_err(CommissioningLiveOpenError::before_oak)?;
    let lightglue = LightGlue::new_from_memory_with_backend(
        inputs.lightglue_model.bytes(),
        runtime,
        inference.lightglue_backend().runtime(),
    )
    .map_err(CommissioningLiveOpenPrimaryError::Inference)
    .map_err(CommissioningLiveOpenError::before_oak)?;

    let health_period = NanoAccessoryHealthPeriod::try_from_duration(ACCESSORY_HEALTH_PERIOD)
        .map_err(CommissioningLiveOpenPrimaryError::AccessoryHealthPeriod)
        .map_err(CommissioningLiveOpenError::before_oak)?;
    let accessory_config = NanoAccessoryWorkerConfig::from_manifest_bound_policy(
        prepared.accessory_policy(),
        accessory_stream_epoch,
        health_period,
    )
    .map_err(CommissioningLiveOpenPrimaryError::AccessoryConfig)
    .map_err(CommissioningLiveOpenError::before_oak)?;
    let accessory = NanoAccessoryWorker::start(accessory_config)
        .map_err(CommissioningLiveOpenPrimaryError::AccessoryStart)
        .map_err(CommissioningLiveOpenError::before_oak)?;
    let mut accessory_guard = Some(accessory);

    let calibration = prepared.calibration().clone();
    let device = match Device::connect(
        calibration.oak_mxid().as_str(),
        live_graph.oak().device_config(),
    ) {
        Ok(device) => device,
        Err(source) => {
            let accessory_shutdown = accessory_guard
                .take()
                .and_then(|worker| shutdown_accessory(worker).err());
            return Err(CommissioningLiveOpenError {
                primary: Box::new(CommissioningLiveOpenPrimaryError::OakConnect(source)),
                oak_close: None,
                accessory_shutdown,
            });
        }
    };

    let mut device_guard = Some(device);
    let connected = (|| {
        let device = device_guard
            .as_mut()
            .expect("connected OAK remains in the preparation guard");
        let connected_mxid = device
            .connected_identity()
            .map_err(CommissioningLiveOpenPrimaryError::OakIdentity)?
            .mxid()
            .to_owned();
        calibration
            .require_connected_oak_mxid(&connected_mxid)
            .map_err(CommissioningLiveOpenPrimaryError::Calibration)?;
        device
            .usb_transport_evidence()
            .map_err(CommissioningLiveOpenPrimaryError::OakUsb)?;

        let (left, right, observed_stereo) =
            bootstrap_stereo(&mut *device, live_graph.oak().rectified_stereo(), &running)?;
        calibration
            .require_observed_stereo(&observed_stereo)
            .map_err(CommissioningLiveOpenPrimaryError::Calibration)?;
        let rectified = RectifiedStereo::from_calibration(&observed_stereo)
            .map_err(CommissioningLiveOpenPrimaryError::Stereo)?;
        let tracker_config = commissioning_tracker_config(
            inference.maximum_keypoints(),
            inference.downscale_factor(),
        )?;
        let tracker = SlamTracker::try_new(
            superpoint_left,
            superpoint_right,
            lightglue,
            rectified,
            tracker_config,
        )
        .map_err(CommissioningLiveOpenPrimaryError::TrackerInit)?;
        let mut pairer = StereoPairer::new_with_max_pending(
            PairingWindowNs::try_from_u64(PAIRING_WINDOW_NS)
                .map_err(CommissioningLiveOpenPrimaryError::PairingConfig)?,
            usize::try_from(live_graph.oak().queue_size())
                .map_err(|_| CommissioningLiveOpenPrimaryError::QueueSizeOutOfRange)?,
        )
        .map_err(CommissioningLiveOpenPrimaryError::PairingConfig)?;
        pairer
            .push_left(
                oak_to_frame(left, SensorId::StereoLeft)
                    .map_err(CommissioningLiveOpenPrimaryError::Frame)?,
            )
            .map_err(CommissioningLiveOpenPrimaryError::Pairing)?;
        pairer
            .push_right(
                oak_to_frame(right, SensorId::StereoRight)
                    .map_err(CommissioningLiveOpenPrimaryError::Frame)?,
            )
            .map_err(CommissioningLiveOpenPrimaryError::Pairing)?;

        let visual_source_id = prepared.expected_visual_velocity_source_id();
        let stream = prepared
            .admit_same_owner_stream(
                &connected_mxid,
                &observed_stereo,
                visual_source_id.as_str(),
                clock_epoch,
            )
            .map_err(CommissioningLiveOpenPrimaryError::StreamAdmission)?;
        let sample_timeout = Duration::from_nanos(prepared.maximum_sample_gap_ns().get());
        let minimum_sample_period = Duration::from_nanos(prepared.minimum_sample_period_ns().get());
        let mut engine = LiveObservationEngine {
            device: device_guard.take(),
            pairer,
            tracker,
            raw_imu_calibration: calibration.raw_imu_calibration().clone(),
            tracking_to_base: Pose64::try_from_pose32(calibration.tracking_camera_to_base().pose())
                .map_err(CommissioningLiveOpenPrimaryError::Pose)?,
            expected_width: live_graph.oak().rectified_stereo().width_px(),
            expected_height: live_graph.oak().rectified_stereo().height_px(),
            expected_rgb_width: live_graph.oak().rgb().width_px(),
            expected_rgb_height: live_graph.oak().rgb().height_px(),
            clock_origin,
            running,
            latest_imu: None,
            last_imu_sequence: None,
            pending_visual: None,
            sample_timeout,
            minimum_sample_period,
            last_emitted_visual_observed_at_ns: None,
            accessory: accessory_guard.take(),
        };
        if let Err(source) = engine.next_observation_with_timeout(TRACKER_WARMUP_TIMEOUT) {
            let cleanup = engine.close_resources().err();
            return Err(CommissioningLiveOpenPrimaryError::Warmup {
                source: Box::new(source),
                cleanup: cleanup.map(Box::new),
            });
        }
        if let Err(source) = engine.require_accessory_ready(TRACKER_WARMUP_TIMEOUT) {
            let cleanup = engine.close_resources().err();
            return Err(CommissioningLiveOpenPrimaryError::Warmup {
                source: Box::new(source),
                cleanup: cleanup.map(Box::new),
            });
        }
        // Warmup proves that the graph can produce both modalities but is not
        // commissioning evidence. Require a fresh visual/IMU pair for the
        // first controller-owned sample.
        engine.reset_after_warmup();
        Ok((stream, engine))
    })();

    match connected {
        Ok((stream, engine)) => Ok(PreparedCommissioningLiveObservation {
            source: NanoCommissioningLiveObservationSource {
                stream: stream.clone(),
                engine,
            },
            stream,
        }),
        Err(primary) => {
            let oak_close = device_guard.and_then(|device| device.close().err());
            let accessory_shutdown = accessory_guard
                .take()
                .and_then(|worker| shutdown_accessory(worker).err());
            Err(CommissioningLiveOpenError {
                primary: Box::new(primary),
                oak_close,
                accessory_shutdown,
            })
        }
    }
}

fn commissioning_tracker_config(
    maximum_keypoints: u32,
    downscale_factor: u32,
) -> Result<TrackerConfig, CommissioningLiveOpenPrimaryError> {
    let max_keypoints = KeypointLimit::try_from(
        usize::try_from(maximum_keypoints)
            .map_err(|_| CommissioningLiveOpenPrimaryError::KeypointLimitOutOfRange)?,
    )
    .map_err(CommissioningLiveOpenPrimaryError::KeypointLimit)?;
    let downscale = DownscaleFactor::new(
        NonZeroU32::new(downscale_factor)
            .ok_or(CommissioningLiveOpenPrimaryError::ZeroDownscale)?,
    );
    let defaults = RansacConfig::default();
    let ransac = RansacConfig::try_new(
        defaults.max_iterations(),
        defaults.reprojection_threshold_px(),
        15,
        defaults.seed(),
    )
    .map_err(CommissioningLiveOpenPrimaryError::Ransac)?;
    let keyframe_policy =
        KeyframePolicy::new(20, 15.0, 0.5).map_err(CommissioningLiveOpenPrimaryError::Keyframe)?;
    let ba = LocalBaConfig::new(10, 6, 8, 3.0, LmConfig::default())
        .map_err(CommissioningLiveOpenPrimaryError::LocalBa)?;
    Ok(TrackerConfig {
        max_keypoints,
        downscale,
        min_keyframe_points: 80,
        ransac,
        triangulation: TriangulationConfig::default(),
        keyframe_policy,
        ba,
        redundancy: None,
        backend: None,
        loop_subsystem: LoopSubsystemConfig::Disabled,
    })
}

fn bootstrap_stereo(
    device: &mut Device,
    expected: super::NanoOakImageStream,
    running: &AtomicBool,
) -> Result<(OakImageFrame, OakImageFrame, Calibration), CommissioningLiveOpenPrimaryError> {
    let started = Instant::now();
    let mut left = None;
    let mut right = None;
    while left.is_none() || right.is_none() {
        if !running.load(Ordering::Acquire) {
            return Err(CommissioningLiveOpenPrimaryError::Interrupted);
        }
        if started.elapsed() >= OAK_BOOTSTRAP_TIMEOUT {
            return Err(CommissioningLiveOpenPrimaryError::StereoBootstrapTimedOut {
                received_left: left.is_some(),
                received_right: right.is_some(),
            });
        }
        let mut received = false;
        if left.is_none() {
            match device.mono_left(OAK_POLL_TIMEOUT_MS) {
                Ok(frame) => {
                    validate_mono_frame(&frame, OakStreamId::MonoLeft, expected)?;
                    left = Some(frame);
                    received = true;
                }
                Err(OakImageError::Timeout { .. } | OakImageError::QueueEmpty) => {}
                Err(source) => {
                    return Err(CommissioningLiveOpenPrimaryError::OakLeft(source));
                }
            }
        }
        if right.is_none() {
            match device.mono_right(OAK_POLL_TIMEOUT_MS) {
                Ok(frame) => {
                    validate_mono_frame(&frame, OakStreamId::MonoRight, expected)?;
                    right = Some(frame);
                    received = true;
                }
                Err(OakImageError::Timeout { .. } | OakImageError::QueueEmpty) => {}
                Err(source) => {
                    return Err(CommissioningLiveOpenPrimaryError::OakRight(source));
                }
            }
        }
        if !received {
            thread::sleep(OAK_IDLE_SLEEP);
        }
    }

    let left = left.expect("loop exits only after left frame");
    let right = right.expect("loop exits only after right frame");
    let left_intrinsics = left.intrinsics();
    let right_intrinsics = right.intrinsics();
    let baseline_m = device
        .stereo_baseline_m()
        .map_err(CommissioningLiveOpenPrimaryError::OakStereoCalibration)?;
    let observed = Calibration {
        left: CameraIntrinsics {
            fx: left_intrinsics.fx(),
            fy: left_intrinsics.fy(),
            cx: left_intrinsics.cx(),
            cy: left_intrinsics.cy(),
            width: left_intrinsics.width(),
            height: left_intrinsics.height(),
        },
        right: CameraIntrinsics {
            fx: right_intrinsics.fx(),
            fy: right_intrinsics.fy(),
            cx: right_intrinsics.cx(),
            cy: right_intrinsics.cy(),
            width: right_intrinsics.width(),
            height: right_intrinsics.height(),
        },
        baseline_m,
        rectified: true,
    };
    Ok((left, right, observed))
}

fn validate_mono_frame(
    frame: &OakImageFrame,
    expected_stream: OakStreamId,
    expected: super::NanoOakImageStream,
) -> Result<(), CommissioningLiveOpenPrimaryError> {
    if frame.stream != expected_stream {
        return Err(CommissioningLiveOpenPrimaryError::UnexpectedMonoStream {
            expected: expected_stream,
            actual: frame.stream,
        });
    }
    if frame.width != expected.width_px() || frame.height != expected.height_px() {
        return Err(CommissioningLiveOpenPrimaryError::MonoDimensionsMismatch {
            expected_width: expected.width_px(),
            expected_height: expected.height_px(),
            actual_width: frame.width,
            actual_height: frame.height,
        });
    }
    Ok(())
}

struct LiveObservationEngine {
    device: Option<Device>,
    pairer: StereoPairer,
    tracker: SlamTracker,
    raw_imu_calibration: super::RawImuCalibration,
    tracking_to_base: Pose64,
    expected_width: u32,
    expected_height: u32,
    expected_rgb_width: u32,
    expected_rgb_height: u32,
    clock_origin: Instant,
    running: Arc<AtomicBool>,
    latest_imu: Option<TimedYawRate>,
    last_imu_sequence: Option<u32>,
    pending_visual: Option<TimedBodyVelocity>,
    sample_timeout: Duration,
    minimum_sample_period: Duration,
    last_emitted_visual_observed_at_ns: Option<u64>,
    accessory: Option<NanoAccessoryWorker>,
}

#[derive(Clone, Copy)]
struct TimedYawRate {
    observed_at_ns: u64,
    yaw_rate_rad_s: f64,
}

#[derive(Clone, Copy)]
struct TimedBodyVelocity {
    observed_at_ns: u64,
    forward_mps: f64,
    lateral_mps: f64,
}

impl LiveObservationEngine {
    fn reset_after_warmup(&mut self) {
        self.latest_imu = None;
        self.pending_visual = None;
        self.last_emitted_visual_observed_at_ns = None;
    }

    fn close_resources(&mut self) -> Result<(), CommissioningLiveCloseError> {
        // The base owner is stopped by the caller before this method. Close
        // the sole OAK first, then release eye/head serial ownership without
        // writing the head torque switch.
        let oak_close = self.device.take().and_then(|device| device.close().err());
        let accessory_shutdown = self
            .accessory
            .take()
            .and_then(|worker| shutdown_accessory(worker).err());
        if oak_close.is_none() && accessory_shutdown.is_none() {
            Ok(())
        } else {
            Err(CommissioningLiveCloseError {
                oak_close,
                accessory_shutdown,
            })
        }
    }

    fn next_observation_with_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<CommissioningAlignedObservation, CommissioningLiveSourceError> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(CommissioningLiveSourceError::DeadlineOutOfRange)?;
        loop {
            if !self.running.load(Ordering::Acquire) {
                return Err(CommissioningLiveSourceError::Interrupted);
            }
            self.service_accessory()?;
            self.service_imu()?;
            if let (Some(visual), Some(imu)) = (self.pending_visual, self.latest_imu) {
                let now_ns = self.host_now_ns()?;
                self.pending_visual = None;
                return CommissioningAlignedObservation::try_new(
                    now_ns,
                    visual.observed_at_ns,
                    imu.observed_at_ns,
                    visual.forward_mps,
                    visual.lateral_mps,
                    imu.yaw_rate_rad_s,
                )
                .map_err(CommissioningLiveSourceError::AlignedObservation);
            }
            if Instant::now() >= deadline {
                return Err(CommissioningLiveSourceError::ObservationTimedOut { timeout });
            }

            let mut received = false;
            match self.device_mut()?.mono_left(OAK_POLL_TIMEOUT_MS) {
                Ok(frame) => {
                    self.validate_runtime_frame(&frame, OakStreamId::MonoLeft)?;
                    self.pairer
                        .push_left(
                            oak_to_frame(frame, SensorId::StereoLeft)
                                .map_err(CommissioningLiveSourceError::Frame)?,
                        )
                        .map_err(CommissioningLiveSourceError::Pairing)?;
                    received = true;
                }
                Err(OakImageError::Timeout { .. } | OakImageError::QueueEmpty) => {}
                Err(source) => return Err(CommissioningLiveSourceError::OakLeft(source)),
            }
            match self.device_mut()?.mono_right(OAK_POLL_TIMEOUT_MS) {
                Ok(frame) => {
                    self.validate_runtime_frame(&frame, OakStreamId::MonoRight)?;
                    self.pairer
                        .push_right(
                            oak_to_frame(frame, SensorId::StereoRight)
                                .map_err(CommissioningLiveSourceError::Frame)?,
                        )
                        .map_err(CommissioningLiveSourceError::Pairing)?;
                    received = true;
                }
                Err(OakImageError::Timeout { .. } | OakImageError::QueueEmpty) => {}
                Err(source) => return Err(CommissioningLiveSourceError::OakRight(source)),
            }
            if let Some(pair) = self.pairer.next_pair() {
                let output = self
                    .tracker
                    .process(pair)
                    .map_err(|source| CommissioningLiveSourceError::Tracker(Box::new(source)))?;
                if let Some(increment) = output.visual_increment() {
                    let (forward_mps, lateral_mps) =
                        body_velocity_from_visual_increment(increment, self.tracking_to_base)?;
                    let observed_at_ns = self.host_now_ns()?;
                    let elapsed_ns = self
                        .last_emitted_visual_observed_at_ns
                        .map(|previous| {
                            observed_at_ns.checked_sub(previous).ok_or(
                                CommissioningLiveSourceError::HostClockRegression {
                                    previous_ns: previous,
                                    current_ns: observed_at_ns,
                                },
                            )
                        })
                        .transpose()?;
                    let minimum_ns = u64::try_from(self.minimum_sample_period.as_nanos())
                        .expect("parsed minimum sample period is bounded to u64 nanoseconds");
                    if elapsed_ns.is_none_or(|elapsed| elapsed >= minimum_ns) {
                        self.pending_visual = Some(TimedBodyVelocity {
                            observed_at_ns,
                            forward_mps,
                            lateral_mps,
                        });
                        self.last_emitted_visual_observed_at_ns = Some(observed_at_ns);
                        self.service_imu()?;
                    }
                }
            } else if !received {
                thread::sleep(OAK_IDLE_SLEEP);
            }
            if Instant::now() >= deadline {
                return Err(CommissioningLiveSourceError::ObservationTimedOut { timeout });
            }
        }
    }

    fn require_accessory_ready(
        &mut self,
        timeout: Duration,
    ) -> Result<(), CommissioningLiveSourceError> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(CommissioningLiveSourceError::DeadlineOutOfRange)?;
        loop {
            self.service_accessory()?;
            let accessory = self.accessory_ref()?;
            let health = accessory
                .health_observer()
                .snapshot()
                .map_err(CommissioningLiveSourceError::AccessoryHealth)?;
            if health.head == NanoAccessoryComponentHealth::Ready
                && health.eyes == NanoAccessoryComponentHealth::Ready
                && health.rgb_expression == NanoAccessoryComponentHealth::Ready
                && health.successful_rgb_expression_frames > 0
            {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(CommissioningLiveSourceError::AccessoryWarmupTimedOut {
                    timeout,
                    health,
                });
            }
            thread::sleep(OAK_IDLE_SLEEP);
        }
    }

    fn service_accessory(&mut self) -> Result<(), CommissioningLiveSourceError> {
        if let Some(fault) = self
            .accessory_ref()?
            .try_terminal_fault()
            .map_err(CommissioningLiveSourceError::AccessoryFaultMonitor)?
        {
            return Err(CommissioningLiveSourceError::AccessoryTerminalFault(
                Box::new(fault),
            ));
        }
        let frame = match self.device_mut()?.rgb(OAK_POLL_TIMEOUT_MS) {
            Ok(frame) => frame,
            Err(OakImageError::Timeout { .. } | OakImageError::QueueEmpty) => return Ok(()),
            Err(source) => return Err(CommissioningLiveSourceError::OakRgb(source)),
        };
        if frame.stream != OakStreamId::Rgb {
            return Err(CommissioningLiveSourceError::UnexpectedRgbStream {
                actual: frame.stream,
            });
        }
        if frame.width != self.expected_rgb_width || frame.height != self.expected_rgb_height {
            return Err(CommissioningLiveSourceError::RgbDimensionsMismatch {
                expected_width: self.expected_rgb_width,
                expected_height: self.expected_rgb_height,
                actual_width: frame.width,
                actual_height: frame.height,
            });
        }
        match self.accessory_mut()?.submit_rgb(frame) {
            NanoAccessoryFrameSubmitOutcome::Enqueued
            | NanoAccessoryFrameSubmitOutcome::ReplacedOlderFrame => Ok(()),
            outcome => Err(CommissioningLiveSourceError::AccessoryRgbRejected(outcome)),
        }
    }

    fn service_imu(&mut self) -> Result<(), CommissioningLiveSourceError> {
        let samples = match self.device_mut()?.imu() {
            Ok(samples) => samples,
            Err(OakImuError::Empty) => return Ok(()),
            Err(source) => return Err(CommissioningLiveSourceError::OakImu(source)),
        };
        for sample in samples {
            if self
                .last_imu_sequence
                .is_some_and(|previous| sample.sequence <= previous)
            {
                return Err(CommissioningLiveSourceError::NonIncreasingImuSequence {
                    previous: self.last_imu_sequence.expect("checked Some"),
                    current: sample.sequence,
                });
            }
            let raw = OakImuAngularVelocity::try_new(
                f64::from(sample.gyro.x),
                f64::from(sample.gyro.y),
                f64::from(sample.gyro.z),
            )
            .map_err(CommissioningLiveSourceError::ImuValue)?;
            let calibrated = self
                .raw_imu_calibration
                .calibrate_angular_velocity(raw)
                .map_err(CommissioningLiveSourceError::ImuCalibration)?;
            self.latest_imu = Some(TimedYawRate {
                observed_at_ns: self.host_now_ns()?,
                yaw_rate_rad_s: calibrated.yaw_rate_rad_per_sec(),
            });
            self.last_imu_sequence = Some(sample.sequence);
        }
        Ok(())
    }

    fn validate_runtime_frame(
        &self,
        frame: &OakImageFrame,
        expected_stream: OakStreamId,
    ) -> Result<(), CommissioningLiveSourceError> {
        if frame.stream != expected_stream {
            return Err(CommissioningLiveSourceError::UnexpectedMonoStream {
                expected: expected_stream,
                actual: frame.stream,
            });
        }
        if frame.width != self.expected_width || frame.height != self.expected_height {
            return Err(CommissioningLiveSourceError::MonoDimensionsMismatch {
                expected_width: self.expected_width,
                expected_height: self.expected_height,
                actual_width: frame.width,
                actual_height: frame.height,
            });
        }
        Ok(())
    }

    fn host_now_ns(&self) -> Result<u64, CommissioningLiveSourceError> {
        let elapsed = self.clock_origin.elapsed().as_nanos();
        let value = u64::try_from(elapsed)
            .map_err(|_| CommissioningLiveSourceError::HostClockOutOfRange(elapsed))?;
        if value == 0 {
            return Err(CommissioningLiveSourceError::HostClockZero);
        }
        Ok(value)
    }

    fn device_mut(&mut self) -> Result<&mut Device, CommissioningLiveSourceError> {
        self.device
            .as_mut()
            .ok_or(CommissioningLiveSourceError::OakAlreadyClosed)
    }

    fn accessory_ref(&self) -> Result<&NanoAccessoryWorker, CommissioningLiveSourceError> {
        self.accessory
            .as_ref()
            .ok_or(CommissioningLiveSourceError::AccessoryAlreadyClosed)
    }

    fn accessory_mut(&mut self) -> Result<&mut NanoAccessoryWorker, CommissioningLiveSourceError> {
        self.accessory
            .as_mut()
            .ok_or(CommissioningLiveSourceError::AccessoryAlreadyClosed)
    }
}

/// Same-owner live source consumed directly by the commissioning runtime.
#[must_use = "the source owns the exact OAK and must be explicitly closed"]
pub struct NanoCommissioningLiveObservationSource {
    stream: AdmittedCommissioningObservationStream,
    engine: LiveObservationEngine,
}

impl NanoCommissioningLiveObservationSource {
    pub fn accessory_frame_stats(&self) -> NanoAccessoryFrameStats {
        self.engine.accessory.as_ref().map_or_else(
            NanoAccessoryFrameStats::default,
            NanoAccessoryWorker::frame_stats,
        )
    }

    pub fn close(mut self) -> Result<(), CommissioningLiveCloseError> {
        self.engine.close_resources()
    }
}

impl CommissioningObservationSource for NanoCommissioningLiveObservationSource {
    type Error = CommissioningLiveSourceError;

    fn stream_binding(&self) -> &AdmittedCommissioningObservationStream {
        &self.stream
    }

    fn next_observation(
        &mut self,
        _request: CommissioningSamplingRequest,
    ) -> Result<CommissioningObservationEvent, Self::Error> {
        if !self.engine.running.load(Ordering::Acquire) {
            return Ok(CommissioningObservationEvent::Terminal(
                CommissioningExternalSignal::CancelledByOperator,
            ));
        }
        self.engine
            .next_observation_with_timeout(self.engine.sample_timeout)
            .map(CommissioningObservationEvent::Observation)
    }

    fn terminal_signal_for_error(&self, error: &Self::Error) -> CommissioningExternalSignal {
        error.terminal_signal()
    }
}

fn body_velocity_from_visual_increment(
    increment: VisualIncrement,
    tracking_to_base: Pose64,
) -> Result<(f64, f64), CommissioningLiveSourceError> {
    let from_ns = increment.from().timestamp().as_nanos();
    let to_ns = increment.to().timestamp().as_nanos();
    let dt_ns = to_ns
        .checked_sub(from_ns)
        .filter(|value| *value > 0)
        .ok_or(CommissioningLiveSourceError::VisualTimestampDidNotAdvance { from_ns, to_ns })?;
    body_velocity_from_camera_increment(
        increment.previous_camera_to_current_camera(),
        tracking_to_base,
        u64::try_from(dt_ns).map_err(|_| {
            CommissioningLiveSourceError::VisualTimestampDidNotAdvance { from_ns, to_ns }
        })?,
    )
}

fn body_velocity_from_camera_increment(
    previous_camera_to_current_camera: Pose64,
    tracking_to_base: Pose64,
    dt_ns: u64,
) -> Result<(f64, f64), CommissioningLiveSourceError> {
    if dt_ns == 0 {
        return Err(CommissioningLiveSourceError::ZeroVisualDuration);
    }
    let current_camera_to_previous_camera = previous_camera_to_current_camera
        .try_inverse()
        .map_err(CommissioningLiveSourceError::Pose)?;
    let current_base_to_previous_base = tracking_to_base
        .try_compose(current_camera_to_previous_camera)
        .and_then(|value| value.try_compose(tracking_to_base.try_inverse()?))
        .map_err(CommissioningLiveSourceError::Pose)?;
    let translation = current_base_to_previous_base.translation();
    let dt_s = dt_ns as f64 / 1_000_000_000.0;
    let forward = translation[0] / dt_s;
    let lateral = translation[1] / dt_s;
    if !forward.is_finite() || !lateral.is_finite() {
        return Err(CommissioningLiveSourceError::BodyVelocityNonFinite {
            forward_mps: forward,
            lateral_mps: lateral,
        });
    }
    Ok((forward, lateral))
}

#[derive(Debug)]
pub struct CommissioningLiveCloseError {
    pub oak_close: Option<OakCloseError>,
    pub accessory_shutdown: Option<CommissioningAccessoryShutdownError>,
}

impl fmt::Display for CommissioningLiveCloseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "commissioning live resources did not close cleanly: oak_close={:?}; accessory_shutdown={:?}",
            self.oak_close, self.accessory_shutdown
        )
    }
}

impl std::error::Error for CommissioningLiveCloseError {}

#[derive(Debug)]
pub enum CommissioningAccessoryShutdownError {
    Join(NanoAccessoryWorkerJoinError),
    UnverifiedExit(Box<NanoAccessoryWorkerExit>),
}

impl fmt::Display for CommissioningAccessoryShutdownError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "commissioning accessory owner did not prove eye release and hold-preserving head release: {self:?}"
        )
    }
}

impl std::error::Error for CommissioningAccessoryShutdownError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Join(source) => Some(source),
            Self::UnverifiedExit(_) => None,
        }
    }
}

fn shutdown_accessory(
    worker: NanoAccessoryWorker,
) -> Result<(), CommissioningAccessoryShutdownError> {
    let exit = worker
        .shutdown()
        .map_err(CommissioningAccessoryShutdownError::Join)?;
    match &exit {
        NanoAccessoryWorkerExit::Shutdown {
            terminal_fault: None,
            evidence,
        } if evidence.eye().release_verified()
            && evidence.head().hold_preserving_release_completed() =>
        {
            Ok(())
        }
        _ => Err(CommissioningAccessoryShutdownError::UnverifiedExit(
            Box::new(exit),
        )),
    }
}

#[derive(Debug)]
pub struct CommissioningLiveOpenError {
    pub primary: Box<CommissioningLiveOpenPrimaryError>,
    pub oak_close: Option<OakCloseError>,
    pub accessory_shutdown: Option<CommissioningAccessoryShutdownError>,
}

impl CommissioningLiveOpenError {
    fn before_oak(primary: CommissioningLiveOpenPrimaryError) -> Self {
        Self {
            primary: Box::new(primary),
            oak_close: None,
            accessory_shutdown: None,
        }
    }
}

impl fmt::Display for CommissioningLiveOpenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "commissioning live-graph preparation failed: {}; oak_close={:?}; accessory_shutdown={:?}",
            self.primary, self.oak_close, self.accessory_shutdown
        )
    }
}

impl std::error::Error for CommissioningLiveOpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.primary.as_ref())
    }
}

#[derive(Debug)]
pub enum CommissioningLiveOpenPrimaryError {
    Interrupted,
    Inference(crate::InferenceError),
    OakConnect(OakConnectionError),
    OakIdentity(oak_sys::ConnectedDeviceIdentityError),
    OakUsb(oak_sys::UsbTransportEvidenceError),
    OakLeft(OakImageError),
    OakRight(OakImageError),
    OakStereoCalibration(oak_sys::CalibrationError),
    Calibration(super::NanoCalibrationBindingError),
    Stereo(crate::RectifiedStereoError),
    StereoBootstrapTimedOut {
        received_left: bool,
        received_right: bool,
    },
    UnexpectedMonoStream {
        expected: OakStreamId,
        actual: OakStreamId,
    },
    MonoDimensionsMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    QueueSizeOutOfRange,
    KeypointLimitOutOfRange,
    KeypointLimit(crate::KeypointLimitError),
    ZeroDownscale,
    Ransac(crate::RansacConfigError),
    Keyframe(crate::KeyframePolicyError),
    LocalBa(crate::LocalBaConfigError),
    TrackerInit(TrackerInitError),
    PairingConfig(crate::PairingConfigError),
    Frame(crate::FrameError),
    Pairing(PairingInputError),
    Pose(Pose64Error),
    AccessoryHealthPeriod(NanoAccessoryHealthPeriodError),
    AccessoryConfig(NanoAccessoryWorkerConfigError),
    AccessoryStart(NanoAccessoryWorkerStartError),
    Warmup {
        source: Box<CommissioningLiveSourceError>,
        cleanup: Option<Box<CommissioningLiveCloseError>>,
    },
    StreamAdmission(super::nano_base_commissioning_bootstrap::CommissioningStreamAdmissionError),
}

impl fmt::Display for CommissioningLiveOpenPrimaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "commissioning live-graph error: {self:?}")
    }
}

impl std::error::Error for CommissioningLiveOpenPrimaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Inference(source) => Some(source),
            Self::OakConnect(source) => Some(source),
            Self::OakIdentity(source) => Some(source),
            Self::OakUsb(source) => Some(source),
            Self::OakLeft(source) | Self::OakRight(source) => Some(source),
            Self::OakStereoCalibration(source) => Some(source),
            Self::Calibration(source) => Some(source),
            Self::Stereo(source) => Some(source),
            Self::KeypointLimit(source) => Some(source),
            Self::Ransac(source) => Some(source),
            Self::Keyframe(source) => Some(source),
            Self::LocalBa(source) => Some(source),
            Self::TrackerInit(source) => Some(source),
            Self::PairingConfig(source) => Some(source),
            Self::Frame(source) => Some(source),
            Self::Pairing(source) => Some(source),
            Self::Pose(source) => Some(source),
            Self::AccessoryHealthPeriod(source) => Some(source),
            Self::AccessoryConfig(source) => Some(source),
            Self::AccessoryStart(source) => Some(source),
            Self::Warmup { source, .. } => Some(source.as_ref()),
            Self::StreamAdmission(source) => Some(source),
            Self::Interrupted
            | Self::StereoBootstrapTimedOut { .. }
            | Self::UnexpectedMonoStream { .. }
            | Self::MonoDimensionsMismatch { .. }
            | Self::QueueSizeOutOfRange
            | Self::KeypointLimitOutOfRange
            | Self::ZeroDownscale => None,
        }
    }
}

#[derive(Debug)]
pub enum CommissioningLiveSourceError {
    Interrupted,
    DeadlineOutOfRange,
    ObservationTimedOut {
        timeout: Duration,
    },
    OakAlreadyClosed,
    AccessoryAlreadyClosed,
    OakLeft(OakImageError),
    OakRight(OakImageError),
    OakRgb(OakImageError),
    OakImu(OakImuError),
    UnexpectedRgbStream {
        actual: OakStreamId,
    },
    RgbDimensionsMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    AccessoryFaultMonitor(NanoAccessoryFaultWaitError),
    AccessoryTerminalFault(Box<NanoAccessoryTerminalFault>),
    AccessoryHealth(NanoAccessoryHealthStatusError),
    AccessoryWarmupTimedOut {
        timeout: Duration,
        health: super::NanoAccessoryRuntimeHealth,
    },
    AccessoryRgbRejected(NanoAccessoryFrameSubmitOutcome),
    UnexpectedMonoStream {
        expected: OakStreamId,
        actual: OakStreamId,
    },
    MonoDimensionsMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    Frame(crate::FrameError),
    Pairing(PairingInputError),
    Tracker(Box<TrackerError>),
    ImuValue(crate::InertialValueError),
    ImuCalibration(super::RawImuCalibrationError),
    NonIncreasingImuSequence {
        previous: u32,
        current: u32,
    },
    HostClockOutOfRange(u128),
    HostClockZero,
    HostClockRegression {
        previous_ns: u64,
        current_ns: u64,
    },
    VisualTimestampDidNotAdvance {
        from_ns: i64,
        to_ns: i64,
    },
    ZeroVisualDuration,
    Pose(Pose64Error),
    BodyVelocityNonFinite {
        forward_mps: f64,
        lateral_mps: f64,
    },
    AlignedObservation(
        super::nano_base_commissioning_bootstrap::CommissioningAlignedObservationError,
    ),
}

impl CommissioningLiveSourceError {
    fn terminal_signal(&self) -> CommissioningExternalSignal {
        match self {
            Self::Interrupted => CommissioningExternalSignal::CancelledByOperator,
            Self::OakImu(_)
            | Self::ImuValue(_)
            | Self::ImuCalibration(_)
            | Self::NonIncreasingImuSequence { .. } => CommissioningExternalSignal::ImuFault,
            Self::OakLeft(_)
            | Self::OakRight(_)
            | Self::OakRgb(_)
            | Self::UnexpectedRgbStream { .. }
            | Self::RgbDimensionsMismatch { .. }
            | Self::UnexpectedMonoStream { .. }
            | Self::MonoDimensionsMismatch { .. }
            | Self::Frame(_)
            | Self::Pairing(_)
            | Self::Tracker(_)
            | Self::VisualTimestampDidNotAdvance { .. }
            | Self::ZeroVisualDuration
            | Self::Pose(_)
            | Self::BodyVelocityNonFinite { .. } => CommissioningExternalSignal::VisualFault,
            Self::AccessoryFaultMonitor(_)
            | Self::AccessoryTerminalFault(_)
            | Self::AccessoryHealth(_)
            | Self::AccessoryWarmupTimedOut { .. }
            | Self::AccessoryRgbRejected(_)
            | Self::DeadlineOutOfRange
            | Self::ObservationTimedOut { .. }
            | Self::OakAlreadyClosed
            | Self::AccessoryAlreadyClosed
            | Self::HostClockOutOfRange(_)
            | Self::HostClockZero
            | Self::HostClockRegression { .. }
            | Self::AlignedObservation(_) => CommissioningExternalSignal::SupervisorFault,
        }
    }
}

impl fmt::Display for CommissioningLiveSourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "commissioning live observation failed: {self:?}")
    }
}

impl std::error::Error for CommissioningLiveSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OakLeft(source) | Self::OakRight(source) | Self::OakRgb(source) => Some(source),
            Self::OakImu(source) => Some(source),
            Self::AccessoryFaultMonitor(source) => Some(source),
            Self::AccessoryTerminalFault(source) => Some(source.as_ref()),
            Self::AccessoryHealth(source) => Some(source),
            Self::Frame(source) => Some(source),
            Self::Pairing(source) => Some(source),
            Self::Tracker(source) => Some(source.as_ref()),
            Self::ImuValue(source) => Some(source),
            Self::ImuCalibration(source) => Some(source),
            Self::Pose(source) => Some(source),
            Self::AlignedObservation(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    #[test]
    fn visual_camera_transform_is_inverted_once_into_positive_base_velocity() {
        // A camera that moved +0.10 m in the previous frame observes the
        // previous frame translated -0.10 m in current-camera coordinates.
        let previous_to_current =
            Pose64::from_rt(IDENTITY, [-0.10, 0.04, 0.0]).expect("proper pose");
        let (forward, lateral) = body_velocity_from_camera_increment(
            previous_to_current,
            Pose64::identity(),
            50_000_000,
        )
        .expect("finite body velocity");
        assert!((forward - 2.0).abs() <= 1.0e-12);
        assert!((lateral + 0.8).abs() <= 1.0e-12);
    }

    #[test]
    fn body_velocity_rejects_zero_duration_instead_of_dividing() {
        assert!(matches!(
            body_velocity_from_camera_increment(Pose64::identity(), Pose64::identity(), 0,),
            Err(CommissioningLiveSourceError::ZeroVisualDuration)
        ));
    }
}
