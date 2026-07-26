//! Typed boundary between one borrowed OAK RGB frame and face association.
//!
//! This module does not own an OAK device, camera pipeline, worker thread, or
//! actuator. [`NanoFacePerception`] owns exactly one separately constructed
//! OpenCV detector and one allocation-free [`FaceTracker`]. A call borrows the
//! original [`ImageFrame`], verifies that its expression observation describes
//! that exact capture, runs the detector, rechecks every provenance field
//! returned by the native boundary, and converts each retained rectangle once
//! into Kiko's domain types.
//!
//! OpenCV's Haar level weight remains opaque ranking metadata. It is never
//! interpreted as probability, confidence, person presence, range, or a
//! permission to move the head.

use core::fmt;

#[cfg(any(feature = "nano-agent", test))]
use kiko_expression_core::{ImageLayout, RgbObservation};
use kiko_expression_runtime::{
    DetectorResultSequence, FaceDetectionBatch, FaceDetectionBatchError, FaceDetectionError,
    FaceTracker, FaceTrackingConfig, FaceTrackingError, FaceTrackingUpdate, MAX_FACE_DETECTIONS,
};
#[cfg(any(feature = "nano-agent", test))]
use kiko_expression_runtime::{FaceDetection, FaceDetectorSource};
use kiko_eye_runtime::ClockError;
#[cfg(any(feature = "nano-agent", test))]
use kiko_eye_runtime::MonotonicClock;
#[cfg(feature = "nano-agent")]
use oak_sys::HaarFaceDetectionBatch;
use oak_sys::{
    CameraTimestampReference, DeviceFrameSequence, FrameDeliverySequence, HaarFaceDetectionError,
    ImageFrame, OpenCvHaarFaceDetector, OpenCvHaarFaceDetectorConfig,
    OpenCvHaarFaceDetectorLoadError, StreamId, Timestamp,
};
#[cfg(any(feature = "nano-agent", test))]
use oak_sys::{HaarFaceDetection, HaarFaceDetectionSource};

#[cfg(feature = "nano-agent")]
use super::expression_bridge::ParsedIngressRgbFrame;

/// OAK identity carried across the synchronous detector boundary.
///
/// The detector is called directly with the borrowed frame, so these fields
/// identify the same allocation on both sides of the call without hashing or
/// copying its pixels. The expression observation has a smaller identity
/// surface and is checked separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OakFaceFrameProvenance {
    stream: StreamId,
    device_capture_sequence: DeviceFrameSequence,
    host_delivery_sequence: FrameDeliverySequence,
    timestamp: Timestamp,
    timestamp_reference: CameraTimestampReference,
    width_px: u32,
    height_px: u32,
}

impl OakFaceFrameProvenance {
    pub fn from_frame(frame: &ImageFrame) -> Self {
        Self {
            stream: frame.stream,
            device_capture_sequence: frame.device_capture_sequence,
            host_delivery_sequence: frame.host_delivery_sequence,
            timestamp: frame.timestamp,
            timestamp_reference: frame.timestamp_reference,
            width_px: frame.width,
            height_px: frame.height,
        }
    }

    #[cfg(feature = "nano-agent")]
    fn from_detector_batch(batch: &HaarFaceDetectionBatch) -> Self {
        Self {
            stream: batch.stream(),
            device_capture_sequence: batch.device_capture_sequence(),
            host_delivery_sequence: batch.host_delivery_sequence(),
            timestamp: batch.timestamp(),
            timestamp_reference: batch.timestamp_reference(),
            width_px: batch.width(),
            height_px: batch.height(),
        }
    }

    pub const fn stream(self) -> StreamId {
        self.stream
    }

    pub const fn device_capture_sequence(self) -> DeviceFrameSequence {
        self.device_capture_sequence
    }

    pub const fn host_delivery_sequence(self) -> FrameDeliverySequence {
        self.host_delivery_sequence
    }

    pub const fn timestamp(self) -> Timestamp {
        self.timestamp
    }

    pub const fn timestamp_reference(self) -> CameraTimestampReference {
        self.timestamp_reference
    }

    pub const fn width_px(self) -> u32 {
        self.width_px
    }

    pub const fn height_px(self) -> u32 {
        self.height_px
    }
}

/// One detector batch and the association decision derived from that same
/// batch. Both values retain the exact expression observation and detector
/// result sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoFacePerceptionOutput {
    batch: FaceDetectionBatch,
    tracking: FaceTrackingUpdate,
}

impl NanoFacePerceptionOutput {
    #[cfg(any(feature = "nano-agent", test))]
    const fn from_parts(batch: FaceDetectionBatch, tracking: FaceTrackingUpdate) -> Self {
        Self { batch, tracking }
    }

    pub const fn batch(self) -> FaceDetectionBatch {
        self.batch
    }

    pub const fn tracking(self) -> FaceTrackingUpdate {
        self.tracking
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NanoFacePerceptionLoadError {
    DetectorRetainedCapacityExceedsTracker { configured: u32, maximum: usize },
    Detector(OpenCvHaarFaceDetectorLoadError),
}

impl fmt::Display for NanoFacePerceptionLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "face perception construction failed: {self:?}")
    }
}

impl std::error::Error for NanoFacePerceptionLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Detector(source) => Some(source),
            Self::DetectorRetainedCapacityExceedsTracker { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NanoFacePerceptionError {
    DetectorResultSequenceExhausted,
    Clock(ClockError),
    Detector(HaarFaceDetectionError),
    DetectorFrameProvenanceMismatch {
        frame: OakFaceFrameProvenance,
        detector: OakFaceFrameProvenance,
    },
    NativeCountLessThanRetained {
        native: usize,
        retained: usize,
    },
    RetainedCountExceedsTrackerCapacity {
        retained: usize,
        maximum: usize,
    },
    TruncatedCountExceedsU32 {
        truncated: usize,
    },
    Detection {
        index: usize,
        source: FaceDetectionError,
    },
    Batch(FaceDetectionBatchError),
    Tracking(FaceTrackingError),
}

impl fmt::Display for NanoFacePerceptionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "face perception rejected a detector result: {self:?}"
        )
    }
}

impl std::error::Error for NanoFacePerceptionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Clock(source) => Some(source),
            Self::Detector(source) => Some(source),
            Self::Detection { source, .. } => Some(source),
            Self::Batch(source) => Some(source),
            Self::Tracking(source) => Some(source),
            Self::DetectorResultSequenceExhausted
            | Self::DetectorFrameProvenanceMismatch { .. }
            | Self::NativeCountLessThanRetained { .. }
            | Self::RetainedCountExceedsTrackerCapacity { .. }
            | Self::TruncatedCountExceedsU32 { .. } => None,
        }
    }
}

/// One detector-result identity stream.
///
/// A number is consumed after native detection succeeds, even if typed
/// conversion or association subsequently rejects that result. Consequently a
/// later accepted result truthfully reports a gap rather than manufacturing
/// consecutive evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DetectorResultSequencer {
    next: Option<u64>,
}

impl DetectorResultSequencer {
    const fn new() -> Self {
        Self { next: Some(1) }
    }

    const fn peek(self) -> Option<DetectorResultSequence> {
        match self.next {
            Some(value) => Some(DetectorResultSequence::new(value)),
            None => None,
        }
    }

    #[cfg(any(feature = "nano-agent", test))]
    fn consume(&mut self) -> Result<DetectorResultSequence, NanoFacePerceptionError> {
        let value = self
            .next
            .ok_or(NanoFacePerceptionError::DetectorResultSequenceExhausted)?;
        self.next = value.checked_add(1);
        Ok(DetectorResultSequence::new(value))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FacePerceptionCore {
    tracker: FaceTracker,
    result_sequence: DetectorResultSequencer,
}

impl FacePerceptionCore {
    const fn new(config: FaceTrackingConfig) -> Self {
        Self {
            tracker: FaceTracker::new(config),
            result_sequence: DetectorResultSequencer::new(),
        }
    }

    #[cfg(feature = "nano-agent")]
    fn ensure_result_sequence_available(&self) -> Result<(), NanoFacePerceptionError> {
        self.result_sequence
            .peek()
            .map(|_| ())
            .ok_or(NanoFacePerceptionError::DetectorResultSequenceExhausted)
    }

    #[cfg(any(feature = "nano-agent", test))]
    fn reserve_result_sequence(
        &mut self,
    ) -> Result<DetectorResultSequence, NanoFacePerceptionError> {
        self.result_sequence.consume()
    }

    #[cfg(any(feature = "nano-agent", test))]
    fn admit_reserved<C: MonotonicClock, D: HaarDetectionFields>(
        &mut self,
        result_sequence: DetectorResultSequence,
        observation: RgbObservation,
        native_detection_count: usize,
        detections: &[D],
        clock: &C,
    ) -> Result<NanoFacePerceptionOutput, NanoFacePerceptionError> {
        let batch = convert_detection_batch(
            observation,
            result_sequence,
            native_detection_count,
            detections,
        )?;
        let now = clock.now().map_err(NanoFacePerceptionError::Clock)?;
        let tracking = self
            .tracker
            .update(&batch, now)
            .map_err(NanoFacePerceptionError::Tracking)?;
        Ok(NanoFacePerceptionOutput::from_parts(batch, tracking))
    }

    #[cfg(test)]
    fn admit<C: MonotonicClock, D: HaarDetectionFields>(
        &mut self,
        observation: RgbObservation,
        native_detection_count: usize,
        detections: &[D],
        clock: &C,
    ) -> Result<NanoFacePerceptionOutput, NanoFacePerceptionError> {
        let result_sequence = self.reserve_result_sequence()?;
        self.admit_reserved(
            result_sequence,
            observation,
            native_detection_count,
            detections,
            clock,
        )
    }
}

/// Stateful detector and face association boundary.
///
/// Construction rejects a native retained-output cap larger than the tracker's
/// fixed capacity. This avoids making an oversized native batch normal and
/// then silently applying a second cap.
///
/// In a native build this owner is deliberately `!Send`: CXX opaque C++ types
/// do not implement `Send`, and Kiko adds no unsafe override. A dedicated
/// perception thread must construct and retain this value on that same thread;
/// it must not construct it elsewhere and move it through `spawn_blocking`.
pub struct NanoFacePerception {
    detector: OpenCvHaarFaceDetector,
    core: FacePerceptionCore,
}

impl NanoFacePerception {
    /// Parse the exact retained deployment-asset bytes into one native
    /// detector and pair it with a fresh tracker.
    ///
    /// The OAK boundary copies each slice into an in-memory OpenCV
    /// `FileStorage` parse during this call; it never reopens a path. The
    /// caller may release its bootstrap byte buffers after construction.
    pub fn load(
        frontal_cascade_xml: &[u8],
        profile_cascade_xml: &[u8],
        detector_config: OpenCvHaarFaceDetectorConfig,
        tracking_config: FaceTrackingConfig,
    ) -> Result<Self, NanoFacePerceptionLoadError> {
        let configured = detector_config.maximum_retained_detections();
        if usize::try_from(configured).map_or(true, |configured| configured > MAX_FACE_DETECTIONS) {
            return Err(
                NanoFacePerceptionLoadError::DetectorRetainedCapacityExceedsTracker {
                    configured,
                    maximum: MAX_FACE_DETECTIONS,
                },
            );
        }
        let detector =
            OpenCvHaarFaceDetector::load(frontal_cascade_xml, profile_cascade_xml, detector_config)
                .map_err(NanoFacePerceptionLoadError::Detector)?;
        Ok(Self {
            detector,
            core: FacePerceptionCore::new(tracking_config),
        })
    }

    pub fn detector_config(&self) -> &OpenCvHaarFaceDetectorConfig {
        self.detector.config()
    }

    pub const fn tracking_config(&self) -> FaceTrackingConfig {
        self.core.tracker.config()
    }

    pub const fn next_detector_result_sequence(&self) -> Option<DetectorResultSequence> {
        self.core.result_sequence.peek()
    }

    /// Forget camera continuity after a confirmed stream restart.
    ///
    /// Detector result identities and previously issued face track IDs are not
    /// reused. The next accepted result is therefore a cold-start association
    /// while remaining globally distinguishable within this state object.
    pub fn reset_camera_stream(&mut self) {
        self.core.tracker.reset_stream();
    }

    /// Detect and associate faces in this exact, already-parsed OAK frame.
    ///
    /// The private constructor of [`ParsedIngressRgbFrame`] makes mismatched
    /// capture/layout/freshness metadata unrepresentable. `clock` must share
    /// the observation freshness origin; it is sampled after detection and
    /// conversion, immediately before tracker admission, so detector latency
    /// counts against freshness. The caller retains this value so it can move
    /// the same frame into the expression bridge after this borrow.
    #[cfg(feature = "nano-agent")]
    pub(super) fn process_parsed<C: MonotonicClock>(
        &mut self,
        parsed: &ParsedIngressRgbFrame,
        clock: &C,
    ) -> Result<NanoFacePerceptionOutput, NanoFacePerceptionError> {
        self.core.ensure_result_sequence_available()?;
        let frame = parsed.frame();
        let observation = parsed.observation();

        let frame_provenance = OakFaceFrameProvenance::from_frame(frame);
        let detector_batch = self
            .detector
            .detect(frame)
            .map_err(NanoFacePerceptionError::Detector)?;

        // Native detection succeeded, so this result owns an identity even if
        // a defensive provenance/conversion check rejects it below.
        let result_sequence = self.core.reserve_result_sequence()?;
        let detector_provenance = OakFaceFrameProvenance::from_detector_batch(&detector_batch);
        require_matching_provenance(frame_provenance, detector_provenance)?;

        self.core.admit_reserved(
            result_sequence,
            observation,
            detector_batch.native_detection_count(),
            detector_batch.detections(),
            clock,
        )
    }
}

#[cfg(any(feature = "nano-agent", test))]
fn require_matching_provenance(
    frame: OakFaceFrameProvenance,
    detector: OakFaceFrameProvenance,
) -> Result<(), NanoFacePerceptionError> {
    if detector == frame {
        Ok(())
    } else {
        Err(NanoFacePerceptionError::DetectorFrameProvenanceMismatch { frame, detector })
    }
}

#[cfg(any(feature = "nano-agent", test))]
trait HaarDetectionFields {
    fn left_px(&self) -> u32;
    fn top_px(&self) -> u32;
    fn width_px(&self) -> u32;
    fn height_px(&self) -> u32;
    fn level_weight(&self) -> f64;
    fn source(&self) -> HaarFaceDetectionSource;
}

#[cfg(any(feature = "nano-agent", test))]
impl HaarDetectionFields for HaarFaceDetection {
    fn left_px(&self) -> u32 {
        self.bounds().x()
    }

    fn top_px(&self) -> u32 {
        self.bounds().y()
    }

    fn width_px(&self) -> u32 {
        self.bounds().width()
    }

    fn height_px(&self) -> u32 {
        self.bounds().height()
    }

    fn level_weight(&self) -> f64 {
        self.detector_level_weight()
    }

    fn source(&self) -> HaarFaceDetectionSource {
        HaarFaceDetection::source(*self)
    }
}

#[cfg(any(feature = "nano-agent", test))]
fn convert_detection_batch<D: HaarDetectionFields>(
    observation: RgbObservation,
    result_sequence: DetectorResultSequence,
    native_detection_count: usize,
    detections: &[D],
) -> Result<FaceDetectionBatch, NanoFacePerceptionError> {
    if detections.len() > MAX_FACE_DETECTIONS {
        return Err(
            NanoFacePerceptionError::RetainedCountExceedsTrackerCapacity {
                retained: detections.len(),
                maximum: MAX_FACE_DETECTIONS,
            },
        );
    }
    let truncated = native_detection_count.checked_sub(detections.len()).ok_or(
        NanoFacePerceptionError::NativeCountLessThanRetained {
            native: native_detection_count,
            retained: detections.len(),
        },
    )?;
    let detector_truncated_count = u32::try_from(truncated)
        .map_err(|_| NanoFacePerceptionError::TruncatedCountExceedsU32 { truncated })?;

    let Some(first) = detections.first() else {
        return FaceDetectionBatch::try_new(
            observation,
            result_sequence,
            detector_truncated_count,
            &[],
        )
        .map_err(NanoFacePerceptionError::Batch);
    };
    let first = convert_detection(observation.layout(), 0, first)?;
    let mut converted = [first; MAX_FACE_DETECTIONS];
    for (index, detection) in detections.iter().enumerate().skip(1) {
        converted[index] = convert_detection(observation.layout(), index, detection)?;
    }
    FaceDetectionBatch::try_new(
        observation,
        result_sequence,
        detector_truncated_count,
        &converted[..detections.len()],
    )
    .map_err(NanoFacePerceptionError::Batch)
}

#[cfg(any(feature = "nano-agent", test))]
fn convert_detection<D: HaarDetectionFields>(
    layout: ImageLayout,
    index: usize,
    detection: &D,
) -> Result<FaceDetection, NanoFacePerceptionError> {
    FaceDetection::try_new(
        layout,
        detection.left_px(),
        detection.top_px(),
        detection.width_px(),
        detection.height_px(),
        detection.level_weight(),
        map_source(detection.source()),
    )
    .map_err(|source| NanoFacePerceptionError::Detection { index, source })
}

#[cfg(any(feature = "nano-agent", test))]
const fn map_source(source: HaarFaceDetectionSource) -> FaceDetectorSource {
    match source {
        HaarFaceDetectionSource::Frontal => FaceDetectorSource::Frontal,
        HaarFaceDetectionSource::Profile => FaceDetectorSource::Profile,
        HaarFaceDetectionSource::MirroredProfile => FaceDetectorSource::MirroredProfile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU64, Ordering};
    use kiko_expression_core::{
        ChannelOrder, FrameId, FreshnessWindow, MonotonicTimestamp, NonZeroDuration, StreamEpochId,
    };
    use kiko_expression_runtime::{FaceResultAdmission, FaceTargetState};

    #[derive(Clone, Copy)]
    struct TestDetection {
        left_px: u32,
        top_px: u32,
        width_px: u32,
        height_px: u32,
        level_weight: f64,
        source: HaarFaceDetectionSource,
    }

    impl HaarDetectionFields for TestDetection {
        fn left_px(&self) -> u32 {
            self.left_px
        }

        fn top_px(&self) -> u32 {
            self.top_px
        }

        fn width_px(&self) -> u32 {
            self.width_px
        }

        fn height_px(&self) -> u32 {
            self.height_px
        }

        fn level_weight(&self) -> f64 {
            self.level_weight
        }

        fn source(&self) -> HaarFaceDetectionSource {
            self.source
        }
    }

    fn at(nanos: u64) -> MonotonicTimestamp {
        MonotonicTimestamp::from_nanos_since_epoch(nanos)
    }

    struct TestClock(AtomicU64);

    impl TestClock {
        fn new(now_ns: u64) -> Self {
            Self(AtomicU64::new(now_ns))
        }

        fn set(&self, now_ns: u64) {
            self.0.store(now_ns, Ordering::Relaxed);
        }
    }

    impl MonotonicClock for TestClock {
        fn now(&self) -> Result<MonotonicTimestamp, ClockError> {
            Ok(at(self.0.load(Ordering::Relaxed)))
        }
    }

    struct FailingClock;

    impl MonotonicClock for FailingClock {
        fn now(&self) -> Result<MonotonicTimestamp, ClockError> {
            Err(ClockError::SourceUnavailable {
                message: "test clock unavailable".into(),
            })
        }
    }

    struct DetectionThatAdvancesClock<'a> {
        detection: TestDetection,
        clock: &'a TestClock,
        completion_ns: u64,
    }

    impl HaarDetectionFields for DetectionThatAdvancesClock<'_> {
        fn left_px(&self) -> u32 {
            self.detection.left_px
        }

        fn top_px(&self) -> u32 {
            self.detection.top_px
        }

        fn width_px(&self) -> u32 {
            self.detection.width_px
        }

        fn height_px(&self) -> u32 {
            self.detection.height_px
        }

        fn level_weight(&self) -> f64 {
            self.detection.level_weight
        }

        fn source(&self) -> HaarFaceDetectionSource {
            self.clock.set(self.completion_ns);
            self.detection.source
        }
    }

    fn layout(width: u32, height: u32) -> ImageLayout {
        ImageLayout::try_new(
            width,
            height,
            width.checked_mul(3).unwrap(),
            ChannelOrder::Bgr,
        )
        .unwrap()
    }

    fn observation(sequence: u64, observed_at_ns: u64) -> RgbObservation {
        let freshness = FreshnessWindow::from_ttl(
            at(observed_at_ns),
            NonZeroDuration::try_from_nanos(1_000_000_000).unwrap(),
        )
        .unwrap();
        RgbObservation::new(
            FrameId::new(StreamEpochId::try_new(7).unwrap(), sequence),
            layout(640, 400),
            freshness,
        )
    }

    fn face(source: HaarFaceDetectionSource) -> TestDetection {
        TestDetection {
            left_px: 100,
            top_px: 80,
            width_px: 60,
            height_px: 70,
            level_weight: -2.5,
            source,
        }
    }

    #[test]
    fn detector_provenance_compares_every_carried_identity_field() {
        let baseline = OakFaceFrameProvenance {
            stream: StreamId::Rgb,
            device_capture_sequence: DeviceFrameSequence::try_from_i64(4).unwrap(),
            host_delivery_sequence: FrameDeliverySequence::new(9),
            timestamp: Timestamp::try_from_nanos(50).unwrap(),
            timestamp_reference: CameraTimestampReference::ExposureMidpoint,
            width_px: 640,
            height_px: 400,
        };
        let changed_stream = OakFaceFrameProvenance {
            stream: StreamId::MonoLeft,
            ..baseline
        };
        let changed_capture = OakFaceFrameProvenance {
            device_capture_sequence: DeviceFrameSequence::try_from_i64(5).unwrap(),
            ..baseline
        };
        let changed_delivery = OakFaceFrameProvenance {
            host_delivery_sequence: FrameDeliverySequence::new(10),
            ..baseline
        };
        let changed_time = OakFaceFrameProvenance {
            timestamp: Timestamp::try_from_nanos(51).unwrap(),
            ..baseline
        };
        let changed_dimensions = OakFaceFrameProvenance {
            width_px: 639,
            height_px: 399,
            ..baseline
        };
        for changed in [
            changed_stream,
            changed_capture,
            changed_delivery,
            changed_time,
            changed_dimensions,
        ] {
            assert_eq!(
                require_matching_provenance(baseline, changed),
                Err(NanoFacePerceptionError::DetectorFrameProvenanceMismatch {
                    frame: baseline,
                    detector: changed,
                })
            );
        }
        assert_eq!(require_matching_provenance(baseline, baseline), Ok(()));
    }

    #[test]
    fn construction_rejects_an_oversized_native_cap_before_loading_assets() {
        let config = OpenCvHaarFaceDetectorConfig::try_new(
            1.15,
            6,
            4,
            30,
            30,
            u32::try_from(MAX_FACE_DETECTIONS + 1).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            NanoFacePerception::load(b"", b"", config, FaceTrackingConfig::default()),
            Err(
                NanoFacePerceptionLoadError::DetectorRetainedCapacityExceedsTracker {
                    configured,
                    maximum: MAX_FACE_DETECTIONS,
                }
            ) if configured == u32::try_from(MAX_FACE_DETECTIONS + 1).unwrap()
        ));
    }

    #[test]
    fn construction_passes_exact_asset_bytes_to_the_detector_boundary() {
        let config = OpenCvHaarFaceDetectorConfig::try_new(
            1.15,
            6,
            4,
            30,
            30,
            u32::try_from(MAX_FACE_DETECTIONS).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            NanoFacePerception::load(b"", b"<profile/>", config, FaceTrackingConfig::default()),
            Err(NanoFacePerceptionLoadError::Detector(
                OpenCvHaarFaceDetectorLoadError::EmptyCascadeXml { cascade: "frontal" }
            ))
        ));
    }

    #[test]
    fn all_detector_sources_map_without_interpreting_level_weight() {
        for (input, expected) in [
            (
                HaarFaceDetectionSource::Frontal,
                FaceDetectorSource::Frontal,
            ),
            (
                HaarFaceDetectionSource::Profile,
                FaceDetectorSource::Profile,
            ),
            (
                HaarFaceDetectionSource::MirroredProfile,
                FaceDetectorSource::MirroredProfile,
            ),
        ] {
            let input = face(input);
            let batch = convert_detection_batch(
                observation(1, 100),
                DetectorResultSequence::new(1),
                1,
                &[input],
            )
            .unwrap();
            let converted = batch.get(0).unwrap();
            assert_eq!(converted.source(), expected);
            assert_eq!(converted.detector_level_weight().as_f64(), -2.5);
        }
    }

    #[test]
    fn count_conversion_is_checked_and_never_applies_a_second_cap() {
        let detections = [face(HaarFaceDetectionSource::Frontal); MAX_FACE_DETECTIONS + 1];
        assert_eq!(
            convert_detection_batch(
                observation(1, 100),
                DetectorResultSequence::new(1),
                detections.len(),
                &detections,
            ),
            Err(
                NanoFacePerceptionError::RetainedCountExceedsTrackerCapacity {
                    retained: MAX_FACE_DETECTIONS + 1,
                    maximum: MAX_FACE_DETECTIONS,
                }
            )
        );

        let two = [face(HaarFaceDetectionSource::Frontal); 2];
        assert_eq!(
            convert_detection_batch(observation(1, 100), DetectorResultSequence::new(1), 1, &two,),
            Err(NanoFacePerceptionError::NativeCountLessThanRetained {
                native: 1,
                retained: 2,
            })
        );
    }

    #[test]
    fn truncation_and_rectangle_bounds_remain_explicit() {
        let batch = convert_detection_batch(
            observation(1, 100),
            DetectorResultSequence::new(3),
            4,
            &[face(HaarFaceDetectionSource::Profile)],
        )
        .unwrap();
        assert_eq!(batch.detector_truncated_count(), 3);

        let outside = TestDetection {
            left_px: 630,
            width_px: 20,
            ..face(HaarFaceDetectionSource::Frontal)
        };
        assert_eq!(
            convert_detection_batch(
                observation(2, 200),
                DetectorResultSequence::new(4),
                1,
                &[outside],
            ),
            Err(NanoFacePerceptionError::Detection {
                index: 0,
                source: FaceDetectionError::RectangleOutsideFrame {
                    frame_width_px: 640,
                    frame_height_px: 400,
                },
            })
        );
    }

    #[test]
    fn core_assigns_monotonic_result_sequences_and_updates_tracker() {
        let mut core = FacePerceptionCore::new(FaceTrackingConfig::default());
        let clock = TestClock::new(110);
        let first = core
            .admit(
                observation(10, 100),
                1,
                &[face(HaarFaceDetectionSource::Frontal)],
                &clock,
            )
            .unwrap();
        assert_eq!(
            first.batch().detector_result_sequence(),
            DetectorResultSequence::new(1)
        );
        assert!(matches!(
            first.tracking().state(),
            FaceTargetState::Acquiring(_)
        ));
        assert_eq!(first.tracking().admission(), FaceResultAdmission::ColdStart);

        clock.set(210);
        let second = core
            .admit(
                observation(11, 200),
                1,
                &[face(HaarFaceDetectionSource::Frontal)],
                &clock,
            )
            .unwrap();
        assert_eq!(
            second.batch().detector_result_sequence(),
            DetectorResultSequence::new(2)
        );
        assert!(matches!(
            second.tracking().state(),
            FaceTargetState::Tracked(_)
        ));
        assert_eq!(
            second.tracking().admission(),
            FaceResultAdmission::Consecutive {
                previous: DetectorResultSequence::new(1),
                actual: DetectorResultSequence::new(2),
            }
        );
    }

    #[test]
    fn rejected_typed_result_consumes_identity_and_creates_truthful_gap() {
        let mut core = FacePerceptionCore::new(FaceTrackingConfig::default());
        let clock = TestClock::new(110);
        core.admit(
            observation(10, 100),
            1,
            &[face(HaarFaceDetectionSource::Frontal)],
            &clock,
        )
        .unwrap();

        let outside = TestDetection {
            left_px: 630,
            width_px: 20,
            ..face(HaarFaceDetectionSource::Frontal)
        };
        clock.set(210);
        assert!(matches!(
            core.admit(observation(11, 200), 1, &[outside], &clock),
            Err(NanoFacePerceptionError::Detection { .. })
        ));
        clock.set(310);
        let third = core
            .admit(
                observation(12, 300),
                1,
                &[face(HaarFaceDetectionSource::Frontal)],
                &clock,
            )
            .unwrap();
        assert_eq!(
            third.batch().detector_result_sequence(),
            DetectorResultSequence::new(3)
        );
        let FaceResultAdmission::ForwardGap {
            previous, actual, ..
        } = third.tracking().admission()
        else {
            panic!("third accepted result must report the rejected result as a gap");
        };
        assert_eq!(previous, DetectorResultSequence::new(1));
        assert_eq!(actual, DetectorResultSequence::new(3));
    }

    #[test]
    fn tracker_samples_completion_time_after_detector_conversion() {
        let mut core = FacePerceptionCore::new(FaceTrackingConfig::default());
        let clock = TestClock::new(110);
        let detection = DetectionThatAdvancesClock {
            detection: face(HaarFaceDetectionSource::Frontal),
            clock: &clock,
            completion_ns: 2_000_000_000,
        };

        assert_eq!(
            core.admit(observation(1, 100), 1, &[detection], &clock),
            Err(NanoFacePerceptionError::Tracking(
                FaceTrackingError::StaleFrame {
                    deadline_ns: 1_000_000_100,
                    now_ns: 2_000_000_000,
                }
            ))
        );
    }

    #[test]
    fn clock_failure_is_typed_and_does_not_reuse_result_identity() {
        let mut core = FacePerceptionCore::new(FaceTrackingConfig::default());
        assert!(matches!(
            core.admit(
                observation(1, 100),
                1,
                &[face(HaarFaceDetectionSource::Frontal)],
                &FailingClock,
            ),
            Err(NanoFacePerceptionError::Clock(
                ClockError::SourceUnavailable { .. }
            ))
        ));

        let clock = TestClock::new(210);
        let accepted = core
            .admit(
                observation(2, 200),
                1,
                &[face(HaarFaceDetectionSource::Frontal)],
                &clock,
            )
            .unwrap();
        assert_eq!(
            accepted.batch().detector_result_sequence(),
            DetectorResultSequence::new(2)
        );
    }

    #[test]
    fn detector_result_sequence_exhaustion_is_explicit() {
        let mut sequence = DetectorResultSequencer {
            next: Some(u64::MAX),
        };
        assert_eq!(
            sequence.consume().unwrap(),
            DetectorResultSequence::new(u64::MAX)
        );
        assert_eq!(
            sequence.consume(),
            Err(NanoFacePerceptionError::DetectorResultSequenceExhausted)
        );
    }
}
