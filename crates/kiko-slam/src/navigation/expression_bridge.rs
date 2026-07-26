//! Borrowed OAK RGB frames to bounded KEP2 eye intentions.
//!
//! This bridge deliberately owns no camera, protocol session, transport, eye
//! actor, or head actor. One call borrows the tightly packed BGR888 bytes from
//! an already parsed [`oak_sys::ImageFrame`], derives fixed-size scene-motion
//! state, and returns only a [`PreparedEyeIntent`] for the eye actor's bounded
//! mailbox. Source frame storage is neither copied nor retained.
//!
//! OAK device timestamps and host timestamps have unrelated epochs. The bridge
//! therefore never guesses an offset between them: the device timestamp is
//! checked only for capture-clock continuity, while clones of one injected
//! [`MonotonicClock`] supply ingress observation, processing, reaction, and
//! intent time. The RGB ingress and eye actor must receive clones or shared
//! adapters with that exact same clock origin.

use std::fmt;

use kiko_expression_core::{
    ChannelOrder, ExpressionIntent, ExpressionKind, ExpressionPriority, FrameId, FreshnessWindow,
    GazeTarget, HeadMotionPolicy, ImageLayout, ImageLayoutError, MonotonicTimestamp,
    NonZeroDuration, PositiveUnitAmount, ReactionInputs, ReactionMixer, RgbFrameView,
    RgbObservation, StreamEpochId, TimeError,
};
use kiko_expression_runtime::{
    AdaptError, CameraForwardDepthMeters, CameraToHeadGazeExtrinsics, EyeRenderStyle,
    FaceTargetState, FaceTrackingUpdate, HeadGazeProjectionError, HeadRelativeGaze,
    MonotonicLatestAdmission, MonotonicLatestGap, OakCameraTargetPoint, OakCameraTargetRay,
    PreparedEyeIntent, RayHeadGazeProjectionError, SceneAnalysis, SceneMotionConfig,
    SceneMotionError, SceneMotionExtractor, adapt_reaction_output,
};
use kiko_eye_runtime::{ClockError, MonotonicClock};
use oak_sys::{ImageFrame, StreamId};

use super::NanoRgbExpressionConfig;

/// The RGB expression path can never request expressive head displacement.
pub const RGB_EXPRESSION_HEAD_POLICY: HeadMotionPolicy = HeadMotionPolicy::NaturalHold;

/// Semantic attention strength selected by policy for one associated face.
///
/// This is deliberately independent of OpenCV's arbitrary Haar level weight.
/// It means “an established face target owns the Important gaze lane”; it is
/// not a detector probability or a person-confidence claim.
pub const FACE_ATTENTION_STRENGTH: PositiveUnitAmount = PositiveUnitAmount::ONE;

/// Why the RGB domain seam cannot produce typed head-relative gaze geometry.
///
/// This output is geometry only. It is not a [`HeadMotionPolicy`], head actor
/// command, servo pose, or permission to move the physical head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RgbHeadGazeProjectionError {
    GeometryUnavailable,
    Point(HeadGazeProjectionError),
    Ray(RayHeadGazeProjectionError),
}

impl fmt::Display for RgbHeadGazeProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "RGB expression head-gaze projection failed: {self:?}"
        )
    }
}

impl std::error::Error for RgbHeadGazeProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::GeometryUnavailable => None,
            Self::Point(source) => Some(source),
            Self::Ray(source) => Some(source),
        }
    }
}

/// A successful expression decision and its continuity semantics.
///
/// Each variant contains exactly one small, transport-independent eye intent.
/// A forward gap remains a real comparison against the last accepted frame,
/// while its exact skipped-sequence count remains available to diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RgbExpressionBridgeOutcome {
    ColdStart(PreparedEyeIntent),
    Consecutive(PreparedEyeIntent),
    ForwardGap {
        gap: MonotonicLatestGap,
        prepared: PreparedEyeIntent,
    },
}

impl RgbExpressionBridgeOutcome {
    /// Discard bridge diagnostics and yield the only value accepted by the eye
    /// actor's bounded intent mailbox.
    pub const fn into_prepared(self) -> PreparedEyeIntent {
        match self {
            Self::ColdStart(prepared)
            | Self::Consecutive(prepared)
            | Self::ForwardGap { prepared, .. } => prepared,
        }
    }
}

/// A rejected RGB-to-expression boundary operation.
///
/// Frame, clock, continuity, and scene-analysis failures leave the previous
/// accepted frame in place. `Adapt` can occur only after scene analysis has
/// accepted the frame, so that rare downstream failure advances comparison
/// state and must not be retried as though the frame were unseen. A caller
/// must start a new bridge with a new [`StreamEpochId`] after a camera
/// reconnect; a sequence regression is not guessed to be a restart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RgbExpressionBridgeError {
    Clock(ClockError),
    NotRgbStream {
        actual: StreamId,
    },
    NonTightBgrLayout {
        width_px: u32,
        expected_stride_bytes: u32,
        actual_stride_bytes: u32,
    },
    Layout(ImageLayoutError),
    Freshness(TimeError),
    DeviceCaptureClockNotIncreasing {
        previous_ns: i64,
        actual_ns: i64,
    },
    SceneMotion(SceneMotionError),
    FaceObservationMismatch {
        frame: RgbObservation,
        face: RgbObservation,
    },
    FaceAttentionFreshness(TimeError),
    Adapt(AdaptError),
}

impl fmt::Display for RgbExpressionBridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "RGB expression bridge rejected a frame: {self:?}"
        )
    }
}

impl std::error::Error for RgbExpressionBridgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Clock(source) => Some(source),
            Self::Layout(source) => Some(source),
            Self::Freshness(source) => Some(source),
            Self::SceneMotion(source) => Some(source),
            Self::FaceAttentionFreshness(source) => Some(source),
            Self::Adapt(source) => Some(source),
            Self::NotRgbStream { .. }
            | Self::NonTightBgrLayout { .. }
            | Self::DeviceCaptureClockNotIncreasing { .. }
            | Self::FaceObservationMismatch { .. } => None,
        }
    }
}

#[derive(Clone, Copy)]
struct BorrowedOakFrame<'a> {
    stream: StreamId,
    capture_sequence: u64,
    device_timestamp_ns: i64,
    width_px: u32,
    height_px: u32,
    stride_bytes: u32,
    pixels: &'a [u8],
}

impl<'a> From<&'a ImageFrame> for BorrowedOakFrame<'a> {
    fn from(frame: &'a ImageFrame) -> Self {
        Self {
            stream: frame.stream,
            capture_sequence: frame.device_capture_sequence.as_u64(),
            device_timestamp_ns: frame.timestamp.as_nanos(),
            width_px: frame.width,
            height_px: frame.height,
            stride_bytes: frame.stride_bytes,
            pixels: frame.pixels(),
        }
    }
}

/// One owned RGB frame paired with its host-clock observation at queue ingress.
///
/// `F` is moved into this wrapper. In production it remains the original
/// [`ImageFrame`] allocation; no frame or pixel buffer is cloned.
pub(super) struct IngressObservedRgbFrame<F> {
    frame: F,
    observed_at: MonotonicTimestamp,
}

impl<F> IngressObservedRgbFrame<F> {
    pub(super) const fn new(frame: F, observed_at: MonotonicTimestamp) -> Self {
        Self { frame, observed_at }
    }

    #[cfg(test)]
    pub(super) const fn observed_at(&self) -> MonotonicTimestamp {
        self.observed_at
    }

    fn into_parts(self) -> (F, MonotonicTimestamp) {
        (self.frame, self.observed_at)
    }
}

/// One OAK RGB frame after the weak transport metadata has been parsed into
/// the expression domain exactly once.
///
/// Construction is private to [`parse_ingress_observed_oak_frame`], so the
/// frame allocation and its [`RgbObservation`] cannot disagree. The original
/// [`ImageFrame`] remains owned and is moved through perception without a
/// pixel copy.
pub(super) struct ParsedIngressRgbFrame {
    frame: ImageFrame,
    observation: RgbObservation,
}

impl ParsedIngressRgbFrame {
    #[cfg(any(feature = "nano-agent", all(test, feature = "nano-face-perception")))]
    pub(super) const fn frame(&self) -> &ImageFrame {
        &self.frame
    }

    #[cfg(any(feature = "nano-agent", all(test, feature = "nano-face-perception")))]
    pub(super) const fn observation(&self) -> RgbObservation {
        self.observation
    }

    fn into_parts(self) -> (ImageFrame, RgbObservation) {
        (self.frame, self.observation)
    }
}

/// Parse an ingress-stamped OAK frame into one provenance-bearing domain
/// value. Detection, tracking, and expression processing must carry this
/// value rather than reconstructing an observation from loose fields.
pub(super) fn parse_ingress_observed_oak_frame(
    frame: IngressObservedRgbFrame<ImageFrame>,
    stream_epoch: StreamEpochId,
    freshness_ttl: NonZeroDuration,
) -> Result<ParsedIngressRgbFrame, RgbExpressionBridgeError> {
    let (frame, observed_at) = frame.into_parts();
    let borrowed = BorrowedOakFrame::from(&frame);
    let layout = parse_tight_bgr_layout(borrowed)?;
    let freshness = FreshnessWindow::from_ttl(observed_at, freshness_ttl)
        .map_err(RgbExpressionBridgeError::Freshness)?;
    let observation = RgbObservation::new(
        FrameId::new(stream_epoch, borrowed.capture_sequence),
        layout,
        freshness,
    );
    Ok(ParsedIngressRgbFrame { frame, observation })
}

/// Stateful, allocation-free RGB reaction boundary for one OAK stream epoch.
pub struct RgbExpressionBridge<C> {
    clock: C,
    stream_epoch: StreamEpochId,
    freshness: NonZeroDuration,
    style: EyeRenderStyle,
    gaze_geometry: Option<CameraToHeadGazeExtrinsics>,
    extractor: SceneMotionExtractor,
    mixer: ReactionMixer,
    last_device_timestamp_ns: Option<i64>,
}

impl<C: MonotonicClock> RgbExpressionBridge<C> {
    /// Construct one bridge for one uninterrupted OAK connection.
    ///
    /// `clock` must share its origin with the clock moved into the eye actor.
    /// [`Self::clone_clock_for_eye_actor`] makes that handoff explicit for
    /// clock adapters whose `Clone` implementation preserves the origin, such
    /// as [`kiko_eye_runtime::TokioClock`].
    pub fn new(stream_epoch: StreamEpochId, config: NanoRgbExpressionConfig, clock: C) -> Self {
        Self::from_parts(
            stream_epoch,
            config.scene_motion(),
            config.frame_freshness(),
            config.render_style(),
            config.gaze_geometry(),
            clock,
        )
    }

    fn from_parts(
        stream_epoch: StreamEpochId,
        scene_motion: SceneMotionConfig,
        freshness: NonZeroDuration,
        style: EyeRenderStyle,
        gaze_geometry: Option<CameraToHeadGazeExtrinsics>,
        clock: C,
    ) -> Self {
        Self {
            clock,
            stream_epoch,
            freshness,
            style,
            gaze_geometry,
            extractor: SceneMotionExtractor::new(scene_motion),
            mixer: ReactionMixer::new(RGB_EXPRESSION_HEAD_POLICY),
            last_device_timestamp_ns: None,
        }
    }

    /// Clone the exact configured origin adapter for the sole eye actor.
    ///
    /// The clock type's `Clone` implementation must preserve the origin. This
    /// method cannot make two independently constructed clocks compatible.
    pub fn clone_clock_for_eye_actor(&self) -> C
    where
        C: Clone,
    {
        self.clock.clone()
    }

    /// Project one already-parsed positive-depth OAK point into neutral-head
    /// yaw-right/pitch-down radians. No head intention or actuator command is
    /// constructed by this seam.
    pub fn project_head_gaze_to_point(
        &self,
        target: OakCameraTargetPoint,
    ) -> Result<HeadRelativeGaze, RgbHeadGazeProjectionError> {
        self.gaze_geometry
            .ok_or(RgbHeadGazeProjectionError::GeometryUnavailable)?
            .project_point(target)
            .map_err(RgbHeadGazeProjectionError::Point)
    }

    /// Project a parsed OAK ray at an explicit positive camera-forward depth.
    /// Range is mandatory because the configured camera/head origin offset
    /// makes a direction alone geometrically insufficient.
    pub fn project_head_gaze_along_ray(
        &self,
        ray: OakCameraTargetRay,
        depth: CameraForwardDepthMeters,
    ) -> Result<HeadRelativeGaze, RgbHeadGazeProjectionError> {
        self.gaze_geometry
            .ok_or(RgbHeadGazeProjectionError::GeometryUnavailable)?
            .project_ray_at_forward_depth(ray, depth)
            .map_err(RgbHeadGazeProjectionError::Ray)
    }

    /// Borrow one synchronously delivered OAK RGB frame and prepare one
    /// bounded eye intent.
    ///
    /// This direct-capture entrypoint uses its single clock sample as both
    /// ingress observation and processing time. Queued production delivery
    /// must use [`Self::process_queued_oak_frame`] so queue residence contributes
    /// to freshness.
    pub fn process_oak_frame(
        &mut self,
        frame: &ImageFrame,
    ) -> Result<RgbExpressionBridgeOutcome, RgbExpressionBridgeError> {
        self.process_borrowed(frame.into())
    }

    /// Process a moved frame whose host observation was captured before queue
    /// admission by a clone of this bridge's exact clock origin.
    pub(super) fn process_queued_oak_frame(
        &mut self,
        frame: IngressObservedRgbFrame<ImageFrame>,
    ) -> Result<RgbExpressionBridgeOutcome, RgbExpressionBridgeError> {
        let parsed = parse_ingress_observed_oak_frame(frame, self.stream_epoch, self.freshness)?;
        self.process_parsed_queued_oak_frame(parsed, None)
    }

    /// Mix one face-association result produced from this exact queued frame.
    ///
    /// The equality check prevents a valid face result from being retagged
    /// onto a different camera frame, layout, or freshness window. Haar level
    /// weights never enter the mixer.
    #[cfg(feature = "nano-agent")]
    pub(super) fn process_queued_oak_frame_with_face(
        &mut self,
        frame: ParsedIngressRgbFrame,
        face: FaceTrackingUpdate,
    ) -> Result<RgbExpressionBridgeOutcome, RgbExpressionBridgeError> {
        self.process_parsed_queued_oak_frame(frame, Some(face))
    }

    fn process_parsed_queued_oak_frame(
        &mut self,
        frame: ParsedIngressRgbFrame,
        face: Option<FaceTrackingUpdate>,
    ) -> Result<RgbExpressionBridgeOutcome, RgbExpressionBridgeError> {
        let now = self.clock.now().map_err(RgbExpressionBridgeError::Clock)?;
        let (frame, observation) = frame.into_parts();
        self.process_parsed_borrowed_at((&frame).into(), observation, now, face)
    }

    #[cfg(test)]
    fn process_ingress_observed_borrowed(
        &mut self,
        frame: IngressObservedRgbFrame<BorrowedOakFrame<'_>>,
    ) -> Result<RgbExpressionBridgeOutcome, RgbExpressionBridgeError> {
        self.process_ingress_observed_borrowed_with_face(frame, None)
    }

    #[cfg(test)]
    fn process_ingress_observed_borrowed_with_face(
        &mut self,
        frame: IngressObservedRgbFrame<BorrowedOakFrame<'_>>,
        face: Option<FaceTrackingUpdate>,
    ) -> Result<RgbExpressionBridgeOutcome, RgbExpressionBridgeError> {
        let (frame, observed_at) = frame.into_parts();
        let now = self.clock.now().map_err(RgbExpressionBridgeError::Clock)?;
        self.process_borrowed_at(frame, observed_at, now, face)
    }

    fn process_borrowed(
        &mut self,
        frame: BorrowedOakFrame<'_>,
    ) -> Result<RgbExpressionBridgeOutcome, RgbExpressionBridgeError> {
        let now = self.clock.now().map_err(RgbExpressionBridgeError::Clock)?;
        self.process_borrowed_at(frame, now, now, None)
    }

    fn process_borrowed_at(
        &mut self,
        frame: BorrowedOakFrame<'_>,
        observed_at: MonotonicTimestamp,
        now: MonotonicTimestamp,
        face: Option<FaceTrackingUpdate>,
    ) -> Result<RgbExpressionBridgeOutcome, RgbExpressionBridgeError> {
        let layout = parse_tight_bgr_layout(frame)?;
        let freshness = FreshnessWindow::from_ttl(observed_at, self.freshness)
            .map_err(RgbExpressionBridgeError::Freshness)?;
        let observation = RgbObservation::new(
            FrameId::new(self.stream_epoch, frame.capture_sequence),
            layout,
            freshness,
        );
        self.process_parsed_borrowed_at(frame, observation, now, face)
    }

    fn process_parsed_borrowed_at(
        &mut self,
        frame: BorrowedOakFrame<'_>,
        observation: RgbObservation,
        now: MonotonicTimestamp,
        face: Option<FaceTrackingUpdate>,
    ) -> Result<RgbExpressionBridgeOutcome, RgbExpressionBridgeError> {
        let view = RgbFrameView::try_new(observation, frame.pixels)
            .map_err(RgbExpressionBridgeError::Layout)?;

        self.validate_device_capture_clock(frame.device_timestamp_ns)?;

        if let Some(face) = face
            && face.observation() != observation
        {
            return Err(RgbExpressionBridgeError::FaceObservationMismatch {
                frame: observation,
                face: face.observation(),
            });
        }

        let admitted = self
            .extractor
            .analyze_monotonic_latest(view, now)
            .map_err(RgbExpressionBridgeError::SceneMotion)?;
        let analysis = admitted.analysis();
        self.last_device_timestamp_ns = Some(frame.device_timestamp_ns);

        let scene = analysis.observation();
        let face_intent = face
            .map(face_attention_intent)
            .transpose()
            .map_err(RgbExpressionBridgeError::FaceAttentionFreshness)?
            .flatten();
        let intents = face_intent.as_slice();
        let reaction = self.mixer.mix(
            now,
            ReactionInputs {
                rgb: Some(&observation),
                people: &[],
                scene: Some(&scene),
                intents,
            },
        );
        let prepared = adapt_reaction_output(reaction, ExpressionKind::Curious, self.style, now)
            .map_err(RgbExpressionBridgeError::Adapt)?;

        Ok(match admitted.admission() {
            MonotonicLatestAdmission::ColdStart => {
                debug_assert!(matches!(analysis, SceneAnalysis::ColdStart(_)));
                RgbExpressionBridgeOutcome::ColdStart(prepared)
            }
            MonotonicLatestAdmission::Consecutive { .. } => {
                debug_assert!(!matches!(analysis, SceneAnalysis::ColdStart(_)));
                RgbExpressionBridgeOutcome::Consecutive(prepared)
            }
            MonotonicLatestAdmission::ForwardGap(gap) => {
                debug_assert!(!matches!(analysis, SceneAnalysis::ColdStart(_)));
                RgbExpressionBridgeOutcome::ForwardGap { gap, prepared }
            }
        })
    }

    /// OAK capture time has a device-local epoch and is therefore checked only
    /// for strict monotonicity. Sequence, layout, freshness, stream epoch, and
    /// host-clock admission each remain owned exactly once by the extractor.
    fn validate_device_capture_clock(
        &self,
        device_timestamp_ns: i64,
    ) -> Result<(), RgbExpressionBridgeError> {
        let Some(previous_ns) = self.last_device_timestamp_ns else {
            return Ok(());
        };
        if device_timestamp_ns <= previous_ns {
            return Err(RgbExpressionBridgeError::DeviceCaptureClockNotIncreasing {
                previous_ns,
                actual_ns: device_timestamp_ns,
            });
        }
        Ok(())
    }
}

fn face_attention_intent(
    update: FaceTrackingUpdate,
) -> Result<Option<ExpressionIntent>, TimeError> {
    let (target, freshness) = match update.state() {
        FaceTargetState::Tracked(observation) => (observation, observation.freshness()),
        FaceTargetState::Switched(switched) => {
            let observation = switched.observation();
            (observation, observation.freshness())
        }
        FaceTargetState::Coasting(coasting) => {
            let observation = coasting.last_observation();
            let current = update.observation().freshness();
            let current_deadline = current.valid_until_exclusive();
            let loss_deadline = coasting.loss_deadline();
            let deadline = if current_deadline <= loss_deadline {
                current_deadline
            } else {
                loss_deadline
            };
            let freshness = FreshnessWindow::try_new(current.observed_at(), deadline)?;
            (observation, freshness)
        }
        FaceTargetState::NoTarget | FaceTargetState::Acquiring(_) | FaceTargetState::Lost(_) => {
            return Ok(None);
        }
    };
    Ok(Some(ExpressionIntent::new(
        ExpressionKind::Attentive,
        FACE_ATTENTION_STRENGTH,
        ExpressionPriority::Important,
        Some(GazeTarget::new(target.center())),
        freshness,
    )))
}

fn parse_tight_bgr_layout(
    frame: BorrowedOakFrame<'_>,
) -> Result<ImageLayout, RgbExpressionBridgeError> {
    if frame.stream != StreamId::Rgb {
        return Err(RgbExpressionBridgeError::NotRgbStream {
            actual: frame.stream,
        });
    }
    let layout = ImageLayout::try_new(
        frame.width_px,
        frame.height_px,
        frame.stride_bytes,
        ChannelOrder::Bgr,
    )
    .map_err(RgbExpressionBridgeError::Layout)?;
    let expected_stride_bytes =
        frame
            .width_px
            .checked_mul(3)
            .ok_or(RgbExpressionBridgeError::Layout(
                ImageLayoutError::LayoutSizeOverflow,
            ))?;
    if frame.stride_bytes != expected_stride_bytes {
        return Err(RgbExpressionBridgeError::NonTightBgrLayout {
            width_px: frame.width_px,
            expected_stride_bytes,
            actual_stride_bytes: frame.stride_bytes,
        });
    }
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    };

    use kiko_expression_core::{MonotonicTimestamp, PositiveUnitAmount, UnitAmount};
    use kiko_expression_runtime::{
        DetectorResultSequence, FaceDetection, FaceDetectionBatch, FaceDetectorSource, FaceTracker,
        FaceTrackingConfig, MotionThresholds, SamplingGeometry,
    };
    use kiko_eye_protocol::Expression;

    use super::*;

    struct TestClockState {
        now_ns: AtomicU64,
        failing: AtomicBool,
    }

    #[derive(Clone)]
    struct TestClock(Arc<TestClockState>);

    impl TestClock {
        fn new(now_ns: u64) -> Self {
            Self(Arc::new(TestClockState {
                now_ns: AtomicU64::new(now_ns),
                failing: AtomicBool::new(false),
            }))
        }

        fn set(&self, now_ns: u64) {
            self.0.now_ns.store(now_ns, Ordering::Relaxed);
        }

        fn set_failing(&self, failing: bool) {
            self.0.failing.store(failing, Ordering::Relaxed);
        }
    }

    impl MonotonicClock for TestClock {
        fn now(&self) -> Result<MonotonicTimestamp, ClockError> {
            if self.0.failing.load(Ordering::Relaxed) {
                return Err(ClockError::ElapsedNanosecondsOutOfRange {
                    elapsed_nanoseconds: u128::MAX,
                });
            }
            Ok(MonotonicTimestamp::from_nanos_since_epoch(
                self.0.now_ns.load(Ordering::Relaxed),
            ))
        }
    }

    fn stream_epoch() -> StreamEpochId {
        StreamEpochId::try_new(17).expect("non-zero test epoch")
    }

    fn scene_config() -> SceneMotionConfig {
        SceneMotionConfig::new(
            SamplingGeometry::try_new(2, 2).expect("test sampling grid"),
            MotionThresholds::try_new(
                1,
                PositiveUnitAmount::try_from_basis_points(1).expect("positive active fraction"),
            )
            .expect("test motion thresholds"),
        )
    }

    fn render_style() -> EyeRenderStyle {
        EyeRenderStyle::new(
            UnitAmount::try_from_basis_points(5_000).expect("test brightness"),
            [4, 5, 6],
            true,
        )
    }

    fn bridge(clock: TestClock) -> RgbExpressionBridge<TestClock> {
        RgbExpressionBridge::from_parts(
            stream_epoch(),
            scene_config(),
            NonZeroDuration::try_from_nanos(100).expect("test freshness"),
            render_style(),
            None,
            clock,
        )
    }

    fn frame<'a>(
        stream: StreamId,
        capture_sequence: u64,
        device_timestamp_ns: i64,
        width_px: u32,
        height_px: u32,
        stride_bytes: u32,
        pixels: &'a [u8],
    ) -> BorrowedOakFrame<'a> {
        BorrowedOakFrame {
            stream,
            capture_sequence,
            device_timestamp_ns,
            width_px,
            height_px,
            stride_bytes,
            pixels,
        }
    }

    fn rgb<'a>(
        capture_sequence: u64,
        device_timestamp_ns: i64,
        pixels: &'a [u8; 12],
    ) -> BorrowedOakFrame<'a> {
        frame(
            StreamId::Rgb,
            capture_sequence,
            device_timestamp_ns,
            2,
            2,
            6,
            pixels,
        )
    }

    fn localized_change() -> [u8; 12] {
        let mut pixels = [0_u8; 12];
        pixels[..3].fill(255);
        pixels
    }

    fn queued_rgb<'a>(
        capture_sequence: u64,
        device_timestamp_ns: i64,
        observed_at_ns: u64,
        pixels: &'a [u8; 12],
    ) -> IngressObservedRgbFrame<BorrowedOakFrame<'a>> {
        IngressObservedRgbFrame::new(
            rgb(capture_sequence, device_timestamp_ns, pixels),
            MonotonicTimestamp::from_nanos_since_epoch(observed_at_ns),
        )
    }

    fn face_batch(
        capture_sequence: u64,
        detector_sequence: u64,
        observed_at_ns: u64,
        detection: Option<(u32, u32, u32, u32)>,
    ) -> FaceDetectionBatch {
        let layout = ImageLayout::try_new(2, 2, 6, ChannelOrder::Bgr).expect("face test layout");
        let freshness = FreshnessWindow::from_ttl(
            MonotonicTimestamp::from_nanos_since_epoch(observed_at_ns),
            NonZeroDuration::try_from_nanos(100).expect("face test freshness"),
        )
        .expect("face test deadline");
        let observation = RgbObservation::new(
            FrameId::new(stream_epoch(), capture_sequence),
            layout,
            freshness,
        );
        let parsed = detection.map(|(left, top, width, height)| {
            FaceDetection::try_new(
                layout,
                left,
                top,
                width,
                height,
                -3.0,
                FaceDetectorSource::Frontal,
            )
            .expect("face test detection")
        });
        FaceDetectionBatch::try_new(
            observation,
            DetectorResultSequence::new(detector_sequence),
            0,
            parsed.as_slice(),
        )
        .expect("face test batch")
    }

    #[test]
    fn cold_start_returns_only_styled_neutral_prepared_intent() {
        let clock = TestClock::new(10);
        let eye_clock = clock.clone();
        let mut bridge = bridge(clock);
        assert_eq!(
            bridge
                .clone_clock_for_eye_actor()
                .now()
                .expect("bridge clock"),
            eye_clock.now().expect("eye clock")
        );

        let outcome = bridge
            .process_borrowed(rgb(4, 1_000, &[0; 12]))
            .expect("cold start");
        assert!(matches!(outcome, RgbExpressionBridgeOutcome::ColdStart(_)));
        let prepared = outcome.into_prepared();
        assert_eq!(prepared.generated_at().nanos_since_epoch(), 10);
        assert_eq!(prepared.intent().expression(), Expression::Neutral);
        assert_eq!(prepared.intent().brightness().get(), 500);
        assert_eq!(prepared.intent().color_rgb(), [4, 5, 6]);
        assert!(prepared.intent().flags().requests_blink());
    }

    #[test]
    fn consecutive_frame_uses_borrowed_rgb_and_scene_reaction_inputs() {
        let clock = TestClock::new(10);
        let mut bridge = bridge(clock.clone());
        bridge
            .process_borrowed(rgb(4, 1_000, &[0; 12]))
            .expect("prime extractor");
        clock.set(20);

        let changed = localized_change();
        let outcome = bridge
            .process_borrowed(rgb(5, 2_000, &changed))
            .expect("consecutive expression");
        assert!(matches!(
            outcome,
            RgbExpressionBridgeOutcome::Consecutive(_)
        ));
        let prepared = outcome.into_prepared();
        assert_eq!(prepared.generated_at().nanos_since_epoch(), 20);
        assert_eq!(prepared.intent().expression(), Expression::Curious);
    }

    #[test]
    fn established_face_owns_important_gaze_without_using_haar_rank_as_confidence() {
        assert_eq!(FACE_ATTENTION_STRENGTH, PositiveUnitAmount::ONE);
        let clock = TestClock::new(10);
        let mut bridge = bridge(clock.clone());
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());

        let first_batch = face_batch(4, 1, 10, Some((1, 0, 1, 1)));
        let first_face = tracker
            .update(&first_batch, MonotonicTimestamp::from_nanos_since_epoch(10))
            .expect("acquisition result");
        let first = bridge
            .process_ingress_observed_borrowed_with_face(
                queued_rgb(4, 1_000, 10, &[0; 12]),
                Some(first_face),
            )
            .expect("acquiring face frame");
        assert_eq!(
            first.into_prepared().intent().expression(),
            Expression::Neutral,
            "one detector result is not an established target"
        );

        clock.set(20);
        let second_batch = face_batch(5, 2, 20, Some((1, 0, 1, 1)));
        let second_face = tracker
            .update(
                &second_batch,
                MonotonicTimestamp::from_nanos_since_epoch(20),
            )
            .expect("tracked result");
        let second = bridge
            .process_ingress_observed_borrowed_with_face(
                queued_rgb(5, 2_000, 20, &[0; 12]),
                Some(second_face),
            )
            .expect("tracked face frame")
            .into_prepared();
        assert_eq!(second.intent().expression(), Expression::Curious);
        assert_eq!(second.intent().gaze_x().get(), 500);
        assert_eq!(second.intent().gaze_y().get(), 500);
        assert_eq!(
            second.valid_until_exclusive(),
            Some(
                second_batch
                    .observation()
                    .freshness()
                    .valid_until_exclusive()
            )
        );
    }

    #[test]
    fn coasted_face_uses_current_frame_freshness_capped_by_loss_deadline() {
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        for (camera, detector, observed) in [(4, 1, 10), (5, 2, 20)] {
            tracker
                .update(
                    &face_batch(camera, detector, observed, Some((1, 0, 1, 1))),
                    MonotonicTimestamp::from_nanos_since_epoch(observed),
                )
                .expect("establish face");
        }
        let empty = face_batch(6, 3, 30, None);
        let coasted = tracker
            .update(&empty, MonotonicTimestamp::from_nanos_since_epoch(30))
            .expect("coasted result");
        let intent = face_attention_intent(coasted)
            .expect("valid coast freshness")
            .expect("coasting retains attention");
        assert_eq!(intent.kind(), ExpressionKind::Attentive);
        assert_eq!(intent.strength(), FACE_ATTENTION_STRENGTH);
        assert_eq!(intent.priority(), ExpressionPriority::Important);
        assert_eq!(
            intent.freshness(),
            empty.observation().freshness(),
            "the current RGB deadline is earlier than the two-second loss deadline"
        );
    }

    #[test]
    fn face_result_cannot_be_retagged_onto_another_rgb_frame() {
        let clock = TestClock::new(10);
        let mut bridge = bridge(clock);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let batch = face_batch(5, 1, 10, Some((1, 0, 1, 1)));
        let face = tracker
            .update(&batch, MonotonicTimestamp::from_nanos_since_epoch(10))
            .expect("face update");
        let mismatched_frame = queued_rgb(4, 1_000, 10, &[0; 12]);
        assert_eq!(
            bridge.process_ingress_observed_borrowed_with_face(mismatched_frame, Some(face)),
            Err(RgbExpressionBridgeError::FaceObservationMismatch {
                frame: face_batch(4, 99, 10, None).observation(),
                face: batch.observation(),
            })
        );

        assert!(matches!(
            bridge
                .process_ingress_observed_borrowed(queued_rgb(4, 1_000, 10, &[0; 12]))
                .expect("mismatch did not advance scene or device-clock state"),
            RgbExpressionBridgeOutcome::ColdStart(_)
        ));
    }

    #[test]
    fn replace_latest_gaps_compare_motion_and_report_exact_skip_evidence() {
        let clock = TestClock::new(10);
        let mut bridge = bridge(clock.clone());
        bridge
            .process_borrowed(rgb(40, 1_000, &[0; 12]))
            .expect("prime extractor");
        clock.set(20);

        let changed = localized_change();
        let outcome = bridge
            .process_borrowed(rgb(42, 2_000, &changed))
            .expect("forward gap remains comparable");
        let RgbExpressionBridgeOutcome::ForwardGap { gap, prepared } = outcome else {
            panic!("forward gap must retain gap evidence");
        };
        assert_eq!(gap.previous_sequence(), 40);
        assert_eq!(gap.actual_sequence(), 42);
        assert_eq!(gap.skipped_sequence_count().get(), 1);
        assert_eq!(prepared.intent().expression(), Expression::Curious);

        clock.set(30);
        let second_gap = bridge
            .process_borrowed(rgb(45, 3_000, &[0; 12]))
            .expect("another replace-latest gap");
        let RgbExpressionBridgeOutcome::ForwardGap { gap, prepared } = second_gap else {
            panic!("second forward gap must retain gap evidence");
        };
        assert_eq!(gap.previous_sequence(), 42);
        assert_eq!(gap.actual_sequence(), 45);
        assert_eq!(gap.skipped_sequence_count().get(), 2);
        assert_eq!(
            prepared.intent().expression(),
            Expression::Curious,
            "repeated replace-latest gaps must not repeatedly cold-reset the eyes"
        );

        clock.set(40);
        let changed_again = localized_change();
        let next = bridge
            .process_borrowed(rgb(46, 4_000, &changed_again))
            .expect("consecutive frame after gaps");
        assert!(matches!(next, RgbExpressionBridgeOutcome::Consecutive(_)));
        assert_eq!(
            next.into_prepared().intent().expression(),
            Expression::Curious
        );
    }

    #[test]
    fn malformed_and_non_tight_layouts_and_non_rgb_streams_are_distinct() {
        let clock = TestClock::new(10);
        let mut bridge = bridge(clock);
        assert_eq!(
            bridge.process_borrowed(frame(StreamId::MonoLeft, 1, 1, 2, 2, 6, &[0; 12])),
            Err(RgbExpressionBridgeError::NotRgbStream {
                actual: StreamId::MonoLeft
            })
        );
        assert_eq!(
            bridge.process_borrowed(frame(StreamId::Rgb, 1, 1, 2, 2, 7, &[0; 14])),
            Err(RgbExpressionBridgeError::NonTightBgrLayout {
                width_px: 2,
                expected_stride_bytes: 6,
                actual_stride_bytes: 7,
            })
        );
        assert_eq!(
            bridge.process_borrowed(frame(StreamId::Rgb, 1, 1, 2, 2, 6, &[0; 11])),
            Err(RgbExpressionBridgeError::Layout(
                ImageLayoutError::PixelLengthMismatch {
                    expected: 12,
                    actual: 11,
                }
            ))
        );
    }

    #[test]
    fn duplicate_and_regressed_capture_sequences_are_not_reset() {
        let clock = TestClock::new(10);
        let mut bridge = bridge(clock.clone());
        bridge
            .process_borrowed(rgb(5, 1_000, &[0; 12]))
            .expect("prime extractor");

        clock.set(20);
        assert_eq!(
            bridge.process_borrowed(rgb(5, 2_000, &[0; 12])),
            Err(RgbExpressionBridgeError::SceneMotion(
                SceneMotionError::DuplicateFrame { sequence: 5 }
            ))
        );
        clock.set(30);
        assert_eq!(
            bridge.process_borrowed(rgb(4, 3_000, &[0; 12])),
            Err(RgbExpressionBridgeError::SceneMotion(
                SceneMotionError::OutOfOrderFrame {
                    previous: 5,
                    actual: 4,
                }
            ))
        );

        clock.set(40);
        assert!(matches!(
            bridge
                .process_borrowed(rgb(6, 4_000, &[0; 12]))
                .expect("rejections did not advance state"),
            RgbExpressionBridgeOutcome::Consecutive(_)
        ));
    }

    #[test]
    fn layout_and_device_clock_faults_are_not_hidden_by_forward_gap_admission() {
        let clock = TestClock::new(10);
        let mut bridge = bridge(clock.clone());
        bridge
            .process_borrowed(rgb(5, 1_000, &[0; 12]))
            .expect("prime extractor");

        clock.set(20);
        assert!(matches!(
            bridge.process_borrowed(frame(StreamId::Rgb, 7, 2_000, 2, 3, 6, &[0; 18])),
            Err(RgbExpressionBridgeError::SceneMotion(
                SceneMotionError::LayoutChanged { .. }
            ))
        ));
        clock.set(30);
        assert_eq!(
            bridge.process_borrowed(rgb(7, 1_000, &[0; 12])),
            Err(RgbExpressionBridgeError::DeviceCaptureClockNotIncreasing {
                previous_ns: 1_000,
                actual_ns: 1_000,
            })
        );

        clock.set(40);
        let changed = localized_change();
        let recovered = bridge
            .process_borrowed(rgb(7, 3_000, &changed))
            .expect("layout and device-clock rejections did not advance either state owner");
        let RgbExpressionBridgeOutcome::ForwardGap { gap, prepared } = recovered else {
            panic!("recovery must retain the original sequence baseline");
        };
        assert_eq!(gap.previous_sequence(), 5);
        assert_eq!(gap.actual_sequence(), 7);
        assert_eq!(gap.skipped_sequence_count().get(), 1);
        assert_eq!(prepared.intent().expression(), Expression::Curious);
    }

    #[test]
    fn host_clock_regression_is_typed_and_does_not_advance_state() {
        let clock = TestClock::new(20);
        let mut bridge = bridge(clock.clone());
        bridge
            .process_borrowed(rgb(5, 1_000, &[0; 12]))
            .expect("prime extractor");

        clock.set(19);
        assert_eq!(
            bridge.process_borrowed(rgb(6, 2_000, &[0; 12])),
            Err(RgbExpressionBridgeError::SceneMotion(
                SceneMotionError::HostClockRegressed {
                    previous_ns: 20,
                    actual_ns: 19,
                }
            ))
        );
        clock.set(30);
        assert!(bridge.process_borrowed(rgb(6, 2_000, &[0; 12])).is_ok());
    }

    #[test]
    fn queued_frame_is_stale_at_the_exclusive_deadline_without_advancing_state() {
        let clock = TestClock::new(10);
        let mut bridge = bridge(clock.clone());
        bridge
            .process_ingress_observed_borrowed(queued_rgb(5, 1_000, 10, &[0; 12]))
            .expect("prime extractor from ingress observation");

        clock.set(120);
        let changed = localized_change();
        assert_eq!(
            bridge.process_ingress_observed_borrowed(queued_rgb(6, 2_000, 20, &changed)),
            Err(RgbExpressionBridgeError::SceneMotion(
                SceneMotionError::StaleFrame {
                    deadline_ns: 120,
                    now_ns: 120,
                }
            ))
        );

        clock.set(130);
        let recovered = bridge
            .process_ingress_observed_borrowed(queued_rgb(6, 2_000, 130, &changed))
            .expect("stale rejection retained sequence, pixels, and device-clock baselines");
        assert!(matches!(
            recovered,
            RgbExpressionBridgeOutcome::Consecutive(_)
        ));
        assert_eq!(
            recovered.into_prepared().intent().expression(),
            Expression::Curious
        );
    }

    #[test]
    fn queued_host_time_failures_do_not_advance_state() {
        let clock = TestClock::new(10);
        let mut bridge = bridge(clock.clone());
        bridge
            .process_ingress_observed_borrowed(queued_rgb(5, 1_000, 10, &[0; 12]))
            .expect("prime extractor from ingress observation");

        clock.set(9);
        assert_eq!(
            bridge.process_ingress_observed_borrowed(queued_rgb(6, 2_000, 9, &[0; 12])),
            Err(RgbExpressionBridgeError::SceneMotion(
                SceneMotionError::HostClockRegressed {
                    previous_ns: 10,
                    actual_ns: 9,
                }
            ))
        );

        clock.set(20);
        assert_eq!(
            bridge.process_ingress_observed_borrowed(queued_rgb(6, 2_000, 10, &[0; 12])),
            Err(RgbExpressionBridgeError::SceneMotion(
                SceneMotionError::ObservationClockNotIncreasing {
                    previous_ns: 10,
                    actual_ns: 10,
                }
            ))
        );
        assert_eq!(
            bridge.process_ingress_observed_borrowed(queued_rgb(6, 2_000, 21, &[0; 12])),
            Err(RgbExpressionBridgeError::SceneMotion(
                SceneMotionError::FrameFromFuture {
                    observed_at_ns: 21,
                    now_ns: 20,
                }
            ))
        );

        clock.set_failing(true);
        assert_eq!(
            bridge.process_ingress_observed_borrowed(queued_rgb(6, 2_000, 20, &[0; 12])),
            Err(RgbExpressionBridgeError::Clock(
                ClockError::ElapsedNanosecondsOutOfRange {
                    elapsed_nanoseconds: u128::MAX,
                }
            ))
        );

        clock.set_failing(false);
        let recovered = bridge
            .process_ingress_observed_borrowed(queued_rgb(6, 2_000, 20, &[0; 12]))
            .expect("all queued host-time rejections retained every admission baseline");
        assert!(matches!(
            recovered,
            RgbExpressionBridgeOutcome::Consecutive(_)
        ));
    }

    #[test]
    fn active_rgb_reaction_preserves_the_natural_head_invariant() {
        assert_eq!(RGB_EXPRESSION_HEAD_POLICY, HeadMotionPolicy::NaturalHold);
        let clock = TestClock::new(10);
        let mut bridge = bridge(clock.clone());
        bridge
            .process_borrowed(rgb(1, 1_000, &[0; 12]))
            .expect("prime extractor");
        clock.set(20);
        let changed = localized_change();
        let prepared = bridge
            .process_borrowed(rgb(2, 2_000, &changed))
            .expect("NaturalHold is accepted by the headless eye adapter")
            .into_prepared();
        assert_eq!(prepared.intent().expression(), Expression::Curious);
    }

    #[test]
    fn optional_gaze_geometry_projects_points_and_rays_without_head_authority() {
        assert_eq!(RGB_EXPRESSION_HEAD_POLICY, HeadMotionPolicy::NaturalHold);
        let clock = TestClock::new(10);
        let unavailable = bridge(clock.clone());
        let point = OakCameraTargetPoint::parse([0.0, 0.0, 1.0]).unwrap();
        assert_eq!(
            unavailable.project_head_gaze_to_point(point),
            Err(RgbHeadGazeProjectionError::GeometryUnavailable)
        );

        let geometry = CameraToHeadGazeExtrinsics::parse(
            kiko_expression_runtime::CameraToHeadGazeExtrinsicsInput {
                head_origin_in_camera_m: [0.0, -0.25, -0.20],
                neutral_head_from_camera_quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
            },
        )
        .unwrap();
        let configured = RgbExpressionBridge::from_parts(
            stream_epoch(),
            scene_config(),
            NonZeroDuration::try_from_nanos(100).expect("test freshness"),
            render_style(),
            Some(geometry),
            clock,
        );
        let point_gaze = configured.project_head_gaze_to_point(point).unwrap();
        assert_eq!(point_gaze.yaw_right_rad(), 0.0);
        assert!(point_gaze.pitch_down_rad() > 0.0);

        let ray = OakCameraTargetRay::parse([0.0, 0.0, 1.0]).unwrap();
        let ray_gaze = configured
            .project_head_gaze_along_ray(ray, CameraForwardDepthMeters::parse(1.0).unwrap())
            .unwrap();
        assert_eq!(ray_gaze, point_gaze);
    }
}
