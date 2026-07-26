//! Bounded, hardware-independent face-target association.
//!
//! This is a hardened derivative of Kiko's Fable face-following policy, not a
//! byte-for-byte reproduction. It deliberately retains the useful behavioural
//! constants (widest-first acquisition, strict 90/140 px association gates,
//! two-result acquisition, five-result closer-target switching, 0.45 EMA, and
//! a two-second loss grace) while adding explicit stream/result provenance,
//! fixed-capacity admission, transactional errors, and capture-time deadline
//! semantics.
//!
//! The OpenCV Haar `detector_level_weight` is only an opaque, finite ranking
//! value. It is neither a probability nor a calibrated confidence and this
//! module never converts it into a `PersonObservation`. Apparent face width is
//! used only for deterministic target preference; no metric range is inferred.

use core::{
    cmp::Ordering,
    fmt,
    num::{NonZeroU8, NonZeroU32, NonZeroU64},
};

use kiko_expression_core::{
    Deadline, FrameId, FreshnessWindow, ImageLayout, ImagePoint, MonotonicTimestamp,
    NonZeroDuration, PersonTrackId, PositiveUnitAmount, RgbObservation, StreamEpochId, UnitAmount,
};

const NORMALIZED_SCALE: u128 = 10_000;
const PIXEL_SUBUNITS_PER_PIXEL: u64 = 10_000;

/// Maximum number of already-retained detections accepted from one detector result.
///
/// The detector boundary owns any raw-result cap and must report its truncation
/// count. This tracker rejects an oversized retained batch instead of silently
/// applying a second, incompatible cap.
pub const MAX_FACE_DETECTIONS: usize = 16;

/// Largest configurable centre/origin association threshold.
pub const MAX_FACE_PIXEL_DISTANCE_PX: u32 = 4_096;

/// Largest configurable number of consecutive detector results.
pub const MAX_FACE_CONSECUTIVE_RESULTS: u8 = 32;

/// Longest configurable coast interval.
pub const MAX_FACE_COASTING_DURATION_NS: u64 = 10_000_000_000;

/// Largest allowed closer-target apparent-width ratio.
pub const MAX_CLOSER_FACE_WIDTH_RATIO: u16 = 8;

/// Fable-derived default first-candidate origin tolerance.
pub const DEFAULT_FACE_ACQUISITION_DISTANCE_PX: u32 = 90;

/// Fable-derived default tracked-centre association tolerance.
pub const DEFAULT_FACE_ASSOCIATION_DISTANCE_PX: u32 = 140;

/// Fable-derived default acquisition confirmation count.
pub const DEFAULT_FACE_ACQUISITION_RESULTS: u8 = 2;

/// Fable-derived default closer-target confirmation count.
pub const DEFAULT_FACE_SWITCH_RESULTS: u8 = 5;

/// Fable-derived exponential smoothing gain in basis points.
pub const DEFAULT_FACE_SMOOTHING_ALPHA_BASIS_POINTS: u16 = 4_500;

/// Fable-derived target-loss grace interval.
pub const DEFAULT_FACE_COASTING_DURATION_NS: u64 = 2_000_000_000;

/// Sequence of detector results accepted from one uninterrupted detector worker.
///
/// This is intentionally distinct from [`FrameId`]. A camera may skip raw
/// frames while consecutive detector results still provide consecutive
/// association evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DetectorResultSequence(u64);

impl DetectorResultSequence {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Which Haar search produced one retained rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaceDetectorSource {
    Frontal,
    Profile,
    MirroredProfile,
}

/// Exact finite OpenCV Haar `detector_level_weight` bits.
///
/// Negative values, zero, and values above one are all valid. Ordering uses
/// IEEE total ordering solely to make otherwise-equal detections deterministic.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DetectorLevelWeight(u64);

impl DetectorLevelWeight {
    pub fn try_new(value: f64) -> Result<Self, FaceDetectionError> {
        if !value.is_finite() {
            return Err(FaceDetectionError::NonFiniteDetectorLevelWeight);
        }
        Ok(Self(value.to_bits()))
    }

    pub const fn to_bits(self) -> u64 {
        self.0
    }

    pub fn as_f64(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl fmt::Debug for DetectorLevelWeight {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("DetectorLevelWeight")
            .field(&self.as_f64())
            .finish()
    }
}

impl PartialOrd for DetectorLevelWeight {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DetectorLevelWeight {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_f64().total_cmp(&other.as_f64())
    }
}

/// Integer rectangle checked against the exact layout from which it was detected.
///
/// Width and height are nonzero by construction. No integer coordinate is
/// weakened through a floating-point representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceRectangle {
    layout: ImageLayout,
    left_px: u32,
    top_px: u32,
    right_exclusive_px: u32,
    bottom_exclusive_px: u32,
}

impl FaceRectangle {
    pub fn try_from_origin_size(
        layout: ImageLayout,
        left_px: u32,
        top_px: u32,
        width_px: u32,
        height_px: u32,
    ) -> Result<Self, FaceDetectionError> {
        let Some(_width) = NonZeroU32::new(width_px) else {
            return Err(FaceDetectionError::ZeroWidth);
        };
        let Some(_height) = NonZeroU32::new(height_px) else {
            return Err(FaceDetectionError::ZeroHeight);
        };
        let right_exclusive_px = left_px
            .checked_add(width_px)
            .ok_or(FaceDetectionError::RectangleArithmeticOverflow)?;
        let bottom_exclusive_px = top_px
            .checked_add(height_px)
            .ok_or(FaceDetectionError::RectangleArithmeticOverflow)?;
        if left_px >= layout.width_px()
            || top_px >= layout.height_px()
            || right_exclusive_px > layout.width_px()
            || bottom_exclusive_px > layout.height_px()
        {
            return Err(FaceDetectionError::RectangleOutsideFrame {
                frame_width_px: layout.width_px(),
                frame_height_px: layout.height_px(),
            });
        }
        Ok(Self {
            layout,
            left_px,
            top_px,
            right_exclusive_px,
            bottom_exclusive_px,
        })
    }

    pub const fn layout(self) -> ImageLayout {
        self.layout
    }

    pub const fn left_px(self) -> u32 {
        self.left_px
    }

    pub const fn top_px(self) -> u32 {
        self.top_px
    }

    pub const fn right_exclusive_px(self) -> u32 {
        self.right_exclusive_px
    }

    pub const fn bottom_exclusive_px(self) -> u32 {
        self.bottom_exclusive_px
    }

    pub const fn width_px(self) -> u32 {
        self.right_exclusive_px - self.left_px
    }

    pub const fn height_px(self) -> u32 {
        self.bottom_exclusive_px - self.top_px
    }

    pub const fn area_px(self) -> u64 {
        self.width_px() as u64 * self.height_px() as u64
    }

    fn center_x_subpixels(self) -> u64 {
        (u64::from(self.left_px) + u64::from(self.right_exclusive_px))
            * (PIXEL_SUBUNITS_PER_PIXEL / 2)
    }

    fn center_y_subpixels(self) -> u64 {
        (u64::from(self.top_px) + u64::from(self.bottom_exclusive_px))
            * (PIXEL_SUBUNITS_PER_PIXEL / 2)
    }

    fn width_subpixels(self) -> u64 {
        u64::from(self.width_px()) * PIXEL_SUBUNITS_PER_PIXEL
    }
}

/// One fully parsed detector result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceDetection {
    rectangle: FaceRectangle,
    detector_level_weight: DetectorLevelWeight,
    source: FaceDetectorSource,
}

impl FaceDetection {
    pub fn try_new(
        layout: ImageLayout,
        left_px: u32,
        top_px: u32,
        width_px: u32,
        height_px: u32,
        detector_level_weight: f64,
        source: FaceDetectorSource,
    ) -> Result<Self, FaceDetectionError> {
        Ok(Self {
            rectangle: FaceRectangle::try_from_origin_size(
                layout, left_px, top_px, width_px, height_px,
            )?,
            detector_level_weight: DetectorLevelWeight::try_new(detector_level_weight)?,
            source,
        })
    }

    pub const fn rectangle(self) -> FaceRectangle {
        self.rectangle
    }

    pub const fn detector_level_weight(self) -> DetectorLevelWeight {
        self.detector_level_weight
    }

    pub const fn source(self) -> FaceDetectorSource {
        self.source
    }

    pub fn center(self) -> ImagePoint {
        ImagePoint::new(
            normalized_subpixel_coordinate(
                self.rectangle.center_x_subpixels(),
                self.rectangle.layout().width_px(),
            ),
            normalized_subpixel_coordinate(
                self.rectangle.center_y_subpixels(),
                self.rectangle.layout().height_px(),
            ),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceDetectionError {
    ZeroWidth,
    ZeroHeight,
    RectangleArithmeticOverflow,
    RectangleOutsideFrame {
        frame_width_px: u32,
        frame_height_px: u32,
    },
    NonFiniteDetectorLevelWeight,
}

impl fmt::Display for FaceDetectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid face detection: {self:?}")
    }
}

impl core::error::Error for FaceDetectionError {}

/// Fixed-capacity, typed output from exactly one detector evaluation.
///
/// All retained detections are sorted widest-first, then by height, opaque
/// level weight, row-major coordinates (`top`, then `left`), and source. This
/// is the same order as the OAK detector boundary, so reparsing cannot change
/// which otherwise-equal candidate is selected. No detection is filtered by
/// its level weight. `detector_truncated_count` is copied from the detector
/// boundary and no additional truncation occurs here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceDetectionBatch {
    observation: RgbObservation,
    detector_result_sequence: DetectorResultSequence,
    detections: [Option<FaceDetection>; MAX_FACE_DETECTIONS],
    retained_count: u8,
    detector_truncated_count: u32,
}

impl FaceDetectionBatch {
    pub fn try_new(
        observation: RgbObservation,
        detector_result_sequence: DetectorResultSequence,
        detector_truncated_count: u32,
        retained_detections: &[FaceDetection],
    ) -> Result<Self, FaceDetectionBatchError> {
        if retained_detections.len() > MAX_FACE_DETECTIONS {
            return Err(FaceDetectionBatchError::RetainedCountExceedsCapacity {
                actual: retained_detections.len(),
                maximum: MAX_FACE_DETECTIONS,
            });
        }
        let mut detections = [None; MAX_FACE_DETECTIONS];
        for (index, detection) in retained_detections.iter().copied().enumerate() {
            if detection.rectangle().layout() != observation.layout() {
                return Err(FaceDetectionBatchError::DetectionLayoutMismatch {
                    index,
                    expected: observation.layout(),
                    actual: detection.rectangle().layout(),
                });
            }
            detections[index] = Some(detection);
        }
        let retained_count = u8::try_from(retained_detections.len())
            .expect("face detection capacity is representable as u8");
        let mut batch = Self {
            observation,
            detector_result_sequence,
            detections,
            retained_count,
            detector_truncated_count,
        };
        batch.sort_retained();
        Ok(batch)
    }

    pub const fn observation(&self) -> RgbObservation {
        self.observation
    }

    pub const fn detector_result_sequence(&self) -> DetectorResultSequence {
        self.detector_result_sequence
    }

    pub const fn retained_count(&self) -> usize {
        self.retained_count as usize
    }

    pub const fn is_empty(&self) -> bool {
        self.retained_count == 0
    }

    pub const fn detector_truncated_count(&self) -> u32 {
        self.detector_truncated_count
    }

    pub fn get(&self, index: usize) -> Option<FaceDetection> {
        if index >= self.retained_count() {
            None
        } else {
            self.detections[index]
        }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = FaceDetection> + '_ {
        self.detections[..self.retained_count()]
            .iter()
            .map(|detection| detection.expect("all retained face-detection slots are populated"))
    }

    fn sort_retained(&mut self) {
        let len = self.retained_count();
        let mut index = 1;
        while index < len {
            let detection =
                self.detections[index].expect("retained face-detection slot is populated");
            let mut position = index;
            while position > 0 {
                let previous = self.detections[position - 1]
                    .expect("retained face-detection slot is populated");
                if compare_detection(previous, detection) != Ordering::Greater {
                    break;
                }
                self.detections[position] = Some(previous);
                position -= 1;
            }
            self.detections[position] = Some(detection);
            index += 1;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceDetectionBatchError {
    RetainedCountExceedsCapacity {
        actual: usize,
        maximum: usize,
    },
    DetectionLayoutMismatch {
        index: usize,
        expected: ImageLayout,
        actual: ImageLayout,
    },
}

impl fmt::Display for FaceDetectionBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid face detector batch: {self:?}")
    }
}

impl core::error::Error for FaceDetectionBatchError {}

/// Nonzero, bounded pixel threshold for one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FacePixelDistance(NonZeroU32);

impl FacePixelDistance {
    pub const fn try_new(value: u32) -> Result<Self, FaceTrackingConfigError> {
        let Some(value) = NonZeroU32::new(value) else {
            return Err(FaceTrackingConfigError::ZeroPixelDistance);
        };
        if value.get() > MAX_FACE_PIXEL_DISTANCE_PX {
            return Err(FaceTrackingConfigError::PixelDistanceTooLarge {
                actual: value.get(),
                maximum: MAX_FACE_PIXEL_DISTANCE_PX,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }

    fn subpixels(self) -> u64 {
        u64::from(self.get()) * PIXEL_SUBUNITS_PER_PIXEL
    }
}

/// Nonzero, bounded consecutive-detector-result threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConsecutiveFaceResults(NonZeroU8);

impl ConsecutiveFaceResults {
    pub const fn try_new(value: u8) -> Result<Self, FaceTrackingConfigError> {
        let Some(value) = NonZeroU8::new(value) else {
            return Err(FaceTrackingConfigError::ZeroConsecutiveResults);
        };
        if value.get() > MAX_FACE_CONSECUTIVE_RESULTS {
            return Err(FaceTrackingConfigError::TooManyConsecutiveResults {
                actual: value.get(),
                maximum: MAX_FACE_CONSECUTIVE_RESULTS,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u8 {
        self.0.get()
    }
}

/// Strict apparent-width ratio required for a closer-target challenger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CloserFaceWidthRatio {
    numerator: NonZeroU32,
    denominator: NonZeroU32,
}

impl CloserFaceWidthRatio {
    pub const fn try_new(
        numerator: u32,
        denominator: u32,
    ) -> Result<Self, FaceTrackingConfigError> {
        let Some(numerator) = NonZeroU32::new(numerator) else {
            return Err(FaceTrackingConfigError::ZeroCloserRatioNumerator);
        };
        let Some(denominator) = NonZeroU32::new(denominator) else {
            return Err(FaceTrackingConfigError::ZeroCloserRatioDenominator);
        };
        if numerator.get() <= denominator.get() {
            return Err(FaceTrackingConfigError::CloserRatioNotGreaterThanOne {
                numerator: numerator.get(),
                denominator: denominator.get(),
            });
        }
        if numerator.get() as u64 > denominator.get() as u64 * MAX_CLOSER_FACE_WIDTH_RATIO as u64 {
            return Err(FaceTrackingConfigError::CloserRatioTooLarge {
                numerator: numerator.get(),
                denominator: denominator.get(),
                maximum: MAX_CLOSER_FACE_WIDTH_RATIO,
            });
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    pub const fn numerator(self) -> u32 {
        self.numerator.get()
    }

    pub const fn denominator(self) -> u32 {
        self.denominator.get()
    }

    fn challenger_is_strictly_larger(
        self,
        challenger_width_subpixels: u64,
        tracked_width_subpixels: u64,
    ) -> bool {
        u128::from(challenger_width_subpixels) * u128::from(self.denominator())
            > u128::from(tracked_width_subpixels) * u128::from(self.numerator())
    }
}

/// Nonzero, bounded target-coasting interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceCoastingDuration(NonZeroDuration);

impl FaceCoastingDuration {
    pub const fn try_new(duration: NonZeroDuration) -> Result<Self, FaceTrackingConfigError> {
        if duration.as_nanos() > MAX_FACE_COASTING_DURATION_NS {
            return Err(FaceTrackingConfigError::CoastingDurationTooLong {
                actual_ns: duration.as_nanos(),
                maximum_ns: MAX_FACE_COASTING_DURATION_NS,
            });
        }
        Ok(Self(duration))
    }

    pub const fn get(self) -> NonZeroDuration {
        self.0
    }
}

/// Fully checked policy for bounded face-target association.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceTrackingConfig {
    acquisition_origin_distance: FacePixelDistance,
    association_center_distance: FacePixelDistance,
    acquisition_results: ConsecutiveFaceResults,
    switch_results: ConsecutiveFaceResults,
    closer_width_ratio: CloserFaceWidthRatio,
    smoothing_alpha: PositiveUnitAmount,
    coasting_duration: FaceCoastingDuration,
}

impl FaceTrackingConfig {
    pub const fn new(
        acquisition_origin_distance: FacePixelDistance,
        association_center_distance: FacePixelDistance,
        acquisition_results: ConsecutiveFaceResults,
        switch_results: ConsecutiveFaceResults,
        closer_width_ratio: CloserFaceWidthRatio,
        smoothing_alpha: PositiveUnitAmount,
        coasting_duration: FaceCoastingDuration,
    ) -> Self {
        Self {
            acquisition_origin_distance,
            association_center_distance,
            acquisition_results,
            switch_results,
            closer_width_ratio,
            smoothing_alpha,
            coasting_duration,
        }
    }

    pub const fn acquisition_origin_distance(self) -> FacePixelDistance {
        self.acquisition_origin_distance
    }

    pub const fn association_center_distance(self) -> FacePixelDistance {
        self.association_center_distance
    }

    pub const fn acquisition_results(self) -> ConsecutiveFaceResults {
        self.acquisition_results
    }

    pub const fn switch_results(self) -> ConsecutiveFaceResults {
        self.switch_results
    }

    pub const fn closer_width_ratio(self) -> CloserFaceWidthRatio {
        self.closer_width_ratio
    }

    pub const fn smoothing_alpha(self) -> PositiveUnitAmount {
        self.smoothing_alpha
    }

    pub const fn coasting_duration(self) -> FaceCoastingDuration {
        self.coasting_duration
    }
}

impl Default for FaceTrackingConfig {
    fn default() -> Self {
        Self::new(
            FacePixelDistance::try_new(DEFAULT_FACE_ACQUISITION_DISTANCE_PX)
                .expect("default acquisition distance is valid"),
            FacePixelDistance::try_new(DEFAULT_FACE_ASSOCIATION_DISTANCE_PX)
                .expect("default association distance is valid"),
            ConsecutiveFaceResults::try_new(DEFAULT_FACE_ACQUISITION_RESULTS)
                .expect("default acquisition count is valid"),
            ConsecutiveFaceResults::try_new(DEFAULT_FACE_SWITCH_RESULTS)
                .expect("default switch count is valid"),
            CloserFaceWidthRatio::try_new(3, 2).expect("default closer ratio is valid"),
            PositiveUnitAmount::try_from_basis_points(DEFAULT_FACE_SMOOTHING_ALPHA_BASIS_POINTS)
                .expect("default smoothing alpha is valid"),
            FaceCoastingDuration::try_new(
                NonZeroDuration::try_from_nanos(DEFAULT_FACE_COASTING_DURATION_NS)
                    .expect("default coasting duration is nonzero"),
            )
            .expect("default coasting duration is bounded"),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceTrackingConfigError {
    ZeroPixelDistance,
    PixelDistanceTooLarge {
        actual: u32,
        maximum: u32,
    },
    ZeroConsecutiveResults,
    TooManyConsecutiveResults {
        actual: u8,
        maximum: u8,
    },
    ZeroCloserRatioNumerator,
    ZeroCloserRatioDenominator,
    CloserRatioNotGreaterThanOne {
        numerator: u32,
        denominator: u32,
    },
    CloserRatioTooLarge {
        numerator: u32,
        denominator: u32,
        maximum: u16,
    },
    CoastingDurationTooLong {
        actual_ns: u64,
        maximum_ns: u64,
    },
}

impl fmt::Display for FaceTrackingConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid face-tracking configuration: {self:?}")
    }
}

impl core::error::Error for FaceTrackingConfigError {}

/// Relationship between the accepted detector result and its predecessor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FaceResultAdmission {
    ColdStart,
    Consecutive {
        previous: DetectorResultSequence,
        actual: DetectorResultSequence,
    },
    ForwardGap {
        previous: DetectorResultSequence,
        actual: DetectorResultSequence,
        skipped_result_count: NonZeroU64,
    },
}

/// Current-frame evidence for a target which is not yet confirmed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AcquiringFaceTarget {
    frame_id: FrameId,
    freshness: FreshnessWindow,
    detection: FaceDetection,
    consecutive_results: NonZeroU8,
    required_results: NonZeroU8,
}

impl AcquiringFaceTarget {
    pub const fn frame_id(self) -> FrameId {
        self.frame_id
    }

    pub const fn freshness(self) -> FreshnessWindow {
        self.freshness
    }

    pub const fn detection(self) -> FaceDetection {
        self.detection
    }

    pub const fn consecutive_results(self) -> NonZeroU8 {
        self.consecutive_results
    }

    pub const fn required_results(self) -> NonZeroU8 {
        self.required_results
    }
}

/// A tracked face associated across frames.
///
/// `center` is the fixed-point EMA result. `detection`, `frame_id`, and
/// `freshness` retain the exact current detector-result provenance. The opaque
/// detector weight is accessible only through `detection`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TrackedFaceObservation {
    frame_id: FrameId,
    freshness: FreshnessWindow,
    track_id: PersonTrackId,
    center: ImagePoint,
    detection: FaceDetection,
}

impl TrackedFaceObservation {
    pub const fn frame_id(self) -> FrameId {
        self.frame_id
    }

    pub const fn freshness(self) -> FreshnessWindow {
        self.freshness
    }

    pub const fn track_id(self) -> PersonTrackId {
        self.track_id
    }

    pub const fn center(self) -> ImagePoint {
        self.center
    }

    pub const fn detection(self) -> FaceDetection {
        self.detection
    }
}

/// Association memory retained while no current-frame face matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoastingFaceTarget {
    last_observation: TrackedFaceObservation,
    evaluated_frame_id: FrameId,
    loss_deadline: Deadline,
}

impl CoastingFaceTarget {
    pub const fn last_observation(self) -> TrackedFaceObservation {
        self.last_observation
    }

    pub const fn evaluated_frame_id(self) -> FrameId {
        self.evaluated_frame_id
    }

    pub const fn loss_deadline(self) -> Deadline {
        self.loss_deadline
    }
}

/// One track whose grace interval expired without continuous capture evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LostFaceTarget {
    last_observation: TrackedFaceObservation,
    evaluated_frame_id: FrameId,
    loss_deadline: Deadline,
}

impl LostFaceTarget {
    pub const fn last_observation(self) -> TrackedFaceObservation {
        self.last_observation
    }

    pub const fn evaluated_frame_id(self) -> FrameId {
        self.evaluated_frame_id
    }

    pub const fn loss_deadline(self) -> Deadline {
        self.loss_deadline
    }
}

/// Exact current-frame evidence for a persistent closer-target switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SwitchedFaceTarget {
    previous_track_id: PersonTrackId,
    observation: TrackedFaceObservation,
}

impl SwitchedFaceTarget {
    pub const fn previous_track_id(self) -> PersonTrackId {
        self.previous_track_id
    }

    pub const fn observation(self) -> TrackedFaceObservation {
        self.observation
    }
}

/// Explicit face-target state produced by one accepted detector result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FaceTargetState {
    NoTarget,
    Acquiring(AcquiringFaceTarget),
    Tracked(TrackedFaceObservation),
    Coasting(CoastingFaceTarget),
    Lost(LostFaceTarget),
    Switched(SwitchedFaceTarget),
}

/// State result paired with exact detector/camera provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceTrackingUpdate {
    observation: RgbObservation,
    detector_result_sequence: DetectorResultSequence,
    detector_truncated_count: u32,
    state: FaceTargetState,
    admission: FaceResultAdmission,
}

impl FaceTrackingUpdate {
    pub const fn observation(self) -> RgbObservation {
        self.observation
    }

    pub const fn detector_result_sequence(self) -> DetectorResultSequence {
        self.detector_result_sequence
    }

    pub const fn detector_truncated_count(self) -> u32 {
        self.detector_truncated_count
    }

    pub const fn state(self) -> FaceTargetState {
        self.state
    }

    pub const fn admission(self) -> FaceResultAdmission {
        self.admission
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceTrackingError {
    FrameFromFuture {
        observed_at_ns: u64,
        now_ns: u64,
    },
    StaleFrame {
        deadline_ns: u64,
        now_ns: u64,
    },
    HostClockRegressed {
        previous_ns: u64,
        actual_ns: u64,
    },
    StreamEpochChanged {
        expected: StreamEpochId,
        actual: StreamEpochId,
    },
    DuplicateCameraFrame {
        sequence: u64,
    },
    OutOfOrderCameraFrame {
        previous: u64,
        actual: u64,
    },
    DuplicateDetectorResult {
        sequence: DetectorResultSequence,
    },
    OutOfOrderDetectorResult {
        previous: DetectorResultSequence,
        actual: DetectorResultSequence,
    },
    LayoutChanged {
        expected: ImageLayout,
        actual: ImageLayout,
    },
    ObservationClockNotIncreasing {
        previous_ns: u64,
        actual_ns: u64,
    },
    LossDeadlineOverflow {
        observed_at_ns: u64,
        duration_ns: u64,
    },
    TrackIdExhausted,
}

impl fmt::Display for FaceTrackingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "cannot update face tracker: {self:?}")
    }
}

impl core::error::Error for FaceTrackingError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AcceptedResult {
    frame_id: FrameId,
    detector_result_sequence: DetectorResultSequence,
    observed_at: MonotonicTimestamp,
    processed_at: MonotonicTimestamp,
    layout: ImageLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DetectionAnchor {
    left_subpixels: u64,
    top_subpixels: u64,
    center_x_subpixels: u64,
    center_y_subpixels: u64,
}

impl DetectionAnchor {
    fn from_detection(detection: FaceDetection) -> Self {
        let rectangle = detection.rectangle();
        Self {
            left_subpixels: u64::from(rectangle.left_px()) * PIXEL_SUBUNITS_PER_PIXEL,
            top_subpixels: u64::from(rectangle.top_px()) * PIXEL_SUBUNITS_PER_PIXEL,
            center_x_subpixels: rectangle.center_x_subpixels(),
            center_y_subpixels: rectangle.center_y_subpixels(),
        }
    }

    fn origin_is_near(self, other: Self, distance: FacePixelDistance) -> bool {
        self.left_subpixels.abs_diff(other.left_subpixels) < distance.subpixels()
            && self.top_subpixels.abs_diff(other.top_subpixels) < distance.subpixels()
    }

    fn center_is_near(self, other: Self, distance: FacePixelDistance) -> bool {
        self.center_x_subpixels.abs_diff(other.center_x_subpixels) < distance.subpixels()
            && self.center_y_subpixels.abs_diff(other.center_y_subpixels) < distance.subpixels()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SmoothedTarget {
    center_x_subpixels: u64,
    center_y_subpixels: u64,
    width_subpixels: u64,
}

impl SmoothedTarget {
    fn from_detection(detection: FaceDetection) -> Self {
        let rectangle = detection.rectangle();
        Self {
            center_x_subpixels: rectangle.center_x_subpixels(),
            center_y_subpixels: rectangle.center_y_subpixels(),
            width_subpixels: rectangle.width_subpixels(),
        }
    }

    fn detection_center_is_near(
        self,
        detection: FaceDetection,
        distance: FacePixelDistance,
    ) -> bool {
        let anchor = DetectionAnchor::from_detection(detection);
        self.center_x_subpixels.abs_diff(anchor.center_x_subpixels) < distance.subpixels()
            && self.center_y_subpixels.abs_diff(anchor.center_y_subpixels) < distance.subpixels()
    }

    fn update(self, detection: FaceDetection, alpha: PositiveUnitAmount) -> Self {
        let rectangle = detection.rectangle();
        Self {
            center_x_subpixels: smooth_fixed(
                self.center_x_subpixels,
                rectangle.center_x_subpixels(),
                alpha,
            ),
            center_y_subpixels: smooth_fixed(
                self.center_y_subpixels,
                rectangle.center_y_subpixels(),
                alpha,
            ),
            width_subpixels: smooth_fixed(self.width_subpixels, rectangle.width_subpixels(), alpha),
        }
    }

    fn center(self, layout: ImageLayout) -> ImagePoint {
        ImagePoint::new(
            normalized_subpixel_coordinate(self.center_x_subpixels, layout.width_px()),
            normalized_subpixel_coordinate(self.center_y_subpixels, layout.height_px()),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Challenger {
    anchor: DetectionAnchor,
    consecutive_results: NonZeroU8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TrackMemory {
    track_id: PersonTrackId,
    smoothed: SmoothedTarget,
    last_observation: TrackedFaceObservation,
    loss_deadline: Deadline,
    challenger: Option<Challenger>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrackerState {
    Idle,
    Acquiring {
        anchor: DetectionAnchor,
        consecutive_results: NonZeroU8,
    },
    Established(TrackMemory),
}

/// Allocation-free face-target association state machine.
///
/// Rejected updates are transactional: result admission, association state,
/// and the next track ID all remain unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceTracker {
    config: FaceTrackingConfig,
    state: TrackerState,
    previous_result: Option<AcceptedResult>,
    next_track_id: Option<NonZeroU64>,
}

impl FaceTracker {
    pub const fn new(config: FaceTrackingConfig) -> Self {
        Self {
            config,
            state: TrackerState::Idle,
            previous_result: None,
            next_track_id: NonZeroU64::new(1),
        }
    }

    pub const fn config(&self) -> FaceTrackingConfig {
        self.config
    }

    /// Clears association and result-admission state for a detector-stream restart.
    ///
    /// Previously issued association IDs are deliberately not reused. The next
    /// accepted batch establishes a fresh camera epoch/layout/result-sequence
    /// baseline; this method does not assert that hardware was restarted.
    pub fn reset_stream(&mut self) {
        self.state = TrackerState::Idle;
        self.previous_result = None;
    }

    pub fn update(
        &mut self,
        batch: &FaceDetectionBatch,
        now: MonotonicTimestamp,
    ) -> Result<FaceTrackingUpdate, FaceTrackingError> {
        let mut proposed = *self;
        let update = proposed.update_inner(batch, now)?;
        *self = proposed;
        Ok(update)
    }

    fn update_inner(
        &mut self,
        batch: &FaceDetectionBatch,
        now: MonotonicTimestamp,
    ) -> Result<FaceTrackingUpdate, FaceTrackingError> {
        let admission = self.validate_result(batch, now)?;
        let consecutive_result = matches!(admission, FaceResultAdmission::Consecutive { .. });
        let state = match self.state {
            TrackerState::Idle => self.update_idle(batch)?,
            TrackerState::Acquiring {
                anchor,
                consecutive_results,
            } => self.update_acquiring(batch, anchor, consecutive_results, consecutive_result)?,
            TrackerState::Established(memory) => {
                self.update_established(batch, now, memory, consecutive_result)?
            }
        };
        let observation = batch.observation();
        self.previous_result = Some(AcceptedResult {
            frame_id: observation.frame_id(),
            detector_result_sequence: batch.detector_result_sequence(),
            observed_at: observation.freshness().observed_at(),
            processed_at: now,
            layout: observation.layout(),
        });
        Ok(FaceTrackingUpdate {
            observation,
            detector_result_sequence: batch.detector_result_sequence(),
            detector_truncated_count: batch.detector_truncated_count(),
            state,
            admission,
        })
    }

    fn update_idle(
        &mut self,
        batch: &FaceDetectionBatch,
    ) -> Result<FaceTargetState, FaceTrackingError> {
        let Some(detection) = batch.get(0) else {
            self.state = TrackerState::Idle;
            return Ok(FaceTargetState::NoTarget);
        };
        if self.config.acquisition_results().get() == 1 {
            return self.acquire(batch, detection);
        }
        let consecutive_results = NonZeroU8::new(1).expect("one is nonzero");
        self.state = TrackerState::Acquiring {
            anchor: DetectionAnchor::from_detection(detection),
            consecutive_results,
        };
        Ok(FaceTargetState::Acquiring(AcquiringFaceTarget {
            frame_id: batch.observation().frame_id(),
            freshness: batch.observation().freshness(),
            detection,
            consecutive_results,
            required_results: NonZeroU8::new(self.config.acquisition_results().get())
                .expect("checked acquisition count is nonzero"),
        }))
    }

    fn update_acquiring(
        &mut self,
        batch: &FaceDetectionBatch,
        previous_anchor: DetectionAnchor,
        previous_count: NonZeroU8,
        consecutive_result: bool,
    ) -> Result<FaceTargetState, FaceTrackingError> {
        let Some(detection) = batch.get(0) else {
            self.state = TrackerState::Idle;
            return Ok(FaceTargetState::NoTarget);
        };
        let anchor = DetectionAnchor::from_detection(detection);
        let count = if consecutive_result
            && anchor.origin_is_near(previous_anchor, self.config.acquisition_origin_distance())
        {
            previous_count.get().saturating_add(1)
        } else {
            1
        };
        if count >= self.config.acquisition_results().get() {
            return self.acquire(batch, detection);
        }
        let consecutive_results =
            NonZeroU8::new(count).expect("acquisition result count starts at one");
        self.state = TrackerState::Acquiring {
            anchor,
            consecutive_results,
        };
        Ok(FaceTargetState::Acquiring(AcquiringFaceTarget {
            frame_id: batch.observation().frame_id(),
            freshness: batch.observation().freshness(),
            detection,
            consecutive_results,
            required_results: NonZeroU8::new(self.config.acquisition_results().get())
                .expect("checked acquisition count is nonzero"),
        }))
    }

    fn acquire(
        &mut self,
        batch: &FaceDetectionBatch,
        detection: FaceDetection,
    ) -> Result<FaceTargetState, FaceTrackingError> {
        let track_id = self.allocate_track_id()?;
        let smoothed = SmoothedTarget::from_detection(detection);
        let observation = tracked_observation(batch, detection, smoothed, track_id);
        let memory = self.new_track_memory(observation, smoothed)?;
        self.state = TrackerState::Established(memory);
        Ok(FaceTargetState::Tracked(observation))
    }

    fn update_established(
        &mut self,
        batch: &FaceDetectionBatch,
        now: MonotonicTimestamp,
        mut memory: TrackMemory,
        consecutive_result: bool,
    ) -> Result<FaceTargetState, FaceTrackingError> {
        let observed_at = batch.observation().freshness().observed_at();
        let selected = batch.iter().find(|detection| {
            memory
                .smoothed
                .detection_center_is_near(*detection, self.config.association_center_distance())
        });

        // A fresh result may be processed after the old deadline. Matching
        // capture-time evidence observed before that deadline proves continuity
        // and must be evaluated before processing-time expiry.
        if let Some(selected) = selected
            && memory.loss_deadline.is_alive_at(observed_at)
        {
            let challenger = self.select_challenger(batch, memory, consecutive_result);
            if let Some((detection, evidence)) = challenger
                && evidence.consecutive_results.get() >= self.config.switch_results().get()
            {
                let previous_track_id = memory.track_id;
                let track_id = self.allocate_track_id()?;
                let smoothed = SmoothedTarget::from_detection(detection);
                let observation = tracked_observation(batch, detection, smoothed, track_id);
                let new_memory = self.new_track_memory(observation, smoothed)?;
                self.state = TrackerState::Established(new_memory);
                return Ok(FaceTargetState::Switched(SwitchedFaceTarget {
                    previous_track_id,
                    observation,
                }));
            }

            memory.challenger = challenger.map(|(_, evidence)| evidence);
            memory.smoothed = memory
                .smoothed
                .update(selected, self.config.smoothing_alpha());
            let observation =
                tracked_observation(batch, selected, memory.smoothed, memory.track_id);
            memory.last_observation = observation;
            memory.loss_deadline = loss_deadline(
                observation.freshness().observed_at(),
                self.config.coasting_duration(),
            )?;
            self.state = TrackerState::Established(memory);
            return Ok(FaceTargetState::Tracked(observation));
        }

        if !memory.loss_deadline.is_alive_at(now) {
            self.state = TrackerState::Idle;
            return Ok(FaceTargetState::Lost(LostFaceTarget {
                last_observation: memory.last_observation,
                evaluated_frame_id: batch.observation().frame_id(),
                loss_deadline: memory.loss_deadline,
            }));
        }

        memory.challenger = None;
        self.state = TrackerState::Established(memory);
        Ok(FaceTargetState::Coasting(CoastingFaceTarget {
            last_observation: memory.last_observation,
            evaluated_frame_id: batch.observation().frame_id(),
            loss_deadline: memory.loss_deadline,
        }))
    }

    fn select_challenger(
        &self,
        batch: &FaceDetectionBatch,
        memory: TrackMemory,
        consecutive_result: bool,
    ) -> Option<(FaceDetection, Challenger)> {
        let eligible = |detection: FaceDetection| {
            !memory
                .smoothed
                .detection_center_is_near(detection, self.config.association_center_distance())
                && self
                    .config
                    .closer_width_ratio()
                    .challenger_is_strictly_larger(
                        detection.rectangle().width_subpixels(),
                        memory.smoothed.width_subpixels,
                    )
        };

        if consecutive_result
            && let Some(previous) = memory.challenger
            && let Some(detection) = batch.iter().find(|detection| {
                eligible(*detection)
                    && DetectionAnchor::from_detection(*detection)
                        .center_is_near(previous.anchor, self.config.association_center_distance())
            })
        {
            let count = previous
                .consecutive_results
                .get()
                .saturating_add(1)
                .min(MAX_FACE_CONSECUTIVE_RESULTS);
            return Some((
                detection,
                Challenger {
                    anchor: DetectionAnchor::from_detection(detection),
                    consecutive_results: NonZeroU8::new(count)
                        .expect("challenger count remains nonzero"),
                },
            ));
        }

        batch
            .iter()
            .find(|detection| eligible(*detection))
            .map(|detection| {
                (
                    detection,
                    Challenger {
                        anchor: DetectionAnchor::from_detection(detection),
                        consecutive_results: NonZeroU8::new(1).expect("one is nonzero"),
                    },
                )
            })
    }

    fn new_track_memory(
        &self,
        observation: TrackedFaceObservation,
        smoothed: SmoothedTarget,
    ) -> Result<TrackMemory, FaceTrackingError> {
        Ok(TrackMemory {
            track_id: observation.track_id(),
            smoothed,
            last_observation: observation,
            loss_deadline: loss_deadline(
                observation.freshness().observed_at(),
                self.config.coasting_duration(),
            )?,
            challenger: None,
        })
    }

    fn allocate_track_id(&mut self) -> Result<PersonTrackId, FaceTrackingError> {
        let Some(raw) = self.next_track_id else {
            return Err(FaceTrackingError::TrackIdExhausted);
        };
        self.next_track_id = raw.get().checked_add(1).and_then(NonZeroU64::new);
        PersonTrackId::try_new(raw.get()).map_err(|_| FaceTrackingError::TrackIdExhausted)
    }

    fn validate_result(
        &self,
        batch: &FaceDetectionBatch,
        now: MonotonicTimestamp,
    ) -> Result<FaceResultAdmission, FaceTrackingError> {
        let observation = batch.observation();
        let freshness = observation.freshness();
        let observed_at = freshness.observed_at();
        if observed_at > now {
            return Err(FaceTrackingError::FrameFromFuture {
                observed_at_ns: observed_at.nanos_since_epoch(),
                now_ns: now.nanos_since_epoch(),
            });
        }
        if !freshness.valid_until_exclusive().is_alive_at(now) {
            return Err(FaceTrackingError::StaleFrame {
                deadline_ns: freshness
                    .valid_until_exclusive()
                    .timestamp()
                    .nanos_since_epoch(),
                now_ns: now.nanos_since_epoch(),
            });
        }

        let Some(previous) = self.previous_result else {
            return Ok(FaceResultAdmission::ColdStart);
        };
        if now < previous.processed_at {
            return Err(FaceTrackingError::HostClockRegressed {
                previous_ns: previous.processed_at.nanos_since_epoch(),
                actual_ns: now.nanos_since_epoch(),
            });
        }
        let frame_id = observation.frame_id();
        if frame_id.stream_epoch() != previous.frame_id.stream_epoch() {
            return Err(FaceTrackingError::StreamEpochChanged {
                expected: previous.frame_id.stream_epoch(),
                actual: frame_id.stream_epoch(),
            });
        }
        if frame_id.sequence() == previous.frame_id.sequence() {
            return Err(FaceTrackingError::DuplicateCameraFrame {
                sequence: frame_id.sequence(),
            });
        }
        if frame_id.sequence() < previous.frame_id.sequence() {
            return Err(FaceTrackingError::OutOfOrderCameraFrame {
                previous: previous.frame_id.sequence(),
                actual: frame_id.sequence(),
            });
        }
        let admission = detector_result_admission(
            previous.detector_result_sequence,
            batch.detector_result_sequence(),
        )?;
        if observation.layout() != previous.layout {
            return Err(FaceTrackingError::LayoutChanged {
                expected: previous.layout,
                actual: observation.layout(),
            });
        }
        if observed_at <= previous.observed_at {
            return Err(FaceTrackingError::ObservationClockNotIncreasing {
                previous_ns: previous.observed_at.nanos_since_epoch(),
                actual_ns: observed_at.nanos_since_epoch(),
            });
        }
        Ok(admission)
    }
}

fn tracked_observation(
    batch: &FaceDetectionBatch,
    detection: FaceDetection,
    smoothed: SmoothedTarget,
    track_id: PersonTrackId,
) -> TrackedFaceObservation {
    TrackedFaceObservation {
        frame_id: batch.observation().frame_id(),
        freshness: batch.observation().freshness(),
        track_id,
        center: smoothed.center(batch.observation().layout()),
        detection,
    }
}

fn compare_detection(left: FaceDetection, right: FaceDetection) -> Ordering {
    let left_rectangle = left.rectangle();
    let right_rectangle = right.rectangle();
    right_rectangle
        .width_px()
        .cmp(&left_rectangle.width_px())
        .then_with(|| right_rectangle.height_px().cmp(&left_rectangle.height_px()))
        .then_with(|| {
            right
                .detector_level_weight()
                .cmp(&left.detector_level_weight())
        })
        .then_with(|| left_rectangle.top_px().cmp(&right_rectangle.top_px()))
        .then_with(|| left_rectangle.left_px().cmp(&right_rectangle.left_px()))
        .then_with(|| left.source().cmp(&right.source()))
}

fn normalized_subpixel_coordinate(value: u64, extent_px: u32) -> UnitAmount {
    let denominator = u128::from(extent_px) * u128::from(PIXEL_SUBUNITS_PER_PIXEL);
    let basis_points = (u128::from(value) * NORMALIZED_SCALE + denominator / 2) / denominator;
    UnitAmount::try_from_basis_points(
        u16::try_from(basis_points.min(NORMALIZED_SCALE))
            .expect("clamped normalized face coordinate fits u16"),
    )
    .expect("normalized face coordinate is bounded")
}

fn smooth_fixed(previous: u64, measurement: u64, alpha: PositiveUnitAmount) -> u64 {
    let alpha = u128::from(alpha.basis_points());
    let inverse = NORMALIZED_SCALE - alpha;
    let numerator =
        u128::from(previous) * inverse + u128::from(measurement) * alpha + NORMALIZED_SCALE / 2;
    u64::try_from(numerator / NORMALIZED_SCALE)
        .expect("convex combination of bounded image coordinates fits u64")
}

fn loss_deadline(
    observed_at: MonotonicTimestamp,
    duration: FaceCoastingDuration,
) -> Result<Deadline, FaceTrackingError> {
    Deadline::after(observed_at, duration.get()).map_err(|_| {
        FaceTrackingError::LossDeadlineOverflow {
            observed_at_ns: observed_at.nanos_since_epoch(),
            duration_ns: duration.get().as_nanos(),
        }
    })
}

fn detector_result_admission(
    previous: DetectorResultSequence,
    actual: DetectorResultSequence,
) -> Result<FaceResultAdmission, FaceTrackingError> {
    if actual == previous {
        return Err(FaceTrackingError::DuplicateDetectorResult { sequence: actual });
    }
    let Some(advance) = actual.get().checked_sub(previous.get()) else {
        return Err(FaceTrackingError::OutOfOrderDetectorResult { previous, actual });
    };
    let Some(skipped_result_count) = advance.checked_sub(1).and_then(NonZeroU64::new) else {
        return Ok(FaceResultAdmission::Consecutive { previous, actual });
    };
    Ok(FaceResultAdmission::ForwardGap {
        previous,
        actual,
        skipped_result_count,
    })
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use kiko_expression_core::{
        ChannelOrder, FreshnessWindow, ImageLayout, NonZeroDuration, RgbObservation, StreamEpochId,
    };

    use super::*;

    const FRAME_TTL_NS: u64 = 5_000_000_000;
    const RESULT_PERIOD_NS: u64 = 100_000_000;

    fn epoch(value: u64) -> StreamEpochId {
        StreamEpochId::try_new(value).expect("nonzero test epoch")
    }

    fn layout(width: u32, height: u32) -> ImageLayout {
        ImageLayout::try_new(
            width,
            height,
            width.checked_mul(3).expect("test width packs"),
            ChannelOrder::Bgr,
        )
        .expect("test layout")
    }

    fn observation_with(
        stream_epoch: u64,
        camera_sequence: u64,
        observed_at_ns: u64,
        ttl_ns: u64,
        image_layout: ImageLayout,
    ) -> RgbObservation {
        let observed_at = MonotonicTimestamp::from_nanos_since_epoch(observed_at_ns);
        let freshness = FreshnessWindow::from_ttl(
            observed_at,
            NonZeroDuration::try_from_nanos(ttl_ns).expect("nonzero test ttl"),
        )
        .expect("bounded test freshness");
        RgbObservation::new(
            FrameId::new(epoch(stream_epoch), camera_sequence),
            image_layout,
            freshness,
        )
    }

    fn observation(camera_sequence: u64, observed_at_ns: u64) -> RgbObservation {
        observation_with(
            1,
            camera_sequence,
            observed_at_ns,
            FRAME_TTL_NS,
            layout(640, 400),
        )
    }

    fn face_with(
        image_layout: ImageLayout,
        left: u32,
        top: u32,
        width: u32,
        height: u32,
        weight: f64,
        source: FaceDetectorSource,
    ) -> FaceDetection {
        FaceDetection::try_new(image_layout, left, top, width, height, weight, source)
            .expect("valid test face")
    }

    fn face(left: u32, top: u32, width: u32, height: u32) -> FaceDetection {
        face_with(
            layout(640, 400),
            left,
            top,
            width,
            height,
            4.25,
            FaceDetectorSource::Frontal,
        )
    }

    fn batch_with(
        camera_sequence: u64,
        result_sequence: u64,
        observed_at_ns: u64,
        truncated: u32,
        detections: &[FaceDetection],
    ) -> FaceDetectionBatch {
        FaceDetectionBatch::try_new(
            observation(camera_sequence, observed_at_ns),
            DetectorResultSequence::new(result_sequence),
            truncated,
            detections,
        )
        .expect("valid test batch")
    }

    fn batch(
        camera_sequence: u64,
        result_sequence: u64,
        detections: &[FaceDetection],
    ) -> FaceDetectionBatch {
        batch_with(
            camera_sequence,
            result_sequence,
            result_sequence * RESULT_PERIOD_NS,
            0,
            detections,
        )
    }

    fn update(
        tracker: &mut FaceTracker,
        camera_sequence: u64,
        result_sequence: u64,
        detections: &[FaceDetection],
    ) -> FaceTrackingUpdate {
        let observed_at_ns = result_sequence * RESULT_PERIOD_NS;
        tracker
            .update(
                &batch(camera_sequence, result_sequence, detections),
                MonotonicTimestamp::from_nanos_since_epoch(observed_at_ns),
            )
            .expect("valid tracker update")
    }

    fn tracked(update: FaceTrackingUpdate) -> TrackedFaceObservation {
        match update.state() {
            FaceTargetState::Tracked(observation) => observation,
            other => panic!("expected tracked observation, got {other:?}"),
        }
    }

    fn acquire_default(
        tracker: &mut FaceTracker,
        detection: FaceDetection,
    ) -> TrackedFaceObservation {
        assert!(matches!(
            update(tracker, 10, 1, &[detection]).state(),
            FaceTargetState::Acquiring(_)
        ));
        tracked(update(tracker, 20, 2, &[detection]))
    }

    #[test]
    fn defaults_are_the_documented_hardened_fable_derivative() {
        let config = FaceTrackingConfig::default();
        assert_eq!(
            config.acquisition_origin_distance().get(),
            DEFAULT_FACE_ACQUISITION_DISTANCE_PX
        );
        assert_eq!(
            config.association_center_distance().get(),
            DEFAULT_FACE_ASSOCIATION_DISTANCE_PX
        );
        assert_eq!(
            config.acquisition_results().get(),
            DEFAULT_FACE_ACQUISITION_RESULTS
        );
        assert_eq!(config.switch_results().get(), DEFAULT_FACE_SWITCH_RESULTS);
        assert_eq!(config.closer_width_ratio().numerator(), 3);
        assert_eq!(config.closer_width_ratio().denominator(), 2);
        assert_eq!(
            config.smoothing_alpha().basis_points(),
            DEFAULT_FACE_SMOOTHING_ALPHA_BASIS_POINTS
        );
        assert_eq!(
            config.coasting_duration().get().as_nanos(),
            DEFAULT_FACE_COASTING_DURATION_NS
        );
    }

    #[test]
    fn detector_level_weight_is_opaque_finite_rank_not_confidence() {
        let image_layout = layout(640, 400);
        let negative = face_with(
            image_layout,
            10,
            10,
            80,
            80,
            -27.5,
            FaceDetectorSource::Profile,
        );
        let above_one = face_with(
            image_layout,
            100,
            10,
            80,
            80,
            9.75,
            FaceDetectorSource::Frontal,
        );
        assert_eq!(negative.detector_level_weight().as_f64(), -27.5);
        assert_eq!(above_one.detector_level_weight().as_f64(), 9.75);
        let batch = FaceDetectionBatch::try_new(
            observation(1, 1),
            DetectorResultSequence::new(1),
            0,
            &[negative, above_one],
        )
        .unwrap();
        assert_eq!(batch.get(0), Some(above_one));
        assert_eq!(
            FaceDetection::try_new(
                image_layout,
                0,
                0,
                1,
                1,
                f64::NAN,
                FaceDetectorSource::Frontal
            ),
            Err(FaceDetectionError::NonFiniteDetectorLevelWeight)
        );
    }

    #[test]
    fn otherwise_equal_detections_use_the_oak_row_major_tie_break() {
        let image_layout = layout(640, 400);
        let earlier_row = face_with(
            image_layout,
            500,
            10,
            80,
            80,
            3.0,
            FaceDetectorSource::Frontal,
        );
        let earlier_column = face_with(
            image_layout,
            10,
            20,
            80,
            80,
            3.0,
            FaceDetectorSource::Frontal,
        );
        let batch = FaceDetectionBatch::try_new(
            observation(1, 1),
            DetectorResultSequence::new(1),
            0,
            &[earlier_column, earlier_row],
        )
        .unwrap();
        assert_eq!(batch.get(0), Some(earlier_row));
    }

    #[test]
    fn integer_rectangle_boundary_is_exact_and_checked_once() {
        let image_layout = layout(640, 400);
        let full = FaceRectangle::try_from_origin_size(image_layout, 0, 0, 640, 400).unwrap();
        assert_eq!(full.left_px(), 0);
        assert_eq!(full.right_exclusive_px(), 640);
        assert_eq!(full.bottom_exclusive_px(), 400);
        assert_eq!(full.area_px(), 256_000);
        assert_eq!(
            FaceRectangle::try_from_origin_size(image_layout, 0, 0, 0, 1),
            Err(FaceDetectionError::ZeroWidth)
        );
        assert_eq!(
            FaceRectangle::try_from_origin_size(image_layout, u32::MAX, 0, 2, 1),
            Err(FaceDetectionError::RectangleArithmeticOverflow)
        );
        assert!(matches!(
            FaceRectangle::try_from_origin_size(image_layout, 639, 399, 2, 1),
            Err(FaceDetectionError::RectangleOutsideFrame { .. })
        ));
    }

    #[test]
    fn batch_preserves_detector_truncation_and_never_applies_a_second_cap() {
        let image_layout = layout(640, 400);
        let mut retained = Vec::new();
        for index in 0..MAX_FACE_DETECTIONS {
            retained.push(face_with(
                image_layout,
                u32::try_from(index * 20).unwrap(),
                0,
                u32::try_from(index + 1).unwrap(),
                20,
                -(index as f64),
                FaceDetectorSource::Frontal,
            ));
        }
        let batch = FaceDetectionBatch::try_new(
            observation(1, 1),
            DetectorResultSequence::new(7),
            31,
            &retained,
        )
        .unwrap();
        assert_eq!(batch.retained_count(), MAX_FACE_DETECTIONS);
        assert_eq!(batch.detector_truncated_count(), 31);
        assert_eq!(
            batch.get(0).unwrap().rectangle().width_px(),
            u32::try_from(MAX_FACE_DETECTIONS).unwrap()
        );

        retained.push(face_with(
            image_layout,
            400,
            0,
            200,
            20,
            100.0,
            FaceDetectorSource::Profile,
        ));
        assert_eq!(
            FaceDetectionBatch::try_new(
                observation(1, 1),
                DetectorResultSequence::new(7),
                30,
                &retained
            ),
            Err(FaceDetectionBatchError::RetainedCountExceedsCapacity {
                actual: MAX_FACE_DETECTIONS + 1,
                maximum: MAX_FACE_DETECTIONS,
            })
        );
    }

    #[test]
    fn widest_eligible_face_cannot_be_starved_by_input_order_or_rank() {
        let narrow_high_rank = face_with(
            layout(640, 400),
            10,
            10,
            40,
            40,
            10_000.0,
            FaceDetectorSource::Frontal,
        );
        let widest_negative_rank = face_with(
            layout(640, 400),
            100,
            10,
            100,
            40,
            -10_000.0,
            FaceDetectorSource::Profile,
        );
        let batch = FaceDetectionBatch::try_new(
            observation(1, 1),
            DetectorResultSequence::new(1),
            0,
            &[narrow_high_rank, widest_negative_rank],
        )
        .unwrap();
        assert_eq!(batch.get(0), Some(widest_negative_rank));

        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let first = tracker
            .update(&batch, MonotonicTimestamp::from_nanos_since_epoch(1))
            .unwrap();
        let FaceTargetState::Acquiring(acquiring) = first.state() else {
            panic!("first result starts acquisition");
        };
        assert_eq!(acquiring.detection(), widest_negative_rank);
    }

    #[test]
    fn skipped_camera_frames_do_not_break_consecutive_detector_results() {
        let detection = face(10, 10, 80, 80);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        assert!(matches!(
            update(&mut tracker, 100, 40, &[detection]).state(),
            FaceTargetState::Acquiring(_)
        ));
        let second = update(&mut tracker, 127, 41, &[detection]);
        assert!(matches!(
            second.admission(),
            FaceResultAdmission::Consecutive {
                previous,
                actual
            } if previous.get() == 40 && actual.get() == 41
        ));
        assert!(matches!(second.state(), FaceTargetState::Tracked(_)));
    }

    #[test]
    fn skipped_detector_result_resets_acquisition_streak() {
        let detection = face(10, 10, 80, 80);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        update(&mut tracker, 10, 1, &[detection]);
        let gap = update(&mut tracker, 20, 3, &[detection]);
        assert!(matches!(
            gap.admission(),
            FaceResultAdmission::ForwardGap {
                skipped_result_count,
                ..
            } if skipped_result_count.get() == 1
        ));
        let FaceTargetState::Acquiring(acquiring) = gap.state() else {
            panic!("result gap restarts acquisition evidence");
        };
        assert_eq!(acquiring.consecutive_results().get(), 1);
        assert!(matches!(
            update(&mut tracker, 30, 4, &[detection]).state(),
            FaceTargetState::Tracked(_)
        ));
    }

    #[test]
    fn current_capture_match_is_evaluated_before_processing_time_expiry() {
        let detection = face(10, 10, 80, 80);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        acquire_default(&mut tracker, detection);
        let old_deadline = 2 * RESULT_PERIOD_NS + DEFAULT_FACE_COASTING_DURATION_NS;
        let observed_at = old_deadline - 1;
        let delayed = batch_with(30, 3, observed_at, 0, &[detection]);
        let processed_at = old_deadline + 500_000_000;
        let update = tracker
            .update(
                &delayed,
                MonotonicTimestamp::from_nanos_since_epoch(processed_at),
            )
            .unwrap();
        let tracked = tracked(update);
        assert_eq!(tracked.frame_id().sequence(), 30);
        assert_eq!(
            tracked.freshness().observed_at().nanos_since_epoch(),
            observed_at
        );
    }

    #[test]
    fn output_preserves_frame_source_rank_and_detector_truncation_provenance() {
        let detection = face_with(
            layout(640, 400),
            10,
            20,
            90,
            70,
            -3.125,
            FaceDetectorSource::MirroredProfile,
        );
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        update(&mut tracker, 400, 1, &[detection]);
        let second = batch_with(455, 2, 200_000_000, 9, &[detection]);
        let update = tracker
            .update(
                &second,
                MonotonicTimestamp::from_nanos_since_epoch(200_000_000),
            )
            .unwrap();
        let tracked = tracked(update);
        assert_eq!(update.observation(), second.observation());
        assert_eq!(update.detector_result_sequence().get(), 2);
        assert_eq!(update.detector_truncated_count(), 9);
        assert_eq!(tracked.frame_id().sequence(), 455);
        assert_eq!(
            tracked.detection().source(),
            FaceDetectorSource::MirroredProfile
        );
        assert_eq!(tracked.detection().detector_level_weight().as_f64(), -3.125);
    }

    #[test]
    fn fixed_point_smoothing_is_point_four_five_and_bounded() {
        let image_layout = layout(1_000, 100);
        let first = face_with(
            image_layout,
            0,
            0,
            100,
            20,
            0.0,
            FaceDetectorSource::Frontal,
        );
        let moved = face_with(
            image_layout,
            100,
            0,
            100,
            20,
            0.0,
            FaceDetectorSource::Frontal,
        );
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        for sequence in [1, 2] {
            let rgb = observation_with(
                1,
                sequence,
                sequence * RESULT_PERIOD_NS,
                FRAME_TTL_NS,
                image_layout,
            );
            let batch = FaceDetectionBatch::try_new(
                rgb,
                DetectorResultSequence::new(sequence),
                0,
                &[first],
            )
            .unwrap();
            let result = tracker
                .update(
                    &batch,
                    MonotonicTimestamp::from_nanos_since_epoch(sequence * RESULT_PERIOD_NS),
                )
                .unwrap();
            if sequence == 2 {
                assert_eq!(tracked(result).center().x_right().basis_points(), 500);
            }
        }
        let rgb = observation_with(1, 3, 3 * RESULT_PERIOD_NS, FRAME_TTL_NS, image_layout);
        let batch =
            FaceDetectionBatch::try_new(rgb, DetectorResultSequence::new(3), 0, &[moved]).unwrap();
        assert_eq!(
            tracked(
                tracker
                    .update(
                        &batch,
                        MonotonicTimestamp::from_nanos_since_epoch(3 * RESULT_PERIOD_NS)
                    )
                    .unwrap()
            )
            .center()
            .x_right()
            .basis_points(),
            950
        );

        for previous in (0_u64..=10_000_000).step_by(91_337) {
            for measurement in (0_u64..=10_000_000).step_by(77_777) {
                for alpha in (1_u16..=10_000).step_by(997) {
                    let value = smooth_fixed(
                        previous,
                        measurement,
                        PositiveUnitAmount::try_from_basis_points(alpha).unwrap(),
                    );
                    assert!(
                        (previous.min(measurement)..=previous.max(measurement)).contains(&value)
                    );
                }
            }
        }
    }

    #[test]
    fn strict_association_and_closer_switch_match_policy_boundaries() {
        let current = face(0, 0, 100, 50);
        let boundary = face(140, 0, 100, 50);
        let closer = face(300, 0, 151, 50);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let original = acquire_default(&mut tracker, current);

        let FaceTargetState::Coasting(coasting) = update(&mut tracker, 30, 3, &[boundary]).state()
        else {
            panic!("exact 140-pixel center displacement is not associated");
        };
        assert_eq!(coasting.last_observation(), original);

        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let original = acquire_default(&mut tracker, current);
        for result_sequence in 3..7 {
            let result = update(
                &mut tracker,
                result_sequence * 10,
                result_sequence,
                &[closer, current],
            );
            assert!(matches!(result.state(), FaceTargetState::Tracked(_)));
        }
        let switched = update(&mut tracker, 70, 7, &[closer, current]);
        let FaceTargetState::Switched(switched) = switched.state() else {
            panic!("fifth consecutive closer result switches");
        };
        assert_eq!(switched.previous_track_id(), original.track_id());
        assert_eq!(switched.observation().track_id().get(), 2);
    }

    #[test]
    fn coasting_never_relabels_old_frame_evidence() {
        let detection = face(10, 10, 80, 80);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let acquired = acquire_default(&mut tracker, detection);
        let FaceTargetState::Coasting(coasting) = update(&mut tracker, 55, 3, &[]).state() else {
            panic!("missing result coasts");
        };
        assert_eq!(coasting.last_observation(), acquired);
        assert_eq!(coasting.last_observation().frame_id().sequence(), 20);
        assert_eq!(coasting.evaluated_frame_id().sequence(), 55);
    }

    #[test]
    fn deadline_is_exclusive_when_no_continuous_capture_match_exists() {
        let detection = face(10, 10, 80, 80);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let acquired = acquire_default(&mut tracker, detection);
        let deadline = 2 * RESULT_PERIOD_NS + DEFAULT_FACE_COASTING_DURATION_NS;
        let at_deadline = batch_with(30, 3, deadline, 0, &[detection]);
        let result = tracker
            .update(
                &at_deadline,
                MonotonicTimestamp::from_nanos_since_epoch(deadline),
            )
            .unwrap();
        let FaceTargetState::Lost(lost) = result.state() else {
            panic!("capture at exclusive deadline does not prove continuity");
        };
        assert_eq!(lost.last_observation(), acquired);
        assert_eq!(lost.evaluated_frame_id().sequence(), 30);
    }

    #[test]
    fn result_and_frame_rejections_are_transactional() {
        let detection = face(10, 10, 30, 30);
        let first = batch_with(10, 7, 100, 0, &[detection]);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        tracker
            .update(&first, MonotonicTimestamp::from_nanos_since_epoch(100))
            .unwrap();

        let duplicate_result = batch_with(11, 7, 101, 0, &[detection]);
        let before = tracker;
        assert_eq!(
            tracker.update(
                &duplicate_result,
                MonotonicTimestamp::from_nanos_since_epoch(101)
            ),
            Err(FaceTrackingError::DuplicateDetectorResult {
                sequence: DetectorResultSequence::new(7)
            })
        );
        assert_eq!(tracker, before);

        let duplicate_frame = batch_with(10, 8, 101, 0, &[detection]);
        let before = tracker;
        assert_eq!(
            tracker.update(
                &duplicate_frame,
                MonotonicTimestamp::from_nanos_since_epoch(101)
            ),
            Err(FaceTrackingError::DuplicateCameraFrame { sequence: 10 })
        );
        assert_eq!(tracker, before);

        let stale_observation = observation_with(1, 11, 101, 1, layout(640, 400));
        let stale =
            FaceDetectionBatch::try_new(stale_observation, DetectorResultSequence::new(8), 0, &[])
                .unwrap();
        let before = tracker;
        assert!(matches!(
            tracker.update(&stale, MonotonicTimestamp::from_nanos_since_epoch(102)),
            Err(FaceTrackingError::StaleFrame { .. })
        ));
        assert_eq!(tracker, before);
    }

    #[test]
    fn reset_stream_preserves_global_track_id_uniqueness() {
        let detection = face(10, 10, 40, 40);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        assert_eq!(acquire_default(&mut tracker, detection).track_id().get(), 1);
        tracker.reset_stream();

        let image_layout = layout(640, 400);
        let restarted_detection = face_with(
            image_layout,
            10,
            10,
            40,
            40,
            0.0,
            FaceDetectorSource::Frontal,
        );
        for result_sequence in [0, 1] {
            let rgb = observation_with(
                2,
                result_sequence,
                result_sequence + 1,
                FRAME_TTL_NS,
                image_layout,
            );
            let batch = FaceDetectionBatch::try_new(
                rgb,
                DetectorResultSequence::new(result_sequence),
                0,
                &[restarted_detection],
            )
            .unwrap();
            let update = tracker
                .update(
                    &batch,
                    MonotonicTimestamp::from_nanos_since_epoch(result_sequence + 1),
                )
                .unwrap();
            if result_sequence == 1 {
                assert_eq!(tracked(update).track_id().get(), 2);
            }
        }
    }

    #[test]
    fn config_and_loss_overflow_are_explicit() {
        assert_eq!(
            ConsecutiveFaceResults::try_new(0),
            Err(FaceTrackingConfigError::ZeroConsecutiveResults)
        );
        assert!(matches!(
            FacePixelDistance::try_new(MAX_FACE_PIXEL_DISTANCE_PX + 1),
            Err(FaceTrackingConfigError::PixelDistanceTooLarge { .. })
        ));

        let detection = face(0, 0, 10, 10);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let first_time = u64::MAX - 3_000_000_000;
        let first = FaceDetectionBatch::try_new(
            observation_with(1, 1, first_time, 1_000_000_000, layout(640, 400)),
            DetectorResultSequence::new(1),
            0,
            &[detection],
        )
        .unwrap();
        tracker
            .update(
                &first,
                MonotonicTimestamp::from_nanos_since_epoch(first_time),
            )
            .unwrap();
        let second_time = u64::MAX - 1_000_000_000;
        let second = FaceDetectionBatch::try_new(
            observation_with(1, 2, second_time, 500_000_000, layout(640, 400)),
            DetectorResultSequence::new(2),
            0,
            &[detection],
        )
        .unwrap();
        let before = tracker;
        assert!(matches!(
            tracker.update(
                &second,
                MonotonicTimestamp::from_nanos_since_epoch(second_time)
            ),
            Err(FaceTrackingError::LossDeadlineOverflow { .. })
        ));
        assert_eq!(tracker, before);
    }
}
