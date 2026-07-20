//! Deterministic, allocation-free scene motion from sampled RGB luminance.

use core::{fmt, mem, num::NonZeroU16};

use kiko_expression_core::{
    ChannelOrder, FrameId, ImageLayout, ImagePoint, MonotonicTimestamp, PositiveUnitAmount,
    RgbFrameView, SceneMotion, SceneObservation, StreamEpochId, UnitAmount,
};

/// Maximum number of luminance samples retained for adjacent-frame comparison.
pub const MAX_SCENE_SAMPLES: usize = 4_096;
const CORE_UNIT_SCALE: u64 = UnitAmount::ONE.basis_points() as u64;
const CHANNEL_MAX: u64 = 255;
const MAX_COMPENSATED_DELTA: u16 = 510;
const MAX_COMPENSATED_DELTA_TWICE: u64 = MAX_COMPENSATED_DELTA as u64 * 2;
const DELTA_HISTOGRAM_BINS: usize = 511;

/// A fixed cell grid sampled at each cell's integer centre.
///
/// For image extent `L`, cell index `i`, and cell count `N`, the sampled pixel
/// is `floor(((2*i + 1) * L) / (2*N))`. Frames smaller than the configured grid
/// are rejected so every cell maps to a distinct pixel on each axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplingGeometry {
    columns: NonZeroU16,
    rows: NonZeroU16,
    sample_count: u16,
}

impl SamplingGeometry {
    pub fn try_new(columns: u16, rows: u16) -> Result<Self, SamplingGeometryError> {
        let columns = NonZeroU16::new(columns).ok_or(SamplingGeometryError::ZeroColumns)?;
        let rows = NonZeroU16::new(rows).ok_or(SamplingGeometryError::ZeroRows)?;
        let sample_count = u32::from(columns.get())
            .checked_mul(u32::from(rows.get()))
            .ok_or(SamplingGeometryError::TooManySamples {
                requested: u32::MAX,
                maximum: MAX_SCENE_SAMPLES,
            })?;
        if sample_count > u32::try_from(MAX_SCENE_SAMPLES).expect("sample maximum fits u32") {
            return Err(SamplingGeometryError::TooManySamples {
                requested: sample_count,
                maximum: MAX_SCENE_SAMPLES,
            });
        }
        Ok(Self {
            columns,
            rows,
            sample_count: u16::try_from(sample_count).expect("bounded sample count fits u16"),
        })
    }

    pub const fn columns(self) -> u16 {
        self.columns.get()
    }

    pub const fn rows(self) -> u16 {
        self.rows.get()
    }

    pub const fn sample_count(self) -> u16 {
        self.sample_count
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingGeometryError {
    ZeroColumns,
    ZeroRows,
    TooManySamples { requested: u32, maximum: usize },
}

impl fmt::Display for SamplingGeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid RGB sampling geometry: {self:?}")
    }
}

impl core::error::Error for SamplingGeometryError {}

/// Robust-motion detection thresholds in explicit luminance/sample units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MotionThresholds {
    minimum_residual_luma: NonZeroU16,
    minimum_active_fraction: PositiveUnitAmount,
}

impl MotionThresholds {
    /// `minimum_residual_luma` is on the inclusive compensated-delta scale
    /// `1..=510`. `minimum_active_fraction` is the positive fraction of sampled
    /// cells which must cross that residual threshold.
    pub fn try_new(
        minimum_residual_luma: u16,
        minimum_active_fraction: PositiveUnitAmount,
    ) -> Result<Self, SceneMotionConfigError> {
        let minimum_residual_luma = NonZeroU16::new(minimum_residual_luma)
            .ok_or(SceneMotionConfigError::ZeroResidualThreshold)?;
        if minimum_residual_luma.get() > MAX_COMPENSATED_DELTA {
            return Err(SceneMotionConfigError::ResidualThresholdOutOfRange {
                actual: minimum_residual_luma.get(),
                maximum: MAX_COMPENSATED_DELTA,
            });
        }
        Ok(Self {
            minimum_residual_luma,
            minimum_active_fraction,
        })
    }

    pub const fn minimum_residual_luma(self) -> u16 {
        self.minimum_residual_luma.get()
    }

    pub const fn minimum_active_fraction(self) -> PositiveUnitAmount {
        self.minimum_active_fraction
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneMotionConfigError {
    ZeroResidualThreshold,
    ResidualThresholdOutOfRange { actual: u16, maximum: u16 },
}

impl fmt::Display for SceneMotionConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid scene-motion threshold: {self:?}")
    }
}

impl core::error::Error for SceneMotionConfigError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SceneMotionConfig {
    geometry: SamplingGeometry,
    thresholds: MotionThresholds,
}

impl SceneMotionConfig {
    pub const fn new(geometry: SamplingGeometry, thresholds: MotionThresholds) -> Self {
        Self {
            geometry,
            thresholds,
        }
    }

    pub const fn geometry(self) -> SamplingGeometry {
        self.geometry
    }

    pub const fn thresholds(self) -> MotionThresholds {
        self.thresholds
    }
}

/// Explicit first-frame/no-motion/motion semantics.
///
/// `SceneObservation::motion()` is `None` for both `ColdStart` and `NoMotion`;
/// this enum preserves why it is absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SceneAnalysis {
    ColdStart(SceneObservation),
    NoMotion(SceneObservation),
    Motion(SceneObservation),
}

impl SceneAnalysis {
    pub const fn observation(self) -> SceneObservation {
        match self {
            Self::ColdStart(observation)
            | Self::NoMotion(observation)
            | Self::Motion(observation) => observation,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ExtractorState {
    Cold,
    Primed {
        frame_id: FrameId,
        observed_at: MonotonicTimestamp,
        processed_at: MonotonicTimestamp,
        layout: ImageLayout,
    },
}

/// Stateful adjacent-frame extractor with fixed sampled-luminance storage.
///
/// Pixel data remains borrowed. Only two fixed arrays of sampled `u8`
/// luminances are retained; no frame is cloned and no heap allocation occurs.
pub struct SceneMotionExtractor {
    config: SceneMotionConfig,
    state: ExtractorState,
    previous_luma: [u8; MAX_SCENE_SAMPLES],
    current_luma: [u8; MAX_SCENE_SAMPLES],
}

impl SceneMotionExtractor {
    pub const fn new(config: SceneMotionConfig) -> Self {
        Self {
            config,
            state: ExtractorState::Cold,
            previous_luma: [0; MAX_SCENE_SAMPLES],
            current_luma: [0; MAX_SCENE_SAMPLES],
        }
    }

    pub const fn config(&self) -> SceneMotionConfig {
        self.config
    }

    /// Explicitly begin a new stream comparison epoch.
    pub fn reset(&mut self) {
        self.state = ExtractorState::Cold;
    }

    /// Analyze one strictly consecutive, fresh frame.
    ///
    /// Every rejection leaves the previous accepted frame unchanged.
    pub fn analyze(
        &mut self,
        frame: RgbFrameView<'_>,
        now: MonotonicTimestamp,
    ) -> Result<SceneAnalysis, SceneMotionError> {
        let observation = frame.observation();
        let layout = observation.layout();
        self.validate_frame(observation.frame_id(), observation.freshness(), layout, now)?;

        let sample_count = usize::from(self.config.geometry.sample_count());
        let mean_luminance = sample_luminances(
            frame,
            self.config.geometry,
            &mut self.current_luma[..sample_count],
        );
        let scene = match self.state {
            ExtractorState::Cold => SceneAnalysis::ColdStart(SceneObservation::new(
                observation.frame_id(),
                observation.freshness(),
                mean_luminance,
                None,
            )),
            ExtractorState::Primed { .. } => {
                let motion = detect_motion(
                    layout,
                    self.config,
                    &self.previous_luma[..sample_count],
                    &self.current_luma[..sample_count],
                );
                let observation = SceneObservation::new(
                    observation.frame_id(),
                    observation.freshness(),
                    mean_luminance,
                    motion,
                );
                if motion.is_some() {
                    SceneAnalysis::Motion(observation)
                } else {
                    SceneAnalysis::NoMotion(observation)
                }
            }
        };

        mem::swap(&mut self.previous_luma, &mut self.current_luma);
        self.state = ExtractorState::Primed {
            frame_id: observation.frame_id(),
            observed_at: observation.freshness().observed_at(),
            processed_at: now,
            layout,
        };
        Ok(scene)
    }

    fn validate_frame(
        &self,
        frame_id: FrameId,
        freshness: kiko_expression_core::FreshnessWindow,
        layout: ImageLayout,
        now: MonotonicTimestamp,
    ) -> Result<(), SceneMotionError> {
        let geometry = self.config.geometry;
        if layout.width_px() < u32::from(geometry.columns())
            || layout.height_px() < u32::from(geometry.rows())
        {
            return Err(SceneMotionError::FrameSmallerThanSamplingGrid {
                width_px: layout.width_px(),
                height_px: layout.height_px(),
                columns: geometry.columns(),
                rows: geometry.rows(),
            });
        }

        let observed_at = freshness.observed_at();
        if observed_at > now {
            return Err(SceneMotionError::FrameFromFuture {
                observed_at_ns: observed_at.nanos_since_epoch(),
                now_ns: now.nanos_since_epoch(),
            });
        }
        if !freshness.valid_until_exclusive().is_alive_at(now) {
            return Err(SceneMotionError::StaleFrame {
                deadline_ns: freshness
                    .valid_until_exclusive()
                    .timestamp()
                    .nanos_since_epoch(),
                now_ns: now.nanos_since_epoch(),
            });
        }

        let ExtractorState::Primed {
            frame_id: previous_frame,
            observed_at: previous_observed_at,
            processed_at: previous_processed_at,
            layout: previous_layout,
        } = self.state
        else {
            return Ok(());
        };

        if now < previous_processed_at {
            return Err(SceneMotionError::HostClockRegressed {
                previous_ns: previous_processed_at.nanos_since_epoch(),
                actual_ns: now.nanos_since_epoch(),
            });
        }
        if frame_id.stream_epoch() != previous_frame.stream_epoch() {
            return Err(SceneMotionError::StreamEpochChanged {
                expected: previous_frame.stream_epoch(),
                actual: frame_id.stream_epoch(),
            });
        }
        let Some(expected_sequence) = previous_frame.sequence().checked_add(1) else {
            return Err(SceneMotionError::FrameSequenceExhausted {
                previous: previous_frame.sequence(),
            });
        };
        let actual_sequence = frame_id.sequence();
        if actual_sequence == previous_frame.sequence() {
            return Err(SceneMotionError::DuplicateFrame {
                sequence: actual_sequence,
            });
        }
        if actual_sequence < previous_frame.sequence() {
            return Err(SceneMotionError::OutOfOrderFrame {
                previous: previous_frame.sequence(),
                actual: actual_sequence,
            });
        }
        if actual_sequence != expected_sequence {
            return Err(SceneMotionError::FrameGap {
                expected: expected_sequence,
                actual: actual_sequence,
            });
        }
        if layout != previous_layout {
            return Err(SceneMotionError::LayoutChanged {
                expected: previous_layout,
                actual: layout,
            });
        }
        if observed_at <= previous_observed_at {
            return Err(SceneMotionError::ObservationClockNotIncreasing {
                previous_ns: previous_observed_at.nanos_since_epoch(),
                actual_ns: observed_at.nanos_since_epoch(),
            });
        }
        Ok(())
    }
}

fn sample_luminances(
    frame: RgbFrameView<'_>,
    geometry: SamplingGeometry,
    destination: &mut [u8],
) -> UnitAmount {
    let layout = frame.observation().layout();
    let pixels = frame.pixels();
    let mut luminance_sum = 0_u64;
    let mut index = 0_usize;
    for row in 0..geometry.rows() {
        let y = sample_coordinate(layout.height_px(), row, geometry.rows());
        for column in 0..geometry.columns() {
            let x = sample_coordinate(layout.width_px(), column, geometry.columns());
            let luminance = pixel_luminance(layout, pixels, x, y);
            destination[index] = luminance;
            luminance_sum += u64::from(luminance);
            index += 1;
        }
    }
    debug_assert_eq!(index, destination.len());
    unit_from_ratio(
        luminance_sum,
        CHANNEL_MAX * u64::try_from(destination.len()).expect("sample count fits u64"),
    )
}

fn sample_coordinate(extent: u32, index: u16, count: u16) -> u32 {
    let numerator = (u64::from(index) * 2 + 1) * u64::from(extent);
    let denominator = u64::from(count) * 2;
    u32::try_from(numerator / denominator).expect("sample coordinate is inside u32 image extent")
}

fn pixel_luminance(layout: ImageLayout, pixels: &[u8], x: u32, y: u32) -> u8 {
    let row = u64::from(y) * u64::from(layout.stride_bytes());
    let column = u64::from(x) * 3;
    let offset = usize::try_from(row + column).expect("checked image layout fits address space");
    let (red, green, blue) = match layout.channel_order() {
        ChannelOrder::Rgb => (pixels[offset], pixels[offset + 1], pixels[offset + 2]),
        ChannelOrder::Bgr => (pixels[offset + 2], pixels[offset + 1], pixels[offset]),
    };
    let weighted =
        77_u32 * u32::from(red) + 150_u32 * u32::from(green) + 29_u32 * u32::from(blue) + 128;
    u8::try_from(weighted >> 8).expect("BT.601 integer luminance remains an eight-bit value")
}

fn detect_motion(
    layout: ImageLayout,
    config: SceneMotionConfig,
    previous: &[u8],
    current: &[u8],
) -> Option<SceneMotion> {
    let mut histogram = [0_u16; DELTA_HISTOGRAM_BINS];
    for (&previous, &current) in previous.iter().zip(current) {
        let delta = i16::from(current) - i16::from(previous);
        let bin = usize::try_from(i32::from(delta) + 255)
            .expect("signed luminance delta maps into the histogram");
        histogram[bin] += 1;
    }
    // Retain an even-sized sample set's half-luma median exactly. Rounding a
    // median such as +0.5 toward either integer would make positive and
    // negative residuals cross an integer threshold asymmetrically.
    let exposure_shift_twice = median_delta_twice(&histogram, previous.len());

    let threshold_twice = u32::from(config.thresholds.minimum_residual_luma()) * 2;
    let minimum_active =
        minimum_active_samples(previous.len(), config.thresholds.minimum_active_fraction());
    let mut active_count = 0_usize;
    let mut residual_sum = 0_u64;
    let mut weighted_x = 0_u128;
    let mut weighted_y = 0_u128;

    for (index, (&previous, &current)) in previous.iter().zip(current).enumerate() {
        let delta_twice = 2 * (i32::from(current) - i32::from(previous));
        let residual = (delta_twice - i32::from(exposure_shift_twice)).unsigned_abs();
        if residual < threshold_twice {
            continue;
        }
        let residual = u64::from(residual);
        active_count += 1;
        residual_sum += residual;
        let column = u16::try_from(index % usize::from(config.geometry.columns()))
            .expect("grid column fits u16");
        let row = u16::try_from(index / usize::from(config.geometry.columns()))
            .expect("grid row fits u16");
        let x = sample_coordinate(layout.width_px(), column, config.geometry.columns());
        let y = sample_coordinate(layout.height_px(), row, config.geometry.rows());
        weighted_x += u128::from(x) * u128::from(residual);
        weighted_y += u128::from(y) * u128::from(residual);
    }

    if active_count < minimum_active || residual_sum == 0 {
        return None;
    }

    let strength_basis_points = rounded_ratio_u128(
        u128::from(residual_sum) * u128::from(CORE_UNIT_SCALE),
        u128::try_from(previous.len()).expect("sample count fits u128")
            * u128::from(MAX_COMPENSATED_DELTA_TWICE),
    )
    .clamp(1, CORE_UNIT_SCALE);
    let strength = PositiveUnitAmount::try_from_basis_points(
        u16::try_from(strength_basis_points).expect("clamped strength fits u16"),
    )
    .expect("motion strength is positive and bounded");
    let center = ImagePoint::new(
        normalized_centroid(weighted_x, residual_sum, layout.width_px()),
        normalized_centroid(weighted_y, residual_sum, layout.height_px()),
    );
    Some(SceneMotion::new(strength, center))
}

/// Twice the exact median signed luminance delta.
///
/// The sum of the two middle order statistics represents both odd and even
/// medians without a floating-point value or directional integer rounding.
fn median_delta_twice(histogram: &[u16; DELTA_HISTOGRAM_BINS], sample_count: usize) -> i16 {
    let lower_rank = (sample_count - 1) / 2;
    let upper_rank = sample_count / 2;
    let mut seen = 0_usize;
    let mut lower = None;
    for (bin, &count) in histogram.iter().enumerate() {
        let end = seen + usize::from(count);
        if lower.is_none() && lower_rank < end {
            lower = Some(bin);
        }
        if upper_rank < end {
            let lower = lower.expect("lower median rank precedes upper rank");
            let lower_delta = i32::try_from(lower).expect("histogram index fits i32") - 255;
            let upper_delta = i32::try_from(bin).expect("histogram index fits i32") - 255;
            return i16::try_from(lower_delta + upper_delta)
                .expect("twice the median luminance delta fits i16");
        }
        seen = end;
    }
    unreachable!("histogram contains exactly the configured sample count")
}

fn minimum_active_samples(sample_count: usize, fraction: PositiveUnitAmount) -> usize {
    let numerator = u64::try_from(sample_count).expect("sample count fits u64")
        * u64::from(fraction.basis_points());
    let rounded_up = numerator.div_ceil(CORE_UNIT_SCALE);
    usize::try_from(rounded_up.max(1)).expect("active sample count fits usize")
}

fn normalized_centroid(weighted: u128, weight_sum: u64, extent: u32) -> UnitAmount {
    if extent == 1 {
        return UnitAmount::try_from_basis_points(5_000).expect("image centre is in range");
    }
    let denominator = u128::from(weight_sum) * u128::from(extent - 1);
    let basis_points = rounded_ratio_u128(weighted * u128::from(CORE_UNIT_SCALE), denominator)
        .min(CORE_UNIT_SCALE);
    UnitAmount::try_from_basis_points(
        u16::try_from(basis_points).expect("normalized centroid fits u16"),
    )
    .expect("normalized centroid is bounded")
}

fn unit_from_ratio(numerator: u64, denominator: u64) -> UnitAmount {
    let basis_points = (numerator * CORE_UNIT_SCALE + denominator / 2) / denominator;
    UnitAmount::try_from_basis_points(
        u16::try_from(basis_points).expect("normalized ratio fits u16"),
    )
    .expect("normalized ratio is bounded")
}

fn rounded_ratio_u128(numerator: u128, denominator: u128) -> u64 {
    u64::try_from((numerator + denominator / 2) / denominator).expect("normalized ratio fits u64")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneMotionError {
    FrameSmallerThanSamplingGrid {
        width_px: u32,
        height_px: u32,
        columns: u16,
        rows: u16,
    },
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
    FrameSequenceExhausted {
        previous: u64,
    },
    DuplicateFrame {
        sequence: u64,
    },
    OutOfOrderFrame {
        previous: u64,
        actual: u64,
    },
    FrameGap {
        expected: u64,
        actual: u64,
    },
    LayoutChanged {
        expected: ImageLayout,
        actual: ImageLayout,
    },
    ObservationClockNotIncreasing {
        previous_ns: u64,
        actual_ns: u64,
    },
}

impl fmt::Display for SceneMotionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "cannot extract scene motion: {self:?}")
    }
}

impl core::error::Error for SceneMotionError {}

#[cfg(test)]
mod tests {
    extern crate std;

    use kiko_expression_core::{
        ChannelOrder, FreshnessWindow, ImageLayout, NonZeroDuration, RgbObservation,
    };

    use super::*;

    fn config(columns: u16, rows: u16, threshold: u16, active_bp: u16) -> SceneMotionConfig {
        SceneMotionConfig::new(
            SamplingGeometry::try_new(columns, rows).expect("geometry"),
            MotionThresholds::try_new(
                threshold,
                PositiveUnitAmount::try_from_basis_points(active_bp).expect("active fraction"),
            )
            .expect("thresholds"),
        )
    }

    fn freshness(observed_ns: u64) -> FreshnessWindow {
        FreshnessWindow::from_ttl(
            MonotonicTimestamp::from_nanos_since_epoch(observed_ns),
            NonZeroDuration::try_from_nanos(100).expect("ttl"),
        )
        .expect("freshness")
    }

    fn view<'a>(
        pixels: &'a [u8],
        order: ChannelOrder,
        stride: u32,
        sequence: u64,
        observed_ns: u64,
    ) -> RgbFrameView<'a> {
        let layout = ImageLayout::try_new(3, 3, stride, order).expect("layout");
        let observation = RgbObservation::new(
            FrameId::new(StreamEpochId::try_new(7).expect("epoch"), sequence),
            layout,
            freshness(observed_ns),
        );
        RgbFrameView::try_new(observation, pixels).expect("view")
    }

    fn solid(value: u8, stride: usize) -> [u8; 36] {
        let mut pixels = [0_u8; 36];
        for y in 0..3 {
            for x in 0..3 {
                let offset = y * stride + x * 3;
                pixels[offset..offset + 3].fill(value);
            }
        }
        pixels
    }

    #[test]
    fn geometry_is_bounded_and_sample_positions_are_unique_and_inside() {
        assert_eq!(
            SamplingGeometry::try_new(0, 1),
            Err(SamplingGeometryError::ZeroColumns)
        );
        assert!(matches!(
            SamplingGeometry::try_new(65, 65),
            Err(SamplingGeometryError::TooManySamples { .. })
        ));

        for extent in 1_u32..100 {
            for count in 1_u16..=u16::try_from(extent.min(20)).expect("count") {
                let mut previous = None;
                for index in 0..count {
                    let coordinate = sample_coordinate(extent, index, count);
                    assert!(coordinate < extent);
                    if let Some(previous) = previous {
                        assert!(coordinate > previous);
                    }
                    previous = Some(coordinate);
                }
            }
        }
    }

    #[test]
    fn thresholds_small_frames_and_sequence_exhaustion_are_explicit_errors() {
        let fraction = PositiveUnitAmount::try_from_basis_points(1).expect("fraction");
        assert_eq!(
            MotionThresholds::try_new(0, fraction),
            Err(SceneMotionConfigError::ZeroResidualThreshold)
        );
        assert_eq!(
            MotionThresholds::try_new(511, fraction),
            Err(SceneMotionConfigError::ResidualThresholdOutOfRange {
                actual: 511,
                maximum: 510,
            })
        );

        let mut extractor = SceneMotionExtractor::new(config(4, 3, 1, 1));
        let pixels = solid(0, 12);
        assert!(matches!(
            extractor.analyze(
                view(&pixels, ChannelOrder::Rgb, 12, 0, 1),
                MonotonicTimestamp::from_nanos_since_epoch(1)
            ),
            Err(SceneMotionError::FrameSmallerThanSamplingGrid { .. })
        ));

        let mut extractor = SceneMotionExtractor::new(config(3, 3, 1, 1));
        extractor
            .analyze(
                view(&pixels, ChannelOrder::Rgb, 12, u64::MAX, 1),
                MonotonicTimestamp::from_nanos_since_epoch(1),
            )
            .expect("last frame ID may prime an epoch");
        assert_eq!(
            extractor.analyze(
                view(&pixels, ChannelOrder::Rgb, 12, u64::MAX, 2),
                MonotonicTimestamp::from_nanos_since_epoch(2)
            ),
            Err(SceneMotionError::FrameSequenceExhausted { previous: u64::MAX })
        );
    }

    #[test]
    fn cold_start_and_uniform_exposure_change_are_not_motion() {
        let mut extractor = SceneMotionExtractor::new(config(3, 3, 4, 1_000));
        let first = solid(20, 12);
        let second = solid(120, 12);
        assert!(matches!(
            extractor
                .analyze(
                    view(&first, ChannelOrder::Rgb, 12, 4, 10),
                    MonotonicTimestamp::from_nanos_since_epoch(10)
                )
                .expect("first"),
            SceneAnalysis::ColdStart(_)
        ));
        let second = extractor
            .analyze(
                view(&second, ChannelOrder::Rgb, 12, 5, 20),
                MonotonicTimestamp::from_nanos_since_epoch(20),
            )
            .expect("second");
        assert!(matches!(second, SceneAnalysis::NoMotion(_)));
        assert_eq!(second.observation().motion(), None);
    }

    #[test]
    fn even_sample_median_retains_half_luminance_without_sign_bias() {
        let mut histogram = [0_u16; DELTA_HISTOGRAM_BINS];
        histogram[255] = 1; // delta 0
        histogram[256] = 1; // delta +1
        assert_eq!(median_delta_twice(&histogram, 2), 1);

        histogram.fill(0);
        histogram[254] = 1; // delta -1
        histogram[255] = 1; // delta 0
        assert_eq!(median_delta_twice(&histogram, 2), -1);
    }

    #[test]
    fn localized_motion_has_a_weighted_pixel_centroid() {
        let mut extractor = SceneMotionExtractor::new(config(3, 3, 10, 1_000));
        let first = solid(0, 12);
        let mut second = solid(0, 12);
        second[2 * 3..2 * 3 + 3].fill(255);
        extractor
            .analyze(
                view(&first, ChannelOrder::Rgb, 12, 0, 1),
                MonotonicTimestamp::from_nanos_since_epoch(1),
            )
            .expect("prime");
        let analysis = extractor
            .analyze(
                view(&second, ChannelOrder::Rgb, 12, 1, 2),
                MonotonicTimestamp::from_nanos_since_epoch(2),
            )
            .expect("motion");
        let SceneAnalysis::Motion(observation) = analysis else {
            panic!("localized delta must be motion");
        };
        let motion = observation.motion().expect("motion value");
        assert_eq!(motion.center().x_right().basis_points(), 10_000);
        assert_eq!(motion.center().y_down().basis_points(), 0);
        assert!(motion.strength().basis_points() > 0);
    }

    #[test]
    fn rgb_bgr_and_row_padding_produce_identical_analysis() {
        let mut rgb_first = [0_u8; 36];
        let mut rgb_second = [0_u8; 36];
        let mut bgr_first = [0_u8; 36];
        let mut bgr_second = [0_u8; 36];
        for y in 0..3 {
            for x in 0..3 {
                let offset = y * 12 + x * 3;
                let rgb = [10, 40, 90];
                let changed = if x == 0 && y == 2 { [240, 5, 80] } else { rgb };
                rgb_first[offset..offset + 3].copy_from_slice(&rgb);
                rgb_second[offset..offset + 3].copy_from_slice(&changed);
                bgr_first[offset..offset + 3].copy_from_slice(&[rgb[2], rgb[1], rgb[0]]);
                bgr_second[offset..offset + 3]
                    .copy_from_slice(&[changed[2], changed[1], changed[0]]);
            }
            rgb_first[y * 12 + 9..y * 12 + 12].fill(0xa5);
            rgb_second[y * 12 + 9..y * 12 + 12].fill(0x5a);
            bgr_first[y * 12 + 9..y * 12 + 12].fill(0x11);
            bgr_second[y * 12 + 9..y * 12 + 12].fill(0xee);
        }

        let mut rgb = SceneMotionExtractor::new(config(3, 3, 1, 1));
        let mut bgr = SceneMotionExtractor::new(config(3, 3, 1, 1));
        rgb.analyze(
            view(&rgb_first, ChannelOrder::Rgb, 12, 0, 1),
            MonotonicTimestamp::from_nanos_since_epoch(1),
        )
        .expect("rgb prime");
        bgr.analyze(
            view(&bgr_first, ChannelOrder::Bgr, 12, 0, 1),
            MonotonicTimestamp::from_nanos_since_epoch(1),
        )
        .expect("bgr prime");
        let rgb = rgb
            .analyze(
                view(&rgb_second, ChannelOrder::Rgb, 12, 1, 2),
                MonotonicTimestamp::from_nanos_since_epoch(2),
            )
            .expect("rgb motion");
        let bgr = bgr
            .analyze(
                view(&bgr_second, ChannelOrder::Bgr, 12, 1, 2),
                MonotonicTimestamp::from_nanos_since_epoch(2),
            )
            .expect("bgr motion");
        assert_eq!(rgb.observation(), bgr.observation());
    }

    #[test]
    fn ordering_epoch_layout_and_clock_faults_do_not_advance_state() {
        let mut extractor = SceneMotionExtractor::new(config(3, 3, 1, 1));
        let pixels = solid(0, 12);
        extractor
            .analyze(
                view(&pixels, ChannelOrder::Rgb, 12, 9, 10),
                MonotonicTimestamp::from_nanos_since_epoch(10),
            )
            .expect("prime");

        assert!(matches!(
            extractor.analyze(
                view(&pixels, ChannelOrder::Rgb, 12, 9, 11),
                MonotonicTimestamp::from_nanos_since_epoch(11)
            ),
            Err(SceneMotionError::DuplicateFrame { .. })
        ));
        assert!(matches!(
            extractor.analyze(
                view(&pixels, ChannelOrder::Rgb, 12, 8, 12),
                MonotonicTimestamp::from_nanos_since_epoch(12)
            ),
            Err(SceneMotionError::OutOfOrderFrame { .. })
        ));
        assert!(matches!(
            extractor.analyze(
                view(&pixels, ChannelOrder::Rgb, 12, 11, 13),
                MonotonicTimestamp::from_nanos_since_epoch(13)
            ),
            Err(SceneMotionError::FrameGap {
                expected: 10,
                actual: 11
            })
        ));
        assert!(matches!(
            extractor.analyze(
                view(&pixels, ChannelOrder::Rgb, 12, 10, 9),
                MonotonicTimestamp::from_nanos_since_epoch(9)
            ),
            Err(SceneMotionError::HostClockRegressed { .. })
                | Err(SceneMotionError::FrameFromFuture { .. })
        ));

        let changed_layout = view(&pixels, ChannelOrder::Bgr, 12, 10, 14);
        assert!(matches!(
            extractor.analyze(
                changed_layout,
                MonotonicTimestamp::from_nanos_since_epoch(14)
            ),
            Err(SceneMotionError::LayoutChanged { .. })
        ));

        let layout = ImageLayout::try_new(3, 3, 12, ChannelOrder::Rgb).expect("layout");
        let other_epoch = RgbObservation::new(
            FrameId::new(StreamEpochId::try_new(8).expect("epoch"), 10),
            layout,
            freshness(15),
        );
        let other_epoch = RgbFrameView::try_new(other_epoch, &pixels).expect("view");
        assert!(matches!(
            extractor.analyze(other_epoch, MonotonicTimestamp::from_nanos_since_epoch(15)),
            Err(SceneMotionError::StreamEpochChanged { .. })
        ));

        assert!(matches!(
            extractor
                .analyze(
                    view(&pixels, ChannelOrder::Rgb, 12, 10, 16),
                    MonotonicTimestamp::from_nanos_since_epoch(16)
                )
                .expect("state did not advance"),
            SceneAnalysis::NoMotion(_)
        ));
    }

    #[test]
    fn stale_future_and_non_increasing_observation_clocks_are_rejected() {
        let mut extractor = SceneMotionExtractor::new(config(3, 3, 1, 1));
        let pixels = solid(0, 12);
        assert!(matches!(
            extractor.analyze(
                view(&pixels, ChannelOrder::Rgb, 12, 0, 10),
                MonotonicTimestamp::from_nanos_since_epoch(9)
            ),
            Err(SceneMotionError::FrameFromFuture { .. })
        ));
        assert!(matches!(
            extractor.analyze(
                view(&pixels, ChannelOrder::Rgb, 12, 0, 10),
                MonotonicTimestamp::from_nanos_since_epoch(110)
            ),
            Err(SceneMotionError::StaleFrame { .. })
        ));
        extractor
            .analyze(
                view(&pixels, ChannelOrder::Rgb, 12, 0, 10),
                MonotonicTimestamp::from_nanos_since_epoch(10),
            )
            .expect("prime");
        assert!(matches!(
            extractor.analyze(
                view(&pixels, ChannelOrder::Rgb, 12, 1, 10),
                MonotonicTimestamp::from_nanos_since_epoch(11)
            ),
            Err(SceneMotionError::ObservationClockNotIncreasing { .. })
        ));
    }
}
