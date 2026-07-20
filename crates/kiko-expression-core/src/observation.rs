//! Checked RGB-frame metadata and observations derived from a frame.

use core::{fmt, num::NonZeroU64};

use crate::{FreshnessWindow, PositiveUnitAmount, UnitAmount};

/// Non-zero identity for one uninterrupted camera stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamEpochId(NonZeroU64);

impl StreamEpochId {
    pub const fn try_new(value: u64) -> Result<Self, ObservationValueError> {
        match NonZeroU64::new(value) {
            Some(value) => Ok(Self(value)),
            None => Err(ObservationValueError::ZeroStreamEpochId),
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Identity within a camera stream epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameId {
    stream_epoch: StreamEpochId,
    sequence: u64,
}

impl FrameId {
    pub const fn new(stream_epoch: StreamEpochId, sequence: u64) -> Self {
        Self {
            stream_epoch,
            sequence,
        }
    }

    pub const fn stream_epoch(self) -> StreamEpochId {
        self.stream_epoch
    }

    pub const fn sequence(self) -> u64 {
        self.sequence
    }
}

/// Byte order for one interleaved, three-channel, eight-bit RGB pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChannelOrder {
    Rgb,
    Bgr,
}

/// A checked interleaved RGB image layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageLayout {
    width_px: u32,
    height_px: u32,
    stride_bytes: u32,
    byte_len: usize,
    channel_order: ChannelOrder,
}

impl ImageLayout {
    pub fn try_new(
        width_px: u32,
        height_px: u32,
        stride_bytes: u32,
        channel_order: ChannelOrder,
    ) -> Result<Self, ImageLayoutError> {
        if width_px == 0 {
            return Err(ImageLayoutError::ZeroWidth);
        }
        if height_px == 0 {
            return Err(ImageLayoutError::ZeroHeight);
        }
        let packed_stride = width_px
            .checked_mul(3)
            .ok_or(ImageLayoutError::LayoutSizeOverflow)?;
        if stride_bytes < packed_stride {
            return Err(ImageLayoutError::StrideTooShort {
                stride_bytes,
                required_bytes: packed_stride,
            });
        }
        let byte_len_u64 = u64::from(stride_bytes)
            .checked_mul(u64::from(height_px))
            .ok_or(ImageLayoutError::LayoutSizeOverflow)?;
        let byte_len =
            usize::try_from(byte_len_u64).map_err(|_| ImageLayoutError::LayoutSizeOverflow)?;
        Ok(Self {
            width_px,
            height_px,
            stride_bytes,
            byte_len,
            channel_order,
        })
    }

    pub const fn width_px(self) -> u32 {
        self.width_px
    }

    pub const fn height_px(self) -> u32 {
        self.height_px
    }

    pub const fn stride_bytes(self) -> u32 {
        self.stride_bytes
    }

    pub const fn byte_len(self) -> usize {
        self.byte_len
    }

    pub const fn channel_order(self) -> ChannelOrder {
        self.channel_order
    }
}

/// Metadata for one RGB frame, including its freshness contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RgbObservation {
    frame_id: FrameId,
    layout: ImageLayout,
    freshness: FreshnessWindow,
}

impl RgbObservation {
    pub const fn new(frame_id: FrameId, layout: ImageLayout, freshness: FreshnessWindow) -> Self {
        Self {
            frame_id,
            layout,
            freshness,
        }
    }

    pub const fn frame_id(self) -> FrameId {
        self.frame_id
    }

    pub const fn layout(self) -> ImageLayout {
        self.layout
    }

    pub const fn freshness(self) -> FreshnessWindow {
        self.freshness
    }
}

/// A borrowed RGB frame whose byte length exactly matches its checked layout.
///
/// Keeping pixels borrowed avoids a second frame copy. The mixer consumes only
/// [`RgbObservation`], so inference and visualization can retain separate
/// ownership without making expression decisions allocate.
#[derive(Clone, Copy)]
pub struct RgbFrameView<'a> {
    observation: RgbObservation,
    pixels: &'a [u8],
}

impl<'a> RgbFrameView<'a> {
    pub fn try_new(
        observation: RgbObservation,
        pixels: &'a [u8],
    ) -> Result<Self, ImageLayoutError> {
        let expected = observation.layout.byte_len;
        if pixels.len() != expected {
            return Err(ImageLayoutError::PixelLengthMismatch {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            observation,
            pixels,
        })
    }

    pub const fn observation(self) -> RgbObservation {
        self.observation
    }

    pub const fn pixels(self) -> &'a [u8] {
        self.pixels
    }
}

impl fmt::Debug for RgbFrameView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RgbFrameView")
            .field("observation", &self.observation)
            .field("pixel_bytes", &self.pixels.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayoutError {
    ZeroWidth,
    ZeroHeight,
    StrideTooShort {
        stride_bytes: u32,
        required_bytes: u32,
    },
    LayoutSizeOverflow,
    PixelLengthMismatch {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for ImageLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWidth => formatter.write_str("RGB image width must be non-zero"),
            Self::ZeroHeight => formatter.write_str("RGB image height must be non-zero"),
            Self::StrideTooShort {
                stride_bytes,
                required_bytes,
            } => write!(
                formatter,
                "RGB stride {stride_bytes} bytes is shorter than required {required_bytes} bytes"
            ),
            Self::LayoutSizeOverflow => formatter.write_str("RGB image layout size overflows"),
            Self::PixelLengthMismatch { expected, actual } => write!(
                formatter,
                "RGB pixel buffer has {actual} bytes; layout requires exactly {expected}"
            ),
        }
    }
}

impl core::error::Error for ImageLayoutError {}

/// A normalized image coordinate: `x=0` is left, `x=1` is right, `y=0` is
/// top, and `y=1` is bottom.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImagePoint {
    x_right: UnitAmount,
    y_down: UnitAmount,
}

impl ImagePoint {
    pub const fn new(x_right: UnitAmount, y_down: UnitAmount) -> Self {
        Self { x_right, y_down }
    }

    pub const fn x_right(self) -> UnitAmount {
        self.x_right
    }

    pub const fn y_down(self) -> UnitAmount {
        self.y_down
    }
}

/// Stable identity assigned by one person-tracking epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PersonTrackId(NonZeroU64);

impl PersonTrackId {
    pub const fn try_new(value: u64) -> Result<Self, ObservationValueError> {
        match NonZeroU64::new(value) {
            Some(value) => Ok(Self(value)),
            None => Err(ObservationValueError::ZeroPersonTrackId),
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Positive line-of-sight distance in millimetres.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DistanceMillimeters(NonZeroU64);

impl DistanceMillimeters {
    pub const fn try_new(value: u64) -> Result<Self, ObservationValueError> {
        match NonZeroU64::new(value) {
            Some(value) => Ok(Self(value)),
            None => Err(ObservationValueError::ZeroDistanceMillimeters),
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationValueError {
    ZeroStreamEpochId,
    ZeroPersonTrackId,
    ZeroDistanceMillimeters,
}

impl fmt::Display for ObservationValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroStreamEpochId => formatter.write_str("RGB stream epoch ID must be non-zero"),
            Self::ZeroPersonTrackId => formatter.write_str("person track ID must be non-zero"),
            Self::ZeroDistanceMillimeters => {
                formatter.write_str("observed distance must be positive millimetres")
            }
        }
    }
}

impl core::error::Error for ObservationValueError {}

/// One tracked person derived from a specific RGB frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PersonObservation {
    frame_id: FrameId,
    freshness: FreshnessWindow,
    track_id: PersonTrackId,
    center: ImagePoint,
    confidence: PositiveUnitAmount,
    distance_mm: Option<DistanceMillimeters>,
}

impl PersonObservation {
    pub const fn new(
        frame_id: FrameId,
        freshness: FreshnessWindow,
        track_id: PersonTrackId,
        center: ImagePoint,
        confidence: PositiveUnitAmount,
        distance_mm: Option<DistanceMillimeters>,
    ) -> Self {
        Self {
            frame_id,
            freshness,
            track_id,
            center,
            confidence,
            distance_mm,
        }
    }

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

    pub const fn confidence(self) -> PositiveUnitAmount {
        self.confidence
    }

    pub const fn distance_mm(self) -> Option<DistanceMillimeters> {
        self.distance_mm
    }
}

/// Non-zero scene motion at a normalized image location.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SceneMotion {
    strength: PositiveUnitAmount,
    center: ImagePoint,
}

impl SceneMotion {
    pub const fn new(strength: PositiveUnitAmount, center: ImagePoint) -> Self {
        Self { strength, center }
    }

    pub const fn strength(self) -> PositiveUnitAmount {
        self.strength
    }

    pub const fn center(self) -> ImagePoint {
        self.center
    }
}

/// Lightweight scene statistics derived from a specific RGB frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SceneObservation {
    frame_id: FrameId,
    freshness: FreshnessWindow,
    mean_luminance: UnitAmount,
    motion: Option<SceneMotion>,
}

impl SceneObservation {
    pub const fn new(
        frame_id: FrameId,
        freshness: FreshnessWindow,
        mean_luminance: UnitAmount,
        motion: Option<SceneMotion>,
    ) -> Self {
        Self {
            frame_id,
            freshness,
            mean_luminance,
            motion,
        }
    }

    pub const fn frame_id(self) -> FrameId {
        self.frame_id
    }

    pub const fn freshness(self) -> FreshnessWindow {
        self.freshness
    }

    pub const fn mean_luminance(self) -> UnitAmount {
        self.mean_luminance
    }

    pub const fn motion(self) -> Option<SceneMotion> {
        self.motion
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MonotonicTimestamp, NonZeroDuration};

    fn freshness() -> FreshnessWindow {
        FreshnessWindow::from_ttl(
            MonotonicTimestamp::ZERO,
            NonZeroDuration::try_from_nanos(10).unwrap(),
        )
        .unwrap()
    }

    fn epoch() -> StreamEpochId {
        StreamEpochId::try_new(1).unwrap()
    }

    #[test]
    fn image_layout_rejects_short_stride_and_wrong_buffer_length() {
        assert_eq!(
            StreamEpochId::try_new(0),
            Err(ObservationValueError::ZeroStreamEpochId)
        );
        assert_eq!(
            ImageLayout::try_new(4, 2, 11, ChannelOrder::Bgr),
            Err(ImageLayoutError::StrideTooShort {
                stride_bytes: 11,
                required_bytes: 12,
            })
        );

        let layout = ImageLayout::try_new(4, 2, 16, ChannelOrder::Bgr).unwrap();
        let observation = RgbObservation::new(FrameId::new(epoch(), 2), layout, freshness());
        assert_eq!(layout.byte_len(), 32);
        assert!(matches!(
            RgbFrameView::try_new(observation, &[0; 31]),
            Err(ImageLayoutError::PixelLengthMismatch {
                expected: 32,
                actual: 31,
            })
        ));
        assert!(RgbFrameView::try_new(observation, &[0; 32]).is_ok());
    }

    #[test]
    fn every_small_image_layout_has_exact_checked_length() {
        for width in 1_u32..64 {
            for height in 1_u32..32 {
                let packed = width * 3;
                let padding = (width + height) % 8;
                let layout =
                    ImageLayout::try_new(width, height, packed + padding, ChannelOrder::Rgb)
                        .unwrap();
                assert_eq!(
                    layout.byte_len(),
                    usize::try_from((packed + padding) * height).unwrap()
                );
            }
        }
    }
}
