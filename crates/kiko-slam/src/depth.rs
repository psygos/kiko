use std::marker::PhantomData;
use std::sync::Arc;

use crate::{FrameDimensions, FrameDimensionsError, FrameId, Timestamp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthProvenanceKind {
    MeasuredSensor,
    InterpolatedStereo,
}

pub trait DepthProvenance: Clone + Copy + std::fmt::Debug + Send + Sync + 'static {
    const KIND: DepthProvenanceKind;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeasuredDepth;

impl DepthProvenance for MeasuredDepth {
    const KIND: DepthProvenanceKind = DepthProvenanceKind::MeasuredSensor;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InterpolatedDepth;

impl DepthProvenance for InterpolatedDepth {
    const KIND: DepthProvenanceKind = DepthProvenanceKind::InterpolatedStereo;
}

pub type InterpolatedDepthImage = DepthImage<InterpolatedDepth>;

#[derive(Debug, Clone)]
pub struct DepthImage<P = MeasuredDepth> {
    frame_id: FrameId,
    timestamp: Timestamp,
    dimensions: FrameDimensions,
    depth_m: Arc<[f32]>,
    provenance: PhantomData<P>,
}

#[derive(Debug)]
pub enum DepthImageError {
    InvalidDimensions(FrameDimensionsError),
    DimensionMismatch { expected: usize, actual: usize },
    InvalidSample { index: usize, value: f32 },
}

impl std::fmt::Display for DepthImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DepthImageError::InvalidDimensions(source) => {
                write!(f, "invalid depth image dimensions: {source}")
            }
            DepthImageError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "depth image dimension mismatch: expected {expected} values, got {actual}"
                )
            }
            DepthImageError::InvalidSample { index, value } => {
                write!(f, "invalid depth sample at index {index}: {value}")
            }
        }
    }
}

impl std::error::Error for DepthImageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidDimensions(source) => Some(source),
            Self::DimensionMismatch { .. } | Self::InvalidSample { .. } => None,
        }
    }
}

impl From<FrameDimensionsError> for DepthImageError {
    fn from(source: FrameDimensionsError) -> Self {
        Self::InvalidDimensions(source)
    }
}

fn validate_depth_samples(
    dimensions: FrameDimensions,
    depth_m: &[f32],
) -> Result<(), DepthImageError> {
    let expected = dimensions.area();
    if depth_m.len() != expected {
        return Err(DepthImageError::DimensionMismatch {
            expected,
            actual: depth_m.len(),
        });
    }
    if let Some((index, value)) = depth_m
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value < 0.0)
    {
        return Err(DepthImageError::InvalidSample { index, value });
    }
    Ok(())
}

impl<P> DepthImage<P>
where
    P: DepthProvenance,
{
    fn new_with_provenance(
        frame_id: FrameId,
        timestamp: Timestamp,
        width: u32,
        height: u32,
        depth_m: Vec<f32>,
    ) -> Result<Self, DepthImageError> {
        let dimensions = FrameDimensions::try_new(width, height)?;
        validate_depth_samples(dimensions, &depth_m)?;
        Ok(Self {
            frame_id,
            timestamp,
            dimensions,
            depth_m: Arc::from(depth_m.into_boxed_slice()),
            provenance: PhantomData,
        })
    }

    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    pub fn width(&self) -> u32 {
        self.dimensions.width()
    }

    pub fn height(&self) -> u32 {
        self.dimensions.height()
    }

    pub fn dimensions(&self) -> FrameDimensions {
        self.dimensions
    }

    pub fn provenance_kind(&self) -> DepthProvenanceKind {
        P::KIND
    }

    pub fn depth_m(&self) -> &[f32] {
        self.depth_m.as_ref()
    }

    pub fn depth_m_at(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width() || y >= self.height() {
            return None;
        }
        let idx = (y as usize)
            .checked_mul(self.width() as usize)?
            .checked_add(x as usize)?;
        let depth = *self.depth_m.get(idx)?;
        if depth.is_finite() && depth > 0.0 {
            Some(depth)
        } else {
            None
        }
    }
}

impl DepthImage<MeasuredDepth> {
    pub fn new(
        frame_id: FrameId,
        timestamp: Timestamp,
        width: u32,
        height: u32,
        depth_m: Vec<f32>,
    ) -> Result<Self, DepthImageError> {
        Self::new_with_provenance(frame_id, timestamp, width, height, depth_m)
    }

    pub fn from_depth_mm(
        frame_id: FrameId,
        timestamp: Timestamp,
        width: u32,
        height: u32,
        depth_mm: Vec<u16>,
    ) -> Result<Self, DepthImageError> {
        let depth_m = depth_mm
            .into_iter()
            .map(|mm| if mm == 0 { 0.0 } else { mm as f32 * 0.001 })
            .collect();
        Self::new(frame_id, timestamp, width, height, depth_m)
    }
}

impl DepthImage<InterpolatedDepth> {
    pub fn new_interpolated(
        frame_id: FrameId,
        timestamp: Timestamp,
        width: u32,
        height: u32,
        depth_m: Vec<f32>,
    ) -> Result<Self, DepthImageError> {
        Self::new_with_provenance(frame_id, timestamp, width, height, depth_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn depth_image_rejects_zero_dimensions_with_source_context() {
        let error = DepthImage::new(FrameId::new(1), Timestamp::from_nanos(1), 0, 2, Vec::new())
            .expect_err("zero width must fail");
        assert!(matches!(
            error,
            DepthImageError::InvalidDimensions(FrameDimensionsError::Zero {
                width: 0,
                height: 2,
            })
        ));
        assert!(error.source().is_some());
    }

    #[test]
    fn depth_image_rejects_shape_mismatch() {
        let err = DepthImage::new(
            FrameId::new(1),
            Timestamp::from_nanos(1),
            2,
            2,
            vec![1.0, 2.0, 3.0],
        )
        .expect_err("shape mismatch should fail");
        assert!(matches!(
            err,
            DepthImageError::DimensionMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }

    #[test]
    fn depth_image_from_mm_converts_units_and_invalid_zero() {
        let depth = DepthImage::from_depth_mm(
            FrameId::new(7),
            Timestamp::from_nanos(9),
            2,
            2,
            vec![0, 1000, 2500, 42],
        )
        .expect("valid depth image");
        assert_eq!(depth.provenance_kind(), DepthProvenanceKind::MeasuredSensor);
        assert_eq!(
            depth.dimensions(),
            FrameDimensions::try_new(2, 2).expect("dimensions")
        );
        assert_eq!(depth.depth_m_at(0, 0), None);
        assert_eq!(depth.depth_m_at(1, 0), Some(1.0));
        assert_eq!(depth.depth_m_at(0, 1), Some(2.5));
        assert!(depth.depth_m_at(1, 1).is_some());
    }

    #[test]
    fn interpolated_depth_image_preserves_provenance() {
        let depth = DepthImage::<InterpolatedDepth>::new_interpolated(
            FrameId::new(8),
            Timestamp::from_nanos(11),
            2,
            1,
            vec![0.0, 1.25],
        )
        .expect("valid interpolated depth image");
        assert_eq!(
            depth.provenance_kind(),
            DepthProvenanceKind::InterpolatedStereo
        );
        assert_eq!(depth.depth_m_at(1, 0), Some(1.25));
    }

    #[test]
    fn depth_image_rejects_negative_sample() {
        let err = DepthImage::new(FrameId::new(3), Timestamp::from_nanos(1), 1, 1, vec![-0.5])
            .expect_err("negative depth should fail");
        assert!(matches!(
            err,
            DepthImageError::InvalidSample { index: 0, value } if (value + 0.5).abs() < 1e-6
        ));
    }

    #[test]
    fn depth_image_rejects_non_finite_sample() {
        let err = DepthImage::new(
            FrameId::new(4),
            Timestamp::from_nanos(1),
            1,
            1,
            vec![f32::NAN],
        )
        .expect_err("nan depth should fail");
        assert!(matches!(
            err,
            DepthImageError::InvalidSample { index: 0, .. }
        ));
    }
}
