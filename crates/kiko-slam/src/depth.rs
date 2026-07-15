use std::sync::Arc;

use crate::{FrameDimensions, FrameId, Timestamp};

#[derive(Debug, Clone)]
pub struct DepthImage {
    frame_id: FrameId,
    timestamp: Timestamp,
    dimensions: FrameDimensions,
    depth_m: Arc<[f32]>,
}

#[derive(Debug)]
pub enum DepthImageError {
    ZeroDimensions { width: u32, height: u32 },
    DimensionMismatch { expected: usize, actual: usize },
    InvalidDepthMeters { index: usize, value: f32 },
}

impl std::fmt::Display for DepthImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DepthImageError::ZeroDimensions { width, height } => {
                write!(
                    f,
                    "depth image dimensions must be nonzero, got {width}x{height}"
                )
            }
            DepthImageError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "depth image dimension mismatch: expected {expected} values, got {actual}"
                )
            }
            DepthImageError::InvalidDepthMeters { index, value } => write!(
                f,
                "depth sample {index} must be zero (missing) or finite positive meters, got {value}"
            ),
        }
    }
}

impl std::error::Error for DepthImageError {}

impl DepthImage {
    pub fn new(
        frame_id: FrameId,
        timestamp: Timestamp,
        width: u32,
        height: u32,
        depth_m: Vec<f32>,
    ) -> Result<Self, DepthImageError> {
        let dimensions = FrameDimensions::try_new(width, height)
            .map_err(|_| DepthImageError::ZeroDimensions { width, height })?;
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
            .find(|(_, value)| *value != 0.0 && (!value.is_finite() || *value < 0.0))
        {
            return Err(DepthImageError::InvalidDepthMeters { index, value });
        }
        Ok(Self {
            frame_id,
            timestamp,
            dimensions,
            depth_m: Arc::from(depth_m.into_boxed_slice()),
        })
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

    pub fn depth_m(&self) -> &[f32] {
        self.depth_m.as_ref()
    }

    pub fn depth_m_at(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width() || y >= self.height() {
            return None;
        }
        let idx = (y as usize)
            .saturating_mul(self.width() as usize)
            .saturating_add(x as usize);
        let depth = *self.depth_m.get(idx)?;
        if depth > 0.0 { Some(depth) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn depth_image_rejects_zero_dimensions_and_invalid_meter_samples() {
        let zero_dimensions =
            DepthImage::new(FrameId::new(1), Timestamp::from_nanos(1), 0, 0, vec![])
                .expect_err("zero dimensions are outside the image domain");
        assert!(matches!(
            zero_dimensions,
            DepthImageError::ZeroDimensions {
                width: 0,
                height: 0
            }
        ));

        for invalid in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = DepthImage::new(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                2,
                1,
                vec![0.0, invalid],
            )
            .expect_err("invalid metre sample must be rejected at the boundary");
            assert!(matches!(
                error,
                DepthImageError::InvalidDepthMeters { index: 1, value }
                    if value.to_bits() == invalid.to_bits()
            ));
        }
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
        assert_eq!(depth.depth_m_at(0, 0), None);
        assert_eq!(depth.depth_m_at(1, 0), Some(1.0));
        assert_eq!(depth.depth_m_at(0, 1), Some(2.5));
        assert!(depth.depth_m_at(1, 1).is_some());
    }
}
