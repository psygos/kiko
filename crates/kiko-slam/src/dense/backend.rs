use crate::{DepthImage, PinholeIntrinsics, Pose};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for TSDF voxel reconstruction.
#[derive(Clone, Debug)]
pub struct TsdfConfig {
    /// Side length of a single voxel in metres.
    pub voxel_size: f32,
    /// Truncation distance in metres. Voxels beyond this distance from the
    /// surface receive zero weight.  Typically 3-5x `voxel_size`.
    pub truncation_distance: f32,
    /// Maximum accumulated weight per voxel (caps running average).
    pub max_weight: f32,
    /// Number of voxels per side of a hash-map block (e.g. 8 = 8x8x8 = 512 voxels).
    pub block_side: usize,
    /// Maximum depth value (metres) beyond which depth pixels are ignored.
    pub max_integration_depth: f32,
    /// Minimum depth value (metres) below which depth pixels are ignored.
    pub min_integration_depth: f32,
    /// Extract a mesh every N integrations. 0 means never.
    pub mesh_every_n: u64,
}

impl Default for TsdfConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.02,
            truncation_distance: 0.08,
            max_weight: 64.0,
            block_side: 8,
            max_integration_depth: 4.0,
            min_integration_depth: 0.1,
            mesh_every_n: 5,
        }
    }
}

#[derive(Debug)]
pub enum TsdfConfigError {
    NonPositiveVoxelSize { value: f32 },
    TruncationTooSmall { truncation: f32, voxel_size: f32 },
    NonPositiveMaxWeight { value: f32 },
    InvalidBlockSide { value: usize },
    InvalidDepthRange { min: f32, max: f32 },
}

impl std::fmt::Display for TsdfConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TsdfConfigError::NonPositiveVoxelSize { value } => {
                write!(f, "voxel_size must be > 0, got {value}")
            }
            TsdfConfigError::TruncationTooSmall {
                truncation,
                voxel_size,
            } => write!(
                f,
                "truncation_distance ({truncation}) must be >= voxel_size ({voxel_size})"
            ),
            TsdfConfigError::NonPositiveMaxWeight { value } => {
                write!(f, "max_weight must be > 0, got {value}")
            }
            TsdfConfigError::InvalidBlockSide { value } => {
                write!(f, "block_side must be a power of two >= 2, got {value}")
            }
            TsdfConfigError::InvalidDepthRange { min, max } => {
                write!(
                    f,
                    "depth range invalid: min ({min}) must be > 0 and < max ({max})"
                )
            }
        }
    }
}

impl std::error::Error for TsdfConfigError {}

impl TsdfConfig {
    pub fn validate(&self) -> Result<(), TsdfConfigError> {
        if !self.voxel_size.is_finite() || self.voxel_size <= 0.0 {
            return Err(TsdfConfigError::NonPositiveVoxelSize {
                value: self.voxel_size,
            });
        }
        if !self.truncation_distance.is_finite() || self.truncation_distance < self.voxel_size {
            return Err(TsdfConfigError::TruncationTooSmall {
                truncation: self.truncation_distance,
                voxel_size: self.voxel_size,
            });
        }
        if !self.max_weight.is_finite() || self.max_weight <= 0.0 {
            return Err(TsdfConfigError::NonPositiveMaxWeight {
                value: self.max_weight,
            });
        }
        if self.block_side < 2 || !self.block_side.is_power_of_two() {
            return Err(TsdfConfigError::InvalidBlockSide {
                value: self.block_side,
            });
        }
        if !self.min_integration_depth.is_finite()
            || !self.max_integration_depth.is_finite()
            || self.min_integration_depth <= 0.0
            || self.min_integration_depth >= self.max_integration_depth
        {
            return Err(TsdfConfigError::InvalidDepthRange {
                min: self.min_integration_depth,
                max: self.max_integration_depth,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

/// Triangle mesh extracted from the TSDF volume via marching cubes.
#[derive(Clone, Debug)]
pub struct Mesh {
    /// Vertex positions (x, y, z) in world coordinates.
    pub positions: Vec<[f32; 3]>,
    /// Triangle indices — each triple indexes into `positions`.
    pub indices: Vec<[u32; 3]>,
    /// Per-vertex normals (optional). Same length as `positions` when present.
    pub normals: Option<Vec<[f32; 3]>>,
}

impl Mesh {
    pub fn empty() -> Self {
        Self {
            positions: Vec::new(),
            indices: Vec::new(),
            normals: None,
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum TsdfError {
    /// Error from the backend configuration.
    Config(TsdfConfigError),
    /// Integration failed (e.g. invalid depth data, GPU error).
    Integration(String),
    /// Mesh extraction failed.
    MeshExtraction(String),
    /// Backend encountered an unrecoverable internal error.
    Internal(String),
}

impl std::fmt::Display for TsdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TsdfError::Config(err) => write!(f, "tsdf config error: {err}"),
            TsdfError::Integration(msg) => write!(f, "tsdf integration error: {msg}"),
            TsdfError::MeshExtraction(msg) => write!(f, "tsdf mesh extraction error: {msg}"),
            TsdfError::Internal(msg) => write!(f, "tsdf internal error: {msg}"),
        }
    }
}

impl std::error::Error for TsdfError {}

impl From<TsdfConfigError> for TsdfError {
    fn from(err: TsdfConfigError) -> Self {
        TsdfError::Config(err)
    }
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Backend for TSDF volumetric reconstruction.
///
/// Implementations are **not** required to be `Send` or `Sync`. The backend
/// lives on a dedicated worker thread and is never shared across threads.
/// This allows CUDA-based backends (e.g. nvblox) that use thread-affine
/// contexts.
///
/// The `TsdfBackendFactory` pattern creates the backend on the worker thread:
///
/// ```ignore
/// let factory: TsdfBackendFactory = Box::new(|config| {
///     Ok(Box::new(RustTsdfBackend::new(config)?))
/// });
/// ```
pub trait TsdfBackend {
    /// Integrate a single depth frame into the TSDF volume.
    ///
    /// `pose` is the world-to-camera transform for the frame.
    /// `depth` contains per-pixel depth in metres.
    /// `intrinsics` are the pinhole camera parameters.
    fn integrate(
        &mut self,
        pose: Pose,
        depth: &DepthImage,
        intrinsics: PinholeIntrinsics,
    ) -> Result<(), TsdfError>;

    /// Clear the entire TSDF volume and all internal state.
    fn clear(&mut self) -> Result<(), TsdfError>;

    /// Extract a triangle mesh from the TSDF zero-crossing via marching cubes.
    fn extract_mesh(&self) -> Result<Mesh, TsdfError>;
}

/// Factory closure that creates a `TsdfBackend` on the worker thread.
///
/// The closure itself is `Send` (so it can cross from the main thread to the
/// worker), but the resulting backend need not be `Send`.
pub type TsdfBackendFactory =
    Box<dyn FnOnce(TsdfConfig) -> Result<Box<dyn TsdfBackend>, TsdfError> + Send>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        TsdfConfig::default()
            .validate()
            .expect("default should be valid");
    }

    #[test]
    fn config_rejects_zero_voxel_size() {
        let cfg = TsdfConfig {
            voxel_size: 0.0,
            ..TsdfConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_negative_voxel_size() {
        let cfg = TsdfConfig {
            voxel_size: -0.01,
            ..TsdfConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_truncation_smaller_than_voxel() {
        let cfg = TsdfConfig {
            voxel_size: 0.05,
            truncation_distance: 0.03,
            ..TsdfConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_zero_max_weight() {
        let cfg = TsdfConfig {
            max_weight: 0.0,
            ..TsdfConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_non_power_of_two_block_side() {
        let cfg = TsdfConfig {
            block_side: 7,
            ..TsdfConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_block_side_one() {
        let cfg = TsdfConfig {
            block_side: 1,
            ..TsdfConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_accepts_block_side_powers_of_two() {
        for side in [2, 4, 8, 16] {
            let cfg = TsdfConfig {
                block_side: side,
                ..TsdfConfig::default()
            };
            cfg.validate()
                .unwrap_or_else(|_| panic!("block_side={side} should be valid"));
        }
    }

    #[test]
    fn config_rejects_inverted_depth_range() {
        let cfg = TsdfConfig {
            min_integration_depth: 5.0,
            max_integration_depth: 2.0,
            ..TsdfConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_zero_min_depth() {
        let cfg = TsdfConfig {
            min_integration_depth: 0.0,
            ..TsdfConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn mesh_empty_is_empty() {
        let mesh = Mesh::empty();
        assert!(mesh.is_empty());
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.triangle_count(), 0);
    }

    #[test]
    fn mesh_counts_are_correct() {
        let mesh = Mesh {
            positions: vec![[0.0; 3]; 6],
            indices: vec![[0, 1, 2], [3, 4, 5]],
            normals: None,
        };
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.triangle_count(), 2);
        assert!(!mesh.is_empty());
    }

    #[test]
    fn tsdf_error_display_variants() {
        let e1 = TsdfError::Integration("test".into());
        assert!(e1.to_string().contains("integration"));
        let e2 = TsdfError::MeshExtraction("fail".into());
        assert!(e2.to_string().contains("mesh extraction"));
        let e3 = TsdfError::Internal("boom".into());
        assert!(e3.to_string().contains("internal"));
    }

    #[test]
    fn tsdf_config_error_from_converts() {
        let cfg_err = TsdfConfigError::NonPositiveVoxelSize { value: -1.0 };
        let tsdf_err: TsdfError = cfg_err.into();
        assert!(matches!(tsdf_err, TsdfError::Config(_)));
    }

    // Verify trait object safety — TsdfBackend must be object-safe.
    #[test]
    fn tsdf_backend_is_object_safe() {
        fn _assert_object_safe(_: &dyn TsdfBackend) {}
    }

    // Verify TsdfBackendFactory is Send.
    #[test]
    fn tsdf_backend_factory_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TsdfBackendFactory>();
    }
}
