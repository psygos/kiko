use std::marker::PhantomData;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

// Compatibility shims for ORT static archive built with GCC 14 / glibc 2.38.
// The pre-built ORT 1.24.2 binary references symbols not present in Ubuntu 22.04's
// GCC 11 / glibc 2.35. These stubs are link-time equivalents.
mod ort_compat;

pub use inference::{
    EigenPlaces, End2EndPipeline, End2EndTimings, InferenceBackend, InferenceError, LightGlue,
    PlaceDescriptorExtractor, SuperPoint,
};
mod calibration;
mod capture;
mod channel;
pub mod dataset;
mod dense_cloud;
mod depth;
mod diagnostics;
pub mod env;
mod frontend;
mod geometry;
mod global_map;
mod imu;
mod inference;
mod local_ba;
pub mod loop_closure;
mod loop_manager;
pub mod map;
mod map_from_odom;
mod math;
#[cfg(feature = "record")]
mod oak;
mod observability;
mod pairing;
mod pipeline;
mod place_recognition;
mod pnp;
pub mod pose_graph;
mod preprocess;
mod runtime_imu;
mod surface_map;
#[cfg(test)]
pub(crate) mod test_helpers;
mod tracker;
mod triangulation;
mod tsdf;
#[cfg(feature = "vio")]
mod vio;
mod viz;
pub use calibration::{CalibrationBundle, CalibrationBundleError, InertialCalibration};
pub use capture::{
    CaptureBundle, CaptureBundleError, CaptureId, CaptureImu, CaptureInterval, CaptureIntervalError,
};
pub use channel::{
    ChannelCapacity, ChannelCapacityError, ChannelStats, ChannelStatsHandle, DropPolicy,
    DropReceiver, DropSender, SendOutcome, bounded_channel,
};
pub use dense_cloud::{
    DenseCloudConfig, DenseCloudConfigError, DenseCloudResult, DenseCloudStats, DensePoint,
    StableSurfacePoint, StableSurfacePointError, StableSurfaceResult, StableSurfaceStats,
    generate_dense_cloud, generate_dense_depth_image, generate_stable_surface_points,
};
pub use depth::{
    DepthImage, DepthImageError, DepthProvenance, DepthProvenanceKind, InterpolatedDepth,
    InterpolatedDepthImage, MeasuredDepth,
};
pub use diagnostics::{
    AllObservationsSupport, CountMetric, DiagnosticEvent, DiagnosticMetricError, FrameDiagnostics,
    HeldOutObservationsSupport, KeyframeRemovalReason, KeyframeStatus, LoopClosureRejectReason,
    LoopClosureStatus, MeanSquaredPixelResidualMetric, ObservationSupport,
    ObservationSupportMarker, PixelResidualMetric, PnpAcceptedInlierCountMetric,
    PnpAcceptedInlierPixelResidualMetric, PnpAcceptedInlierReprojectionMsePerAxisPx2Metric,
    PnpAcceptedInliersSupport, PnpInlierRatioMetric, PnpProjectableTrackedObservationCountMetric,
    PnpProjectableTrackedObservationPixelResidualMetric,
    PnpProjectableTrackedObservationReprojectionMsePerAxisPx2Metric,
    PnpProjectableTrackedObservationsSupport, PnpTrackedObservationCountMetric,
    PnpTrackedObservationsSupport, RatioMetric, StableSurfaceRetainedRawObservationsSupport,
    StableSurfaceRetainedRawPixelResidualMetric, TrackingPoseSource, VioProposalDisposition,
    VioProposalProjectableTrackedObservationCountMetric,
    VioProposalProjectableTrackedObservationPixelResidualMetric,
    VioProposalProjectableTrackedObservationsSupport,
    VisualProposalProjectableTrackedObservationCountMetric,
    VisualProposalProjectableTrackedObservationPixelResidualMetric,
    VisualProposalProjectableTrackedObservationsSupport,
    VisualVsVioSharedProjectableTrackedObservationCountMetric,
    VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric,
    VisualVsVioSharedProjectableTrackedObservationReprojectionMsePerAxisPx2Metric,
    VisualVsVioSharedProjectableTrackedObservationsSupport,
};
pub use frontend::MapObservationError;
pub use geometry::{
    BodyFrame, CamLFrame, CamRFrame, Cov3, GeometryError, ImageFrame, Info3, MapFrame, OdomFrame,
    Point3d, PositiveF64, StdDev, Transform3d, UnitRay3d, Variance, Vec3d, VoxelFrame,
};
pub use imu::{
    ImuAccumulator, ImuAccumulatorError, ImuBatch, ImuBatchError, ImuBatchSliceError, ImuBias,
    ImuBiasError, ImuExtrinsics, ImuExtrinsicsError, ImuNoiseModel, ImuNoiseModelError, ImuSample,
    ImuSampleError, ImuTimestampShiftError,
};
#[cfg(feature = "vio")]
pub use local_ba::{
    AnchorFrameInput, InertialFrameInput, VioBiasPrior, VioBiasPriorInformationQuantity,
    VioConvergenceCriterion, VioCostBreakdown, VioEvaluationStage, VioFrameEstimate,
    VioLinearizationQuantity, VioSolveConfig, VioSolveConfigError, VioSolveError, VioSolveResult,
    VioSolveTermination,
};
pub use local_ba::{
    BaCorrection, BaCost, BaCostError, BaExecutionError, BaOptimization, BaOutcomeError, BaResult,
    BaStall, BaStationary, BaTermination, DegenerateReason, LinearSolveError, LmConfig,
    LmConfigError, LocalBaConfig, LocalBaConfigError, LocalBundleAdjuster, MapObservation,
    Matrix3InverseError, ObservationResolveError, ObservationSet, ObservationSetError, PoseBaError,
    PoseBaOutcome, PoseBaRefinement, PoseBaTermination,
};
pub use loop_closure::{
    DescriptorMatchError, DescriptorSource, GlobalDescriptor, GlobalDescriptorError,
    KeyframeDatabase, LoopApplyError, LoopApplyErrorKind, LoopCandidate, LoopClosureConfig,
    LoopClosureConfigError, LoopClosureConfigInput, LoopDetectError, LoopVerificationError,
    PlaceMatch, RelocalizationCandidate, RelocalizationConfig, RelocalizationConfigError,
    RelocalizationConfigInput, RelocalizationMatch, VerifiedLoop, VerifiedRelocalization,
    aggregate_global_descriptor, match_descriptors_for_loop,
};
pub use map::{
    CovisibilityEdge, CovisibilityNode, CovisibilitySnapshot, KeyframeId, MapGeneration,
    MapInstanceId, MapSnapshot,
};
pub use map_from_odom::MapFromOdom;
pub use math::{Pose64, Pose64Error};
#[cfg(feature = "record")]
pub use oak::{OakImuError, oak_to_depth_image, oak_to_frame, oak_to_imu_batch};
pub use pairing::{
    PairingConfigError, PairingDropReason, PairingOutcome, PairingStats, PairingWindowNs,
    PendingFramesCapacity, PendingFramesCapacityError, StereoPairer,
};
pub use pipeline::{
    InferencePipeline, KeypointLimit, KeypointLimitError, PipelineError, PipelineTimings,
};
pub use place_recognition::{DescriptorInitError, DescriptorStats};
pub use pnp::{
    IntrinsicsError, Observation, PinholeIntrinsics, PnpError, PnpRefinementCost,
    PnpRefinementFallback, PnpRefinementObjectiveStage, PnpRefinementStatus,
    PnpRefinementTermination, PnpResult, Pose, RansacConfig, RansacConfigError, solve_pnp_ransac,
};
pub use runtime_imu::{
    RuntimeImuCalibrationError, apply_runtime_imu_calibration_override,
    load_runtime_imu_calibration_from_env,
};
pub use surface_map::{
    SurfaceBeliefMap, SurfaceMapConfig, SurfaceMapConfigError, SurfaceMapSummary,
};
pub use tracker::{
    BackendConfig, BackendConfigError, BackendStats, ComponentHealth, CovisibilityRatio,
    DegradationLevel, GlobalDescriptorConfig, GlobalDescriptorConfigError, KeyframeDecision,
    KeyframeInsertReason, KeyframePolicy, KeyframePolicyError, LoopSubsystemConfig, ParallaxPx,
    PoseStatus, ProjectedMatcherConfig, ProjectedMatcherConfigError, RedundancyPolicy,
    RedundancyPolicyError, SlamTracker, SystemHealth, TrackerConfig, TrackerError,
    TrackerInitError, TrackerOutput, TrackerRuntimeConfig, TrackerRuntimeConfigError,
    TrackingHealth, TrackingMatcher, TrackingPose, VioTelemetry,
};
pub use triangulation::{
    Keyframe, KeyframeError, Point3, RectifiedRowMismatchError, RectifiedRowMismatchPx,
    RectifiedStereo, RectifiedStereoError, SparseStereoSample, SparseStereoSamples,
    TriangulationConfig, TriangulationConfigError, TriangulationError, TriangulationResult,
    TriangulationStats, Triangulator,
};
pub use tsdf::{
    MeshData, TsdfCameraIntrinsics, TsdfCameraIntrinsicsError, TsdfConfig, TsdfConfigError,
    TsdfIntegrateMsg, TsdfIntegrateMsgError, TsdfMeshOutcome, TsdfSubmitOutcome, TsdfWorker,
    TsdfWorkerError,
};
#[cfg(feature = "vio")]
pub use vio::{
    BiasRandomWalkResidualQuantity, BiasRandomWalkVarianceQuantity, CorrectedPreintegration,
    DenseSolveError, DenseSolveInput, FiniteDifferenceSide, FlooredBiasRandomWalkInformation,
    Gravity, GravityError, ImuFactor, ImuJacobianEndpoint, ImuJacobianError,
    ImuResidualCovarianceRegularization, ImuResidualQuantity, ImuResidualVarianceQuantity,
    NavState, NavStateError, NavTangent, PreintegratedImu, PreintegrationError,
    PreintegrationInformationError, PreintegrationQuantity, RegularizedImuResidualInformation,
    VioFactorError,
};
pub use viz::{RerunSink, RerunSinkInitError, VizDecimation, VizDecimationError, VizLogError};

pub fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&'static str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    "unknown panic payload".to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensorId {
    StereoLeft,
    StereoRight,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub struct FrameId(u64);

impl FrameId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub struct Timestamp(i64);

impl Timestamp {
    pub fn from_nanos(ns: i64) -> Self {
        Self(ns)
    }

    pub fn as_nanos(&self) -> i64 {
        self.0
    }

    pub fn delta_ns(self, earlier: Timestamp) -> i64 {
        self.0.saturating_sub(earlier.0)
    }

    pub fn seconds_since(self, earlier: Timestamp) -> f64 {
        self.delta_ns(earlier) as f64 / 1_000_000_000.0
    }

    pub fn absolute_delta_ns(self, other: Timestamp) -> u64 {
        self.0.abs_diff(other.0)
    }

    pub fn midpoint(a: Timestamp, b: Timestamp) -> Timestamp {
        let midpoint = ((a.0 as i128) + (b.0 as i128)) / 2;
        Timestamp(midpoint.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameDimensions {
    width: NonZeroU32,
    height: NonZeroU32,
    area: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameDimensionsError {
    Zero { width: u32, height: u32 },
    AreaOverflow { width: u32, height: u32 },
}

impl std::fmt::Display for FrameDimensionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameDimensionsError::Zero { width, height } => {
                write!(f, "frame dimensions must be nonzero, got {width}x{height}")
            }
            FrameDimensionsError::AreaOverflow { width, height } => write!(
                f,
                "frame dimensions {width}x{height} exceed addressable memory"
            ),
        }
    }
}

impl std::error::Error for FrameDimensionsError {}

impl FrameDimensions {
    pub fn try_new(width: u32, height: u32) -> Result<Self, FrameDimensionsError> {
        let width_nonzero =
            NonZeroU32::new(width).ok_or(FrameDimensionsError::Zero { width, height })?;
        let height_nonzero =
            NonZeroU32::new(height).ok_or(FrameDimensionsError::Zero { width, height })?;
        let area_u64 = u64::from(width) * u64::from(height);
        let area = usize::try_from(area_u64)
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or(FrameDimensionsError::AreaOverflow { width, height })?;
        Ok(Self {
            width: width_nonzero,
            height: height_nonzero,
            area,
        })
    }

    pub fn width(self) -> u32 {
        self.width.get()
    }

    pub fn height(self) -> u32 {
        self.height.get()
    }

    pub fn area(self) -> usize {
        self.area.get()
    }
}

// Define these much more concretely
#[derive(Debug)]
pub enum FrameError {
    InvalidDimensions(FrameDimensionsError),
    DimensionMismatch { expected: usize, actual: usize },
}
impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::InvalidDimensions(source) => {
                write!(f, "invalid frame dimensions: {source}")
            }
            FrameError::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FrameError::InvalidDimensions(source) => Some(source),
            FrameError::DimensionMismatch { .. } => None,
        }
    }
}

impl From<FrameDimensionsError> for FrameError {
    fn from(source: FrameDimensionsError) -> Self {
        Self::InvalidDimensions(source)
    }
}

#[derive(Debug)]
pub enum PairError {
    DimensionMismatch {
        left: FrameDimensions,
        right: FrameDimensions,
    },
    TimestampDelta {
        delta_ns: u64,
        max_delta_ns: u64,
    },
    SensorMismatch {
        left: SensorId,
        right: SensorId,
    },
}

impl std::fmt::Display for PairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PairError::DimensionMismatch { left, right } => {
                write!(
                    f,
                    "stereo dimension mismatch: left={}x{}, right={}x{}",
                    left.width(),
                    left.height(),
                    right.width(),
                    right.height()
                )
            }
            PairError::TimestampDelta {
                delta_ns,
                max_delta_ns,
            } => {
                write!(
                    f,
                    "stereo delta {delta_ns}ns exceeds window {max_delta_ns}ns"
                )
            }
            PairError::SensorMismatch { left, right } => {
                write!(f, "stereo sensor mismatch: left={left:?}, right={right:?}")
            }
        }
    }
}

impl std::error::Error for PairError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DownscaleFactor(NonZeroU32);

impl DownscaleFactor {
    pub fn new(value: NonZeroU32) -> Self {
        Self(value)
    }

    pub fn get(self) -> usize {
        self.0.get() as usize
    }

    pub fn as_u32(self) -> u32 {
        self.0.get()
    }

    pub fn identity() -> Self {
        Self(NonZeroU32::MIN)
    }
}

#[derive(Debug)]
pub enum DownscaleError {
    Zero,
    InvalidInteger {
        value: String,
        source: std::num::ParseIntError,
    },
    TooLarge {
        value: usize,
    },
    InvalidDimensions(FrameDimensionsError),
    InputLenMismatch {
        expected: usize,
        actual: usize,
    },
    NonDivisible {
        width: u32,
        height: u32,
        factor: usize,
    },
}

impl std::fmt::Display for DownscaleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownscaleError::Zero => write!(f, "downscale factor must be > 0"),
            DownscaleError::InvalidInteger { value, source } => {
                write!(f, "invalid downscale factor integer {value:?}: {source}")
            }
            DownscaleError::TooLarge { value } => {
                write!(f, "downscale factor {value} exceeds u32::MAX")
            }
            DownscaleError::InvalidDimensions(source) => {
                write!(f, "invalid downscale dimensions: {source}")
            }
            DownscaleError::InputLenMismatch { expected, actual } => {
                write!(
                    f,
                    "downscale input length mismatch: expected {expected}, got {actual}"
                )
            }
            DownscaleError::NonDivisible {
                width,
                height,
                factor,
            } => write!(
                f,
                "downscale factor {factor} does not divide frame {width}x{height}"
            ),
        }
    }
}

impl std::error::Error for DownscaleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DownscaleError::InvalidInteger { source, .. } => Some(source),
            DownscaleError::InvalidDimensions(source) => Some(source),
            DownscaleError::Zero
            | DownscaleError::TooLarge { .. }
            | DownscaleError::InputLenMismatch { .. }
            | DownscaleError::NonDivisible { .. } => None,
        }
    }
}

impl TryFrom<usize> for DownscaleFactor {
    type Error = DownscaleError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        let value = u32::try_from(value).map_err(|_| DownscaleError::TooLarge { value })?;
        NonZeroU32::new(value)
            .map(DownscaleFactor)
            .ok_or(DownscaleError::Zero)
    }
}

impl std::fmt::Display for DownscaleFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.get())
    }
}

impl std::str::FromStr for DownscaleFactor {
    type Err = DownscaleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value: usize = s
            .trim()
            .parse()
            .map_err(|source| DownscaleError::InvalidInteger {
                value: s.to_string(),
                source,
            })?;
        Self::try_from(value)
    }
}

#[derive(Clone, Debug)]
pub struct Frame {
    sensor_id: SensorId,
    frame_id: FrameId,
    timestamp: Timestamp,
    dimensions: FrameDimensions,
    data: Arc<[u8]>,
}

impl Frame {
    pub fn new(
        sensor_id: SensorId,
        frame_id: FrameId,
        timestamp: Timestamp,
        width: u32,
        height: u32,
        data: Vec<u8>,
    ) -> Result<Self, FrameError> {
        let dimensions = FrameDimensions::try_new(width, height)?;
        let size = dimensions.area();

        if data.len() != size {
            return Err(FrameError::DimensionMismatch {
                expected: size,
                actual: data.len(),
            });
        }

        Ok(Self {
            sensor_id,
            frame_id,
            timestamp,
            dimensions,
            data: Arc::from(data.into_boxed_slice()),
        })
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
    pub fn data(&self) -> &[u8] {
        self.data.as_ref()
    }

    pub fn sensor_id(&self) -> SensorId {
        self.sensor_id
    }

    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
}

pub const DESCRIPTOR_DIM: usize = 256;
const U8_SCALE: f32 = 255.0;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Descriptor(pub [f32; DESCRIPTOR_DIM]);

impl Descriptor {
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn quantize(&self) -> CompactDescriptor {
        let mut out = [0_u8; DESCRIPTOR_DIM];
        for (idx, value) in self.0.iter().enumerate() {
            let clamped = value.clamp(0.0, 1.0);
            out[idx] = (clamped * U8_SCALE).round() as u8;
        }
        CompactDescriptor(out)
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactDescriptor(pub [u8; DESCRIPTOR_DIM]);

impl CompactDescriptor {
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let (dot, norm_a, norm_b) = self.0.iter().zip(other.0.iter()).fold(
            (0_u32, 0_u32, 0_u32),
            |(dot, na, nb), (&a, &b)| {
                let a = a as u32;
                let b = b as u32;
                (
                    dot.saturating_add(a.saturating_mul(b)),
                    na.saturating_add(a.saturating_mul(a)),
                    nb.saturating_add(b.saturating_mul(b)),
                )
            },
        );
        if norm_a == 0 || norm_b == 0 {
            return 0.0;
        }
        (dot as f32) / ((norm_a as f32).sqrt() * (norm_b as f32).sqrt())
    }
}

#[derive(Debug)]
pub enum DetectionError {
    InvalidDimensions(FrameDimensionsError),
    ShapeMismatch {
        keypoints_len: usize,
        scores_len: usize,
        descriptors_len: usize,
    },
    NonFiniteKeypoint {
        index: usize,
        x: f32,
        y: f32,
    },
    KeypointOutOfBounds {
        index: usize,
        x: f32,
        y: f32,
        width: u32,
        height: u32,
    },
    NonFiniteScore {
        index: usize,
        value: f32,
    },
    NonFiniteDescriptor {
        detection_index: usize,
        component: usize,
        value: f32,
    },
    SelectionIndexOutOfBounds {
        selection_index: usize,
        detection_index: usize,
        detection_count: usize,
    },
}

impl std::fmt::Display for DetectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectionError::InvalidDimensions(source) => {
                write!(f, "invalid detection dimensions: {source}")
            }
            DetectionError::ShapeMismatch {
                keypoints_len,
                scores_len,
                descriptors_len,
            } => write!(
                f,
                "detections shape mismatch: keypoints={keypoints_len}, scores={scores_len}, descriptors={descriptors_len}"
            ),
            DetectionError::NonFiniteKeypoint { index, x, y } => {
                write!(f, "detection {index} has non-finite keypoint ({x}, {y})")
            }
            DetectionError::KeypointOutOfBounds {
                index,
                x,
                y,
                width,
                height,
            } => write!(
                f,
                "detection {index} keypoint ({x}, {y}) lies outside {width}x{height} image"
            ),
            DetectionError::NonFiniteScore { index, value } => {
                write!(f, "detection {index} has non-finite score {value}")
            }
            DetectionError::NonFiniteDescriptor {
                detection_index,
                component,
                value,
            } => write!(
                f,
                "detection {detection_index} descriptor component {component} is non-finite: {value}"
            ),
            DetectionError::SelectionIndexOutOfBounds {
                selection_index,
                detection_index,
                detection_count,
            } => write!(
                f,
                "detection selection {selection_index} references index {detection_index}, but the batch contains {detection_count} detections"
            ),
        }
    }
}

impl std::error::Error for DetectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DetectionError::InvalidDimensions(source) => Some(source),
            DetectionError::ShapeMismatch { .. }
            | DetectionError::NonFiniteKeypoint { .. }
            | DetectionError::KeypointOutOfBounds { .. }
            | DetectionError::NonFiniteScore { .. }
            | DetectionError::NonFiniteDescriptor { .. }
            | DetectionError::SelectionIndexOutOfBounds { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Detections {
    sensor_id: SensorId,
    frame_id: FrameId,
    dimensions: FrameDimensions,
    keypoints: Vec<Keypoint>,
    scores: Vec<f32>,
    descriptors: Vec<Descriptor>,
}

impl Detections {
    pub fn sensor_id(&self) -> SensorId {
        self.sensor_id
    }

    pub fn frame_id(&self) -> FrameId {
        self.frame_id
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

    pub fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }

    pub fn keypoints_flat(&self) -> &[f32] {
        bytemuck::cast_slice(self.keypoints.as_slice())
    }

    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    pub fn descriptors(&self) -> &[Descriptor] {
        &self.descriptors
    }

    pub fn descriptors_flat(&self) -> &[f32] {
        bytemuck::cast_slice(self.descriptors.as_slice())
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    pub fn new(
        sensor_id: SensorId,
        frame_id: FrameId,
        width: u32,
        height: u32,
        keypoints: Vec<Keypoint>,
        scores: Vec<f32>,
        descriptors: Vec<Descriptor>,
    ) -> Result<Self, DetectionError> {
        let dimensions =
            FrameDimensions::try_new(width, height).map_err(DetectionError::InvalidDimensions)?;
        if keypoints.len() != descriptors.len() || descriptors.len() != scores.len() {
            return Err(DetectionError::ShapeMismatch {
                keypoints_len: keypoints.len(),
                scores_len: scores.len(),
                descriptors_len: descriptors.len(),
            });
        }

        for (index, keypoint) in keypoints.iter().enumerate() {
            if !keypoint.x.is_finite() || !keypoint.y.is_finite() {
                return Err(DetectionError::NonFiniteKeypoint {
                    index,
                    x: keypoint.x,
                    y: keypoint.y,
                });
            }
            if keypoint.x < 0.0
                || keypoint.y < 0.0
                || keypoint.x >= width as f32
                || keypoint.y >= height as f32
            {
                return Err(DetectionError::KeypointOutOfBounds {
                    index,
                    x: keypoint.x,
                    y: keypoint.y,
                    width,
                    height,
                });
            }
        }
        for (index, &value) in scores.iter().enumerate() {
            if !value.is_finite() {
                return Err(DetectionError::NonFiniteScore { index, value });
            }
        }
        for (detection_index, descriptor) in descriptors.iter().enumerate() {
            for (component, &value) in descriptor.0.iter().enumerate() {
                if !value.is_finite() {
                    return Err(DetectionError::NonFiniteDescriptor {
                        detection_index,
                        component,
                        value,
                    });
                }
            }
        }

        Ok(Self {
            sensor_id,
            frame_id,
            dimensions,
            keypoints,
            scores,
            descriptors,
        })
    }

    pub fn top_k(self, max: usize) -> Self {
        if self.descriptors.len() <= max {
            return self;
        }

        let Detections {
            sensor_id,
            frame_id,
            dimensions,
            keypoints,
            scores,
            descriptors,
        } = self;

        let mut order: Vec<usize> = (0..descriptors.len()).collect();
        let kth = max.saturating_sub(1);
        let cmp_desc = |&a: &usize, &b: &usize| scores[b].total_cmp(&scores[a]);
        order.select_nth_unstable_by(kth, cmp_desc);
        order.truncate(max);
        order.sort_unstable_by(cmp_desc);

        let mut new_keypoints = Vec::with_capacity(order.len());
        let mut new_scores = Vec::with_capacity(order.len());
        let mut new_descriptors = Vec::with_capacity(order.len());

        for &idx in &order {
            new_keypoints.push(keypoints[idx]);
            new_scores.push(scores[idx]);
            new_descriptors.push(descriptors[idx]);
        }

        Self {
            sensor_id,
            frame_id,
            dimensions,
            keypoints: new_keypoints,
            scores: new_scores,
            descriptors: new_descriptors,
        }
    }

    pub(crate) fn select(&self, indices: &[usize]) -> Result<Self, DetectionError> {
        let mut keypoints = Vec::with_capacity(indices.len());
        let mut scores = Vec::with_capacity(indices.len());
        let mut descriptors = Vec::with_capacity(indices.len());
        for (selection_index, &detection_index) in indices.iter().enumerate() {
            let Some(&keypoint) = self.keypoints.get(detection_index) else {
                return Err(DetectionError::SelectionIndexOutOfBounds {
                    selection_index,
                    detection_index,
                    detection_count: self.len(),
                });
            };
            keypoints.push(keypoint);
            scores.push(self.scores[detection_index]);
            descriptors.push(self.descriptors[detection_index]);
        }
        Ok(Self {
            sensor_id: self.sensor_id,
            frame_id: self.frame_id,
            dimensions: self.dimensions,
            keypoints,
            scores,
            descriptors,
        })
    }
}

pub trait FrameSource {
    fn next_frame(&mut self) -> Option<Frame>;

    fn frames(self) -> Frames<Self>
    where
        Self: Sized,
    {
        Frames::new(self)
    }
}

pub struct Frames<S> {
    source: S,
}

impl<S> Frames<S> {
    pub fn new(source: S) -> Self {
        Self { source }
    }
}

impl<S: FrameSource> Iterator for Frames<S> {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.next_frame()
    }
}

#[derive(Clone, Debug)]
pub struct StereoPair {
    left: Frame,
    right: Frame,
}

impl StereoPair {
    pub fn try_new(left: Frame, right: Frame, window: PairingWindowNs) -> Result<Self, PairError> {
        if left.sensor_id() != SensorId::StereoLeft || right.sensor_id() != SensorId::StereoRight {
            return Err(PairError::SensorMismatch {
                left: left.sensor_id(),
                right: right.sensor_id(),
            });
        }

        if left.width() != right.width() || left.height() != right.height() {
            return Err(PairError::DimensionMismatch {
                left: left.dimensions(),
                right: right.dimensions(),
            });
        }

        let delta = left.timestamp().absolute_delta_ns(right.timestamp());
        if delta > window.as_ns() {
            return Err(PairError::TimestampDelta {
                delta_ns: delta,
                max_delta_ns: window.as_ns(),
            });
        }

        Ok(Self { left, right })
    }

    /// Construct a pair without validation. Use when frames are known to be
    /// correctly paired (e.g. from a pre-validated dataset manifest).
    pub(crate) fn from_parts(left: Frame, right: Frame) -> Self {
        Self { left, right }
    }

    pub fn left(&self) -> &Frame {
        &self.left
    }

    pub fn right(&self) -> &Frame {
        &self.right
    }

    pub fn into_parts(self) -> (Frame, Frame) {
        (self.left, self.right)
    }

    pub fn timestamp_delta_ns(&self) -> u64 {
        self.left
            .timestamp()
            .absolute_delta_ns(self.right.timestamp())
    }

    pub fn capture_time(&self) -> Timestamp {
        Timestamp::midpoint(self.left.timestamp(), self.right.timestamp())
    }
}

pub trait StereoSource {
    fn left(&mut self) -> Option<Frame>;
    fn right(&mut self) -> Option<Frame>;

    fn stereo_pair(&mut self) -> Option<StereoPair> {
        Some(StereoPair::from_parts(self.left()?, self.right()?))
    }

    fn stereo_pairs(self) -> StereoPairs<Self>
    where
        Self: Sized,
    {
        StereoPairs::new(self)
    }

    fn left_frames(self) -> LeftFrames<Self>
    where
        Self: Sized,
    {
        LeftFrames::new(self)
    }

    fn right_frames(self) -> RightFrames<Self>
    where
        Self: Sized,
    {
        RightFrames::new(self)
    }
}

pub struct StereoPairs<S> {
    source: S,
}

impl<S> StereoPairs<S> {
    pub fn new(source: S) -> Self {
        Self { source }
    }
}

impl<S: StereoSource> Iterator for StereoPairs<S> {
    type Item = StereoPair;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.stereo_pair()
    }
}

pub struct LeftFrames<S> {
    source: S,
}

impl<S> LeftFrames<S> {
    pub fn new(source: S) -> Self {
        Self { source }
    }
}

impl<S: StereoSource> Iterator for LeftFrames<S> {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.left()
    }
}

pub struct RightFrames<S> {
    source: S,
}

impl<S> RightFrames<S> {
    pub fn new(source: S) -> Self {
        Self { source }
    }
}

impl<S: StereoSource> Iterator for RightFrames<S> {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.right()
    }
}

// Typestates for Matches
#[derive(Debug)]
pub struct Raw;
#[derive(Debug)]
pub struct Verified;

#[derive(Debug)]
pub enum MatchError {
    Mismatch {
        score_len: usize,
        indices_len: usize,
    },
    SourceAIndexOutOfBounds {
        match_index: usize,
        detection_index: usize,
        detection_count: usize,
    },
    SourceBIndexOutOfBounds {
        match_index: usize,
        detection_index: usize,
        detection_count: usize,
    },
    DuplicateSourceAIndex {
        detection_index: usize,
        first_match: usize,
        duplicate_match: usize,
    },
    DuplicateSourceBIndex {
        detection_index: usize,
        first_match: usize,
        duplicate_match: usize,
    },
    NonFiniteScore {
        match_index: usize,
        value: f32,
    },
    SourceBatchMismatch {
        operation: &'static str,
        actual_frame: FrameId,
        actual_sensor: SensorId,
        expected_frame: FrameId,
        expected_sensor: SensorId,
    },
}

impl std::fmt::Display for MatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchError::Mismatch {
                score_len,
                indices_len,
            } => write!(
                f,
                "match shape mismatch: scores={score_len}, indices={indices_len}"
            ),
            MatchError::SourceAIndexOutOfBounds {
                match_index,
                detection_index,
                detection_count,
            } => write!(
                f,
                "match {match_index} source-a index {detection_index} is out of bounds for {detection_count} detections"
            ),
            MatchError::SourceBIndexOutOfBounds {
                match_index,
                detection_index,
                detection_count,
            } => write!(
                f,
                "match {match_index} source-b index {detection_index} is out of bounds for {detection_count} detections"
            ),
            MatchError::DuplicateSourceAIndex {
                detection_index,
                first_match,
                duplicate_match,
            } => write!(
                f,
                "source-a detection {detection_index} is reused by matches {first_match} and {duplicate_match}"
            ),
            MatchError::DuplicateSourceBIndex {
                detection_index,
                first_match,
                duplicate_match,
            } => write!(
                f,
                "source-b detection {detection_index} is reused by matches {first_match} and {duplicate_match}"
            ),
            MatchError::NonFiniteScore { match_index, value } => {
                write!(f, "match {match_index} has non-finite score {value}")
            }
            MatchError::SourceBatchMismatch {
                operation,
                actual_frame,
                actual_sensor,
                expected_frame,
                expected_sensor,
            } => write!(
                f,
                "cannot {operation}: source-b is a different detection batch (actual={actual_sensor:?}/frame {}, expected={expected_sensor:?}/frame {})",
                actual_frame.as_u64(),
                expected_frame.as_u64()
            ),
        }
    }
}

impl std::error::Error for MatchError {}

#[derive(Debug)]
pub struct Matches<State> {
    source_a: Arc<Detections>,
    source_b: Arc<Detections>,
    indices: Vec<(usize, usize)>,
    scores: Vec<f32>,
    verified_map_instance_id: Option<MapInstanceId>,
    verified_keyframe_id: Option<KeyframeId>,
    _state: PhantomData<State>,
}

impl Matches<Raw> {
    pub fn new(
        source_a: Arc<Detections>,
        source_b: Arc<Detections>,
        indices: Vec<(usize, usize)>,
        scores: Vec<f32>,
    ) -> Result<Self, MatchError> {
        if indices.len() != scores.len() {
            return Err(MatchError::Mismatch {
                score_len: scores.len(),
                indices_len: indices.len(),
            });
        }

        for (match_index, (&(source_a_index, source_b_index), &score)) in
            indices.iter().zip(&scores).enumerate()
        {
            if source_a_index >= source_a.len() {
                return Err(MatchError::SourceAIndexOutOfBounds {
                    match_index,
                    detection_index: source_a_index,
                    detection_count: source_a.len(),
                });
            }
            if source_b_index >= source_b.len() {
                return Err(MatchError::SourceBIndexOutOfBounds {
                    match_index,
                    detection_index: source_b_index,
                    detection_count: source_b.len(),
                });
            }
            if !score.is_finite() {
                return Err(MatchError::NonFiniteScore {
                    match_index,
                    value: score,
                });
            }
        }

        Ok(Self {
            source_a,
            source_b,
            indices,
            scores,
            verified_map_instance_id: None,
            verified_keyframe_id: None,
            _state: PhantomData,
        })
    }

    pub(crate) fn with_landmarks(
        &self,
        map_instance_id: MapInstanceId,
        keyframe_id: KeyframeId,
        keyframe: &Keyframe,
    ) -> Result<Matches<Verified>, MatchError> {
        self.require_source_b(keyframe.detections(), "resolve landmark correspondences")?;
        let mut indices = Vec::with_capacity(self.indices.len());
        let mut scores = Vec::with_capacity(self.scores.len());
        for (idx, &(a, b)) in self.indices.iter().enumerate() {
            if keyframe.landmark_for_detection(b).is_some() {
                indices.push((a, b));
                scores.push(self.scores[idx]);
            }
        }

        Matches::from_verified_subset(
            self.source_a_arc(),
            self.source_b_arc(),
            indices,
            scores,
            map_instance_id,
            keyframe_id,
        )
    }
}

impl Matches<Verified> {
    fn from_verified_subset(
        source_a: Arc<Detections>,
        source_b: Arc<Detections>,
        indices: Vec<(usize, usize)>,
        scores: Vec<f32>,
        map_instance_id: MapInstanceId,
        keyframe_id: KeyframeId,
    ) -> Result<Self, MatchError> {
        if indices.len() < 2 {
            return Ok(Self {
                source_a,
                source_b,
                indices,
                scores,
                verified_map_instance_id: Some(map_instance_id),
                verified_keyframe_id: Some(keyframe_id),
                _state: PhantomData,
            });
        }
        let mut source_a_matches = vec![None; source_a.len()];
        let mut source_b_matches = vec![None; source_b.len()];
        for (match_index, &(source_a_index, source_b_index)) in indices.iter().enumerate() {
            if let Some(first_match) = source_a_matches[source_a_index] {
                return Err(MatchError::DuplicateSourceAIndex {
                    detection_index: source_a_index,
                    first_match,
                    duplicate_match: match_index,
                });
            }
            if let Some(first_match) = source_b_matches[source_b_index] {
                return Err(MatchError::DuplicateSourceBIndex {
                    detection_index: source_b_index,
                    first_match,
                    duplicate_match: match_index,
                });
            }
            source_a_matches[source_a_index] = Some(match_index);
            source_b_matches[source_b_index] = Some(match_index);
        }
        Ok(Self {
            source_a,
            source_b,
            indices,
            scores,
            verified_map_instance_id: Some(map_instance_id),
            verified_keyframe_id: Some(keyframe_id),
            _state: PhantomData,
        })
    }

    pub(crate) fn keyframe_id(&self) -> Option<KeyframeId> {
        self.verified_keyframe_id
    }

    pub(crate) fn map_instance_id(&self) -> Option<MapInstanceId> {
        self.verified_map_instance_id
    }
}

impl<State> Matches<State> {
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn source_a(&self) -> &Detections {
        &self.source_a
    }

    pub fn source_b(&self) -> &Detections {
        &self.source_b
    }

    pub fn source_a_arc(&self) -> Arc<Detections> {
        Arc::clone(&self.source_a)
    }

    pub fn source_b_arc(&self) -> Arc<Detections> {
        Arc::clone(&self.source_b)
    }

    pub fn indices(&self) -> &[(usize, usize)] {
        &self.indices
    }

    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    pub(crate) fn require_source_b(
        &self,
        expected: &Arc<Detections>,
        operation: &'static str,
    ) -> Result<(), MatchError> {
        if Arc::ptr_eq(&self.source_b, expected) {
            return Ok(());
        }
        Err(MatchError::SourceBatchMismatch {
            operation,
            actual_frame: self.source_b.frame_id(),
            actual_sensor: self.source_b.sensor_id(),
            expected_frame: expected.frame_id(),
            expected_sensor: expected.sensor_id(),
        })
    }
}

#[derive(Debug)]
pub enum VizError {
    FrameMismatch {
        left: FrameId,
        right: FrameId,
        matches_left: FrameId,
        matches_right: FrameId,
    },
    SensorMismatch {
        left: SensorId,
        right: SensorId,
        matches_left: SensorId,
        matches_right: SensorId,
    },
}

impl std::fmt::Display for VizError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizError::FrameMismatch {
                left,
                right,
                matches_left,
                matches_right,
            } => write!(
                f,
                "viz packet frame mismatch: left={}, right={}, matches_left={}, matches_right={}",
                left.as_u64(),
                right.as_u64(),
                matches_left.as_u64(),
                matches_right.as_u64()
            ),
            VizError::SensorMismatch {
                left,
                right,
                matches_left,
                matches_right,
            } => write!(
                f,
                "viz packet sensor mismatch: left={left:?}, right={right:?}, matches_left={matches_left:?}, matches_right={matches_right:?}"
            ),
        }
    }
}

impl std::error::Error for VizError {}

#[derive(Debug)]
pub struct VizPacket<State> {
    left: Frame,
    right: Frame,
    matches: Matches<State>,
}

impl<State> VizPacket<State> {
    pub fn try_new(left: Frame, right: Frame, matches: Matches<State>) -> Result<Self, VizError> {
        let matches_left = matches.source_a().frame_id();
        let matches_right = matches.source_b().frame_id();
        let matches_left_sensor = matches.source_a().sensor_id();
        let matches_right_sensor = matches.source_b().sensor_id();
        if left.frame_id() != matches_left || right.frame_id() != matches_right {
            return Err(VizError::FrameMismatch {
                left: left.frame_id(),
                right: right.frame_id(),
                matches_left,
                matches_right,
            });
        }
        if left.sensor_id() != matches_left_sensor || right.sensor_id() != matches_right_sensor {
            return Err(VizError::SensorMismatch {
                left: left.sensor_id(),
                right: right.sensor_id(),
                matches_left: matches_left_sensor,
                matches_right: matches_right_sensor,
            });
        }

        Ok(Self {
            left,
            right,
            matches,
        })
    }

    pub fn left(&self) -> &Frame {
        &self.left
    }

    pub fn right(&self) -> &Frame {
        &self.right
    }

    pub fn matches(&self) -> &Matches<State> {
        &self.matches
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;
    use std::sync::Arc;

    use super::map::SlamMap;
    use super::{
        CompactDescriptor, DESCRIPTOR_DIM, Descriptor, DetectionError, Detections, DownscaleError,
        DownscaleFactor, Frame, FrameDimensions, FrameDimensionsError, FrameError, FrameId,
        Keyframe, KeyframeId, Keypoint, MapInstanceId, MatchError, Matches, Point3, Pose, SensorId,
        Timestamp, U8_SCALE,
    };

    fn cosine_f32(a: &Descriptor, b: &Descriptor) -> f32 {
        let mut dot = 0.0_f32;
        let mut norm_a = 0.0_f32;
        let mut norm_b = 0.0_f32;
        for i in 0..DESCRIPTOR_DIM {
            let x = a.0[i];
            let y = b.0[i];
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        if norm_a <= 0.0 || norm_b <= 0.0 {
            return 0.0;
        }
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    fn detection_batch(sensor_id: SensorId, frame_id: u64, len: usize) -> Arc<Detections> {
        let keypoints = (0..len)
            .map(|index| Keypoint {
                x: index as f32 + 1.0,
                y: 1.0,
            })
            .collect();
        Arc::new(
            Detections::new(
                sensor_id,
                FrameId::new(frame_id),
                32,
                24,
                keypoints,
                vec![1.0; len],
                vec![Descriptor([0.0; DESCRIPTOR_DIM]); len],
            )
            .expect("valid detection batch"),
        )
    }

    fn keyframe_provenance(detections: &Detections) -> (MapInstanceId, KeyframeId) {
        let mut map = SlamMap::new();
        let keyframe_id = map
            .add_keyframe_from_detections(detections, Timestamp::from_nanos(1), Pose::identity())
            .expect("map keyframe");
        (map.instance_id(), keyframe_id)
    }

    #[test]
    fn quantize_preserves_similarity_ordering() {
        let mut base = [0.0_f32; DESCRIPTOR_DIM];
        let mut close = [0.0_f32; DESCRIPTOR_DIM];
        let mut far = [0.0_f32; DESCRIPTOR_DIM];
        for i in 0..DESCRIPTOR_DIM {
            let t = i as f32 / U8_SCALE;
            base[i] = t;
            close[i] = (t + 0.02).clamp(0.0, 1.0);
            far[i] = if i < 128 { 1.0 } else { 0.0 };
        }
        let base = Descriptor(base);
        let close = Descriptor(close);
        let far = Descriptor(far);

        let float_close = cosine_f32(&base, &close);
        let float_far = cosine_f32(&base, &far);
        assert!(float_close > float_far);

        let q_base = base.quantize();
        let q_close = close.quantize();
        let q_far = far.quantize();
        let quant_close = q_base.cosine_similarity(&q_close);
        let quant_far = q_base.cosine_similarity(&q_far);
        assert!(quant_close > quant_far);
    }

    #[test]
    fn frame_dimensions_and_frames_reject_zero_sizes() {
        assert!(matches!(
            FrameDimensions::try_new(0, 480),
            Err(FrameDimensionsError::Zero {
                width: 0,
                height: 480
            })
        ));
        assert!(matches!(
            Frame::new(
                SensorId::StereoLeft,
                FrameId::new(1),
                Timestamp::from_nanos(1),
                0,
                480,
                Vec::new(),
            ),
            Err(FrameError::InvalidDimensions(
                FrameDimensionsError::Zero { .. }
            ))
        ));
    }

    #[test]
    fn detections_reject_invalid_external_values() {
        let make = |keypoint: Keypoint, score: f32, descriptor: Descriptor| {
            Detections::new(
                SensorId::StereoLeft,
                FrameId::new(1),
                10,
                10,
                vec![keypoint],
                vec![score],
                vec![descriptor],
            )
        };
        let descriptor = Descriptor([0.0; DESCRIPTOR_DIM]);

        assert!(matches!(
            Detections::new(
                SensorId::StereoLeft,
                FrameId::new(1),
                0,
                10,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ),
            Err(DetectionError::InvalidDimensions(
                FrameDimensionsError::Zero { .. }
            ))
        ));
        assert!(matches!(
            make(
                Keypoint {
                    x: f32::NAN,
                    y: 1.0
                },
                1.0,
                descriptor
            ),
            Err(DetectionError::NonFiniteKeypoint { index: 0, .. })
        ));
        assert!(matches!(
            make(Keypoint { x: 10.0, y: 1.0 }, 1.0, descriptor),
            Err(DetectionError::KeypointOutOfBounds { index: 0, .. })
        ));
        assert!(matches!(
            make(Keypoint { x: 1.0, y: 1.0 }, f32::INFINITY, descriptor),
            Err(DetectionError::NonFiniteScore { index: 0, .. })
        ));

        let mut values = [0.0; DESCRIPTOR_DIM];
        values[7] = f32::NAN;
        assert!(matches!(
            make(Keypoint { x: 1.0, y: 1.0 }, 1.0, Descriptor(values)),
            Err(DetectionError::NonFiniteDescriptor {
                detection_index: 0,
                component: 7,
                ..
            })
        ));

        let batch = detection_batch(SensorId::StereoLeft, 1, 1);
        assert!(matches!(
            batch.select(&[1]),
            Err(DetectionError::SelectionIndexOutOfBounds {
                selection_index: 0,
                detection_index: 1,
                detection_count: 1,
            })
        ));
    }

    #[test]
    fn raw_matches_reject_forged_correspondences() {
        let source_a = detection_batch(SensorId::StereoLeft, 1, 2);
        let source_b = detection_batch(SensorId::StereoRight, 1, 2);

        assert!(matches!(
            Matches::new(
                Arc::clone(&source_a),
                Arc::clone(&source_b),
                vec![(2, 0)],
                vec![1.0],
            ),
            Err(MatchError::SourceAIndexOutOfBounds { match_index: 0, .. })
        ));
        assert!(matches!(
            Matches::new(
                Arc::clone(&source_a),
                Arc::clone(&source_b),
                vec![(0, 2)],
                vec![1.0],
            ),
            Err(MatchError::SourceBIndexOutOfBounds { match_index: 0, .. })
        ));
        assert!(matches!(
            Matches::new(source_a, source_b, vec![(0, 0)], vec![f32::NAN]),
            Err(MatchError::NonFiniteScore { match_index: 0, .. })
        ));
    }

    #[test]
    fn verified_matches_require_the_exact_keyframe_detection_batch() {
        let current = detection_batch(SensorId::StereoLeft, 2, 2);
        let keyframe_detections = detection_batch(SensorId::StereoLeft, 1, 2);
        let keyframe = Keyframe::from_arc(
            Arc::clone(&keyframe_detections),
            vec![Point3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            }],
            vec![1],
        )
        .expect("keyframe");
        let (map_instance_id, keyframe_id) = keyframe_provenance(&keyframe_detections);

        let matches = Matches::new(
            Arc::clone(&current),
            Arc::clone(&keyframe_detections),
            vec![(0, 0), (1, 1)],
            vec![0.8, 0.9],
        )
        .expect("raw matches");
        let verified = matches
            .with_landmarks(map_instance_id, keyframe_id, &keyframe)
            .expect("verified matches");
        assert_eq!(verified.indices(), &[(1, 1)]);
        assert_eq!(verified.map_instance_id(), Some(map_instance_id));
        assert_eq!(verified.keyframe_id(), Some(keyframe_id));

        let same_metadata_different_batch = detection_batch(SensorId::StereoLeft, 1, 2);
        let forged = Matches::new(
            current,
            same_metadata_different_batch,
            vec![(0, 1)],
            vec![1.0],
        )
        .expect("raw matches");
        assert!(matches!(
            forged.with_landmarks(map_instance_id, keyframe_id, &keyframe),
            Err(MatchError::SourceBatchMismatch { .. })
        ));
    }

    #[test]
    fn verified_matches_reject_duplicate_correspondences() {
        let current = detection_batch(SensorId::StereoLeft, 2, 2);
        let keyframe_detections = detection_batch(SensorId::StereoLeft, 1, 2);
        let keyframe = Keyframe::from_arc(
            Arc::clone(&keyframe_detections),
            vec![
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                Point3 {
                    x: 1.0,
                    y: 0.0,
                    z: 1.0,
                },
            ],
            vec![0, 1],
        )
        .expect("keyframe");
        let (map_instance_id, keyframe_id) = keyframe_provenance(&keyframe_detections);

        let duplicate_current = Matches::new(
            Arc::clone(&current),
            Arc::clone(&keyframe_detections),
            vec![(0, 0), (0, 1)],
            vec![1.0, 0.9],
        )
        .expect("duplicates are lawful in raw stereo matches");
        assert!(matches!(
            duplicate_current.with_landmarks(map_instance_id, keyframe_id, &keyframe),
            Err(MatchError::DuplicateSourceAIndex {
                first_match: 0,
                duplicate_match: 1,
                ..
            })
        ));

        let duplicate_landmark = Matches::new(
            current,
            keyframe_detections,
            vec![(0, 0), (1, 0)],
            vec![1.0, 0.9],
        )
        .expect("duplicates are lawful in raw stereo matches");
        assert!(matches!(
            duplicate_landmark.with_landmarks(map_instance_id, keyframe_id, &keyframe),
            Err(MatchError::DuplicateSourceBIndex {
                first_match: 0,
                duplicate_match: 1,
                ..
            })
        ));
    }

    #[test]
    fn compact_descriptor_cosine_identical_is_one() {
        let mut data = [0_u8; DESCRIPTOR_DIM];
        for (idx, value) in data.iter_mut().enumerate() {
            *value = ((idx * 7) % 251) as u8;
        }
        let a = CompactDescriptor(data);
        let b = CompactDescriptor(data);
        let sim = a.cosine_similarity(&b);
        assert!((sim - 1.0).abs() < 1e-6, "sim={sim}");
    }

    #[test]
    fn compact_descriptor_cosine_orthogonal_is_zeroish() {
        let mut a = [0_u8; DESCRIPTOR_DIM];
        let mut b = [0_u8; DESCRIPTOR_DIM];
        for value in a.iter_mut().take(128) {
            *value = 255;
        }
        for value in b.iter_mut().skip(128) {
            *value = 255;
        }
        let sim = CompactDescriptor(a).cosine_similarity(&CompactDescriptor(b));
        assert!(sim.abs() < 1e-6, "sim={sim}");
    }

    #[test]
    fn timestamp_delta_ns_preserves_sign() {
        let earlier = Timestamp::from_nanos(10);
        let later = Timestamp::from_nanos(25);
        assert_eq!(later.delta_ns(earlier), 15);
        assert_eq!(earlier.delta_ns(later), -15);
    }

    #[test]
    fn timestamp_absolute_delta_handles_full_i64_domain() {
        let earliest = Timestamp::from_nanos(i64::MIN);
        let latest = Timestamp::from_nanos(i64::MAX);
        assert_eq!(earliest.absolute_delta_ns(latest), u64::MAX);
        assert_eq!(latest.absolute_delta_ns(earliest), u64::MAX);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn downscale_factor_rejects_values_that_cannot_narrow_to_u32() {
        let value = u32::MAX as usize + 1;
        assert!(matches!(
            DownscaleFactor::try_from(value),
            Err(DownscaleError::TooLarge { value: rejected }) if rejected == value
        ));
    }

    #[test]
    fn downscale_factor_parser_preserves_integer_source_and_domain_errors() {
        let parse_error = "not-a-factor"
            .parse::<DownscaleFactor>()
            .expect_err("invalid integer must fail");
        assert!(matches!(
            &parse_error,
            DownscaleError::InvalidInteger { .. }
        ));
        assert!(parse_error.source().is_some());

        assert!(matches!(
            "0".parse::<DownscaleFactor>(),
            Err(DownscaleError::Zero)
        ));
    }

    #[test]
    fn timestamp_seconds_since_converts_nanoseconds() {
        let earlier = Timestamp::from_nanos(1_500_000_000);
        let later = Timestamp::from_nanos(2_750_000_000);
        assert!((later.seconds_since(earlier) - 1.25).abs() < 1e-12);
    }

    #[test]
    fn timestamp_midpoint_handles_order_and_odd_delta() {
        let a = Timestamp::from_nanos(10);
        let b = Timestamp::from_nanos(15);
        assert_eq!(Timestamp::midpoint(a, b).as_nanos(), 12);
        assert_eq!(Timestamp::midpoint(b, a).as_nanos(), 12);
    }
}
