use crate::math::{mat_mul_f64, mat_mul_vec_f64, so3_exp_f64, so3_log_f64};
use crate::{ImuBatch, ImuBias, ImuBiasError, ImuNoiseModel, ImuSample, Pose64, Pose64Error};

const ACCEL_BIAS_FD_STEP_MPS2: f64 = 1e-6;
const GYRO_BIAS_FD_STEP_RADPS: f64 = 1e-6;
const CORRELATION_SYMMETRY_TOLERANCE: f64 = 1e-12;
const CORRELATION_CHOLESKY_PIVOT_TOLERANCE: f64 = 1e-12;
const CORRELATION_SCALED_PIVOT_TOLERANCE: f64 = 1e-12;
const CORRELATION_INVERSE_RESIDUAL_TOLERANCE: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuResidualCovarianceRegularization {
    rotation_variance_rad2: f64,
    velocity_variance_m2_per_s2: f64,
    position_variance_m2: f64,
}

impl ImuResidualCovarianceRegularization {
    const DEFAULT: Self = Self {
        rotation_variance_rad2: 1e-6,
        velocity_variance_m2_per_s2: 1e-6,
        position_variance_m2: 1e-6,
    };

    pub fn rotation_variance_rad2(self) -> f64 {
        self.rotation_variance_rad2
    }

    pub fn velocity_variance_m2_per_s2(self) -> f64 {
        self.velocity_variance_m2_per_s2
    }

    pub fn position_variance_m2(self) -> f64 {
        self.position_variance_m2
    }

    fn diagonal_variances(self) -> [f64; 9] {
        [
            self.rotation_variance_rad2,
            self.rotation_variance_rad2,
            self.rotation_variance_rad2,
            self.velocity_variance_m2_per_s2,
            self.velocity_variance_m2_per_s2,
            self.velocity_variance_m2_per_s2,
            self.position_variance_m2,
            self.position_variance_m2,
            self.position_variance_m2,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuResidualVarianceQuantity {
    RotationRad2,
    VelocityM2PerS2,
    PositionM2,
}

impl ImuResidualVarianceQuantity {
    const BY_AXIS: [Self; 9] = [
        Self::RotationRad2,
        Self::RotationRad2,
        Self::RotationRad2,
        Self::VelocityM2PerS2,
        Self::VelocityM2PerS2,
        Self::VelocityM2PerS2,
        Self::PositionM2,
        Self::PositionM2,
        Self::PositionM2,
    ];
}

impl std::fmt::Display for ImuResidualVarianceQuantity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RotationRad2 => f.write_str("rotation variance (rad^2)"),
            Self::VelocityM2PerS2 => f.write_str("velocity variance (m^2/s^2)"),
            Self::PositionM2 => f.write_str("position variance (m^2)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasRandomWalkVarianceQuantity {
    AccelerometerBiasM2PerS4,
    GyroscopeBiasRad2PerS2,
}

impl std::fmt::Display for BiasRandomWalkVarianceQuantity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AccelerometerBiasM2PerS4 => f.write_str("accelerometer-bias variance (m^2/s^4)"),
            Self::GyroscopeBiasRad2PerS2 => f.write_str("gyroscope-bias variance (rad^2/s^2)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreintegrationInformationError {
    InvalidRegularizedResidualVariance {
        quantity: ImuResidualVarianceQuantity,
        axis: usize,
        value: f64,
    },
    NonFiniteResidualCorrelation {
        row: usize,
        col: usize,
        value: f64,
    },
    AsymmetricResidualCorrelation {
        row: usize,
        col: usize,
        upper: f64,
        lower: f64,
        tolerance: f64,
    },
    NonPositiveDefiniteResidualCorrelation {
        pivot: usize,
        schur_complement: f64,
        tolerance: f64,
    },
    IllConditionedResidualCorrelation {
        pivot: usize,
        scaled_pivot: f64,
        tolerance: f64,
    },
    NonFiniteResidualCorrelationInverse {
        row: usize,
        col: usize,
        value: f64,
    },
    NonFiniteResidualInformation {
        row: usize,
        col: usize,
        value: f64,
    },
    InaccurateResidualCorrelationInverse {
        max_abs_identity_error: f64,
        tolerance: f64,
    },
    NonFiniteBiasRandomWalkVariance {
        quantity: BiasRandomWalkVarianceQuantity,
        value: f64,
    },
}

impl std::fmt::Display for PreintegrationInformationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRegularizedResidualVariance {
                quantity,
                axis,
                value,
            } => write!(
                f,
                "regularized IMU residual {quantity} axis {axis} must be finite and > 0, got {value}"
            ),
            Self::NonFiniteResidualCorrelation { row, col, value } => write!(
                f,
                "dimensionless IMU residual correlation entry ({row}, {col}) must be finite, got {value}"
            ),
            Self::AsymmetricResidualCorrelation {
                row,
                col,
                upper,
                lower,
                tolerance,
            } => write!(
                f,
                "dimensionless IMU residual correlation is asymmetric at ({row}, {col}): upper {upper}, lower {lower}, absolute difference exceeds tolerance {tolerance}"
            ),
            Self::NonPositiveDefiniteResidualCorrelation {
                pivot,
                schur_complement,
                tolerance,
            } => write!(
                f,
                "dimensionless IMU residual correlation is not numerically positive definite at pivot {pivot}: Schur complement {schur_complement} must be finite and > tolerance {tolerance}"
            ),
            Self::IllConditionedResidualCorrelation {
                pivot,
                scaled_pivot,
                tolerance,
            } => write!(
                f,
                "dimensionless IMU residual correlation is too ill-conditioned to invert at pivot {pivot}: scaled pivot {scaled_pivot} <= tolerance {tolerance}"
            ),
            Self::NonFiniteResidualCorrelationInverse { row, col, value } => write!(
                f,
                "dimensionless IMU residual correlation inverse entry ({row}, {col}) must be finite, got {value}"
            ),
            Self::NonFiniteResidualInformation { row, col, value } => write!(
                f,
                "IMU residual information entry ({row}, {col}) must be finite, got {value}"
            ),
            Self::InaccurateResidualCorrelationInverse {
                max_abs_identity_error,
                tolerance,
            } => write!(
                f,
                "IMU residual correlation inverse failed its identity residual check: max error {max_abs_identity_error} > tolerance {tolerance}"
            ),
            Self::NonFiniteBiasRandomWalkVariance { quantity, value } => {
                write!(f, "IMU {quantity} must be finite, got {value}")
            }
        }
    }
}

impl std::error::Error for PreintegrationInformationError {}

#[derive(Clone, Debug)]
pub struct RegularizedImuResidualInformation {
    matrix: [[f64; 9]; 9],
    regularization: ImuResidualCovarianceRegularization,
}

impl RegularizedImuResidualInformation {
    pub(crate) fn matrix(&self) -> &[[f64; 9]; 9] {
        &self.matrix
    }

    #[cfg(test)]
    fn diagonal(&self) -> [f64; 9] {
        std::array::from_fn(|axis| self.matrix[axis][axis])
    }

    pub fn regularization(&self) -> ImuResidualCovarianceRegularization {
        self.regularization
    }
}

#[derive(Clone, Debug)]
pub struct FlooredBiasRandomWalkInformation {
    diagonal: [f64; 6],
    accel_variance_floor_applied: bool,
    gyro_variance_floor_applied: bool,
}

impl FlooredBiasRandomWalkInformation {
    pub const ACCEL_VARIANCE_FLOOR_M2_PER_S4: f64 = 1e-6;
    pub const GYRO_VARIANCE_FLOOR_RAD2_PER_S2: f64 = 1e-6;

    pub(crate) fn diagonal(&self) -> [f64; 6] {
        self.diagonal
    }

    pub fn accel_variance_floor_applied(&self) -> bool {
        self.accel_variance_floor_applied
    }

    pub fn gyro_variance_floor_applied(&self) -> bool {
        self.gyro_variance_floor_applied
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreintegrationQuantity {
    DeltaRotation,
    DeltaVelocityMps,
    DeltaPositionM,
    DurationSeconds,
    Covariance,
    RotationGyroBiasJacobian,
    VelocityAccelBiasJacobian,
    VelocityGyroBiasJacobian,
    PositionAccelBiasJacobian,
    PositionGyroBiasJacobian,
}

impl std::fmt::Display for PreintegrationQuantity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::DeltaRotation => "delta rotation",
            Self::DeltaVelocityMps => "delta velocity (m/s)",
            Self::DeltaPositionM => "delta position (m)",
            Self::DurationSeconds => "duration (s)",
            Self::Covariance => "covariance",
            Self::RotationGyroBiasJacobian => "rotation/gyroscope-bias Jacobian",
            Self::VelocityAccelBiasJacobian => "velocity/accelerometer-bias Jacobian",
            Self::VelocityGyroBiasJacobian => "velocity/gyroscope-bias Jacobian",
            Self::PositionAccelBiasJacobian => "position/accelerometer-bias Jacobian",
            Self::PositionGyroBiasJacobian => "position/gyroscope-bias Jacobian",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreintegrationError {
    TooFewSamples {
        len: usize,
    },
    NonPositiveSampleInterval {
        seconds: f64,
    },
    InvalidBiasPerturbation {
        source: ImuBiasError,
    },
    Information {
        source: PreintegrationInformationError,
    },
    InvalidRotation {
        quantity: PreintegrationQuantity,
        source: Pose64Error,
    },
    NonFiniteScalar {
        quantity: PreintegrationQuantity,
        value: f64,
    },
    NonFiniteVector {
        quantity: PreintegrationQuantity,
        axis: usize,
        value: f64,
    },
    NonFiniteMatrix {
        quantity: PreintegrationQuantity,
        row: usize,
        col: usize,
        value: f64,
    },
}

impl std::fmt::Display for PreintegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreintegrationError::TooFewSamples { len } => {
                write!(
                    f,
                    "imu preintegration requires at least 2 samples, got {len}"
                )
            }
            PreintegrationError::NonPositiveSampleInterval { seconds } => {
                write!(
                    f,
                    "imu sample interval must be finite and > 0 s, got {seconds}"
                )
            }
            PreintegrationError::InvalidBiasPerturbation { source } => {
                write!(
                    f,
                    "invalid finite-difference imu bias perturbation: {source}"
                )
            }
            PreintegrationError::Information { source } => {
                write!(f, "invalid IMU preintegration information: {source}")
            }
            PreintegrationError::InvalidRotation { quantity, source } => {
                write!(f, "invalid preintegration {quantity}: {source}")
            }
            PreintegrationError::NonFiniteScalar { quantity, value } => {
                write!(f, "preintegration {quantity} must be finite, got {value}")
            }
            PreintegrationError::NonFiniteVector {
                quantity,
                axis,
                value,
            } => write!(
                f,
                "preintegration {quantity} axis {axis} must be finite, got {value}"
            ),
            PreintegrationError::NonFiniteMatrix {
                quantity,
                row,
                col,
                value,
            } => write!(
                f,
                "preintegration {quantity} entry ({row}, {col}) must be finite, got {value}"
            ),
        }
    }
}

impl std::error::Error for PreintegrationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidBiasPerturbation { source } => Some(source),
            Self::Information { source } => Some(source),
            Self::InvalidRotation { source, .. } => Some(source),
            Self::TooFewSamples { .. }
            | Self::NonPositiveSampleInterval { .. }
            | Self::NonFiniteScalar { .. }
            | Self::NonFiniteVector { .. }
            | Self::NonFiniteMatrix { .. } => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CorrectedPreintegration {
    delta_rotation: [[f64; 3]; 3],
    delta_velocity_mps: [f64; 3],
    delta_position_m: [f64; 3],
}

#[derive(Clone, Debug)]
pub struct PreintegratedImu {
    bias_linearization: ImuBias,
    delta_rotation: [[f64; 3]; 3],
    delta_velocity_mps: [f64; 3],
    delta_position_m: [f64; 3],
    duration_seconds: f64,
    regularized_residual_information: RegularizedImuResidualInformation,
    floored_bias_random_walk_information: FlooredBiasRandomWalkInformation,
    d_rotation_d_gyro_bias: [[f64; 3]; 3],
    d_velocity_d_accel_bias: [[f64; 3]; 3],
    d_velocity_d_gyro_bias: [[f64; 3]; 3],
    d_position_d_accel_bias: [[f64; 3]; 3],
    d_position_d_gyro_bias: [[f64; 3]; 3],
}

impl CorrectedPreintegration {
    fn try_new(
        delta_rotation: [[f64; 3]; 3],
        delta_velocity_mps: [f64; 3],
        delta_position_m: [f64; 3],
    ) -> Result<Self, PreintegrationError> {
        validate_rotation(PreintegrationQuantity::DeltaRotation, delta_rotation)?;
        validate_vector(PreintegrationQuantity::DeltaVelocityMps, delta_velocity_mps)?;
        validate_vector(PreintegrationQuantity::DeltaPositionM, delta_position_m)?;
        Ok(Self {
            delta_rotation,
            delta_velocity_mps,
            delta_position_m,
        })
    }

    pub fn delta_rotation(&self) -> [[f64; 3]; 3] {
        self.delta_rotation
    }

    pub fn delta_velocity_mps(&self) -> [f64; 3] {
        self.delta_velocity_mps
    }

    pub fn delta_position_m(&self) -> [f64; 3] {
        self.delta_position_m
    }
}

impl PreintegratedImu {
    pub fn integrate(
        batch: &ImuBatch,
        bias: &ImuBias,
        noise: &ImuNoiseModel,
    ) -> Result<Self, PreintegrationError> {
        let core = integrate_core(batch.samples(), bias, noise)?;

        let bias_jacobians = finite_difference_bias_jacobians(batch, bias, noise)?;
        let d_rotation_d_gyro_bias = bias_jacobians.rotation_gyro;
        let d_velocity_d_accel_bias = bias_jacobians.velocity_accel;
        let d_velocity_d_gyro_bias = bias_jacobians.velocity_gyro;
        let d_position_d_accel_bias = bias_jacobians.position_accel;
        let d_position_d_gyro_bias = bias_jacobians.position_gyro;

        for (quantity, jacobian) in [
            (
                PreintegrationQuantity::RotationGyroBiasJacobian,
                &d_rotation_d_gyro_bias,
            ),
            (
                PreintegrationQuantity::VelocityAccelBiasJacobian,
                &d_velocity_d_accel_bias,
            ),
            (
                PreintegrationQuantity::VelocityGyroBiasJacobian,
                &d_velocity_d_gyro_bias,
            ),
            (
                PreintegrationQuantity::PositionAccelBiasJacobian,
                &d_position_d_accel_bias,
            ),
            (
                PreintegrationQuantity::PositionGyroBiasJacobian,
                &d_position_d_gyro_bias,
            ),
        ] {
            validate_matrix(quantity, jacobian)?;
        }
        let regularized_residual_information = regularized_residual_information(
            core.covariance,
            ImuResidualCovarianceRegularization::DEFAULT,
        )
        .map_err(|source| PreintegrationError::Information { source })?;
        let floored_bias_random_walk_information =
            bias_random_walk_information(noise, core.duration_seconds)
                .map_err(|source| PreintegrationError::Information { source })?;

        Ok(Self {
            bias_linearization: *bias,
            delta_rotation: core.delta_rotation,
            delta_velocity_mps: core.delta_velocity_mps,
            delta_position_m: core.delta_position_m,
            duration_seconds: core.duration_seconds,
            regularized_residual_information,
            floored_bias_random_walk_information,
            d_rotation_d_gyro_bias,
            d_velocity_d_accel_bias,
            d_velocity_d_gyro_bias,
            d_position_d_accel_bias,
            d_position_d_gyro_bias,
        })
    }

    pub fn bias_linearization(&self) -> &ImuBias {
        &self.bias_linearization
    }

    pub fn delta_rotation(&self) -> [[f64; 3]; 3] {
        self.delta_rotation
    }

    pub fn delta_velocity_mps(&self) -> [f64; 3] {
        self.delta_velocity_mps
    }

    pub fn delta_position_m(&self) -> [f64; 3] {
        self.delta_position_m
    }

    pub fn duration_seconds(&self) -> f64 {
        self.duration_seconds
    }

    pub fn corrected_first_order(
        &self,
        new_bias: &ImuBias,
    ) -> Result<CorrectedPreintegration, PreintegrationError> {
        let delta_accel = sub_vec3(new_bias.accel_mps2(), self.bias_linearization.accel_mps2());
        let delta_gyro = sub_vec3(new_bias.gyro_radps(), self.bias_linearization.gyro_radps());
        let rotation_delta = mat_mul_vec_f64(self.d_rotation_d_gyro_bias, delta_gyro);
        let rotation = mat_mul_f64(so3_exp_f64(rotation_delta), self.delta_rotation);
        let velocity = add_vec3(
            self.delta_velocity_mps,
            add_vec3(
                mat_mul_vec_f64(self.d_velocity_d_accel_bias, delta_accel),
                mat_mul_vec_f64(self.d_velocity_d_gyro_bias, delta_gyro),
            ),
        );
        let position = add_vec3(
            self.delta_position_m,
            add_vec3(
                mat_mul_vec_f64(self.d_position_d_accel_bias, delta_accel),
                mat_mul_vec_f64(self.d_position_d_gyro_bias, delta_gyro),
            ),
        );
        CorrectedPreintegration::try_new(rotation, velocity, position)
    }

    pub fn regularized_residual_information(&self) -> &RegularizedImuResidualInformation {
        &self.regularized_residual_information
    }

    pub fn floored_bias_random_walk_information(&self) -> &FlooredBiasRandomWalkInformation {
        &self.floored_bias_random_walk_information
    }
}

fn validate_rotation(
    quantity: PreintegrationQuantity,
    rotation: [[f64; 3]; 3],
) -> Result<(), PreintegrationError> {
    Pose64::try_from_rt(rotation, [0.0; 3])
        .map(|_| ())
        .map_err(|source| PreintegrationError::InvalidRotation { quantity, source })
}

fn validate_scalar(
    quantity: PreintegrationQuantity,
    value: f64,
) -> Result<(), PreintegrationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PreintegrationError::NonFiniteScalar { quantity, value })
    }
}

fn validate_vector(
    quantity: PreintegrationQuantity,
    vector: [f64; 3],
) -> Result<(), PreintegrationError> {
    for (axis, value) in vector.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(PreintegrationError::NonFiniteVector {
                quantity,
                axis,
                value,
            });
        }
    }
    Ok(())
}

fn validate_matrix<const ROWS: usize, const COLS: usize>(
    quantity: PreintegrationQuantity,
    matrix: &[[f64; COLS]; ROWS],
) -> Result<(), PreintegrationError> {
    for (row, values) in matrix.iter().enumerate() {
        for (col, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(PreintegrationError::NonFiniteMatrix {
                    quantity,
                    row,
                    col,
                    value,
                });
            }
        }
    }
    Ok(())
}

struct CorePreintegration {
    delta_rotation: [[f64; 3]; 3],
    delta_velocity_mps: [f64; 3],
    delta_position_m: [f64; 3],
    duration_seconds: f64,
    covariance: [[f64; 9]; 9],
}

fn integrate_core(
    samples: &[ImuSample],
    bias: &ImuBias,
    noise: &ImuNoiseModel,
) -> Result<CorePreintegration, PreintegrationError> {
    if samples.len() < 2 {
        return Err(PreintegrationError::TooFewSamples { len: samples.len() });
    }

    let mut delta_rotation = identity3();
    let mut delta_velocity_mps = [0.0_f64; 3];
    let mut delta_position_m = [0.0_f64; 3];
    let mut duration_seconds = 0.0_f64;
    let mut covariance = [[0.0_f64; 9]; 9];
    let accel_bias_mps2 = bias.accel_mps2();
    let gyro_bias_radps = bias.gyro_radps();

    for pair in samples.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        let sample_interval_seconds = b.timestamp().seconds_since(a.timestamp());
        if !sample_interval_seconds.is_finite() || sample_interval_seconds <= 0.0 {
            return Err(PreintegrationError::NonPositiveSampleInterval {
                seconds: sample_interval_seconds,
            });
        }

        let gyro_a = sub_vec3(a.gyro_radps(), gyro_bias_radps);
        let gyro_b = sub_vec3(b.gyro_radps(), gyro_bias_radps);
        let omega_mid = scale_vec3(add_vec3(gyro_a, gyro_b), 0.5);

        let accel_a = sub_vec3(a.accel_mps2(), accel_bias_mps2);
        let accel_b = sub_vec3(b.accel_mps2(), accel_bias_mps2);

        let delta_rotation_next = mat_mul_f64(
            delta_rotation,
            so3_exp_f64(scale_vec3(omega_mid, sample_interval_seconds)),
        );
        let accel_mid = scale_vec3(
            add_vec3(
                mat_mul_vec_f64(delta_rotation, accel_a),
                mat_mul_vec_f64(delta_rotation_next, accel_b),
            ),
            0.5,
        );

        delta_position_m = add_vec3(
            delta_position_m,
            add_vec3(
                scale_vec3(delta_velocity_mps, sample_interval_seconds),
                scale_vec3(
                    accel_mid,
                    0.5 * sample_interval_seconds * sample_interval_seconds,
                ),
            ),
        );
        delta_velocity_mps = add_vec3(
            delta_velocity_mps,
            scale_vec3(accel_mid, sample_interval_seconds),
        );
        delta_rotation = delta_rotation_next;
        duration_seconds += sample_interval_seconds;

        let f = error_state_transition(omega_mid, accel_mid, sample_interval_seconds);
        let g = noise_injection(delta_rotation, sample_interval_seconds);
        let mut q = [[0.0_f64; 6]; 6];
        // Noise density is continuous-time (units: rad/s/√Hz or m/s²/√Hz).
        // The discrete-time noise covariance per step is Q_d = σ² / dt.
        // This is because G already contains dt, so G Q_d G^T = dt * σ²
        // which matches the continuous-time integral ∫σ² dτ = σ² * dt.
        let inv_dt = 1.0 / sample_interval_seconds;
        let gyro_variance = noise.gyro_noise_density() * noise.gyro_noise_density() * inv_dt;
        let accel_variance = noise.accel_noise_density() * noise.accel_noise_density() * inv_dt;
        for axis in 0..3 {
            q[axis][axis] = gyro_variance;
            q[3 + axis][3 + axis] = accel_variance;
        }
        covariance = propagate_covariance(covariance, f, g, q);
    }

    let core = CorePreintegration {
        delta_rotation,
        delta_velocity_mps,
        delta_position_m,
        duration_seconds,
        covariance,
    };
    validate_rotation(PreintegrationQuantity::DeltaRotation, core.delta_rotation)?;
    validate_vector(
        PreintegrationQuantity::DeltaVelocityMps,
        core.delta_velocity_mps,
    )?;
    validate_vector(
        PreintegrationQuantity::DeltaPositionM,
        core.delta_position_m,
    )?;
    validate_scalar(
        PreintegrationQuantity::DurationSeconds,
        core.duration_seconds,
    )?;
    validate_matrix(PreintegrationQuantity::Covariance, &core.covariance)?;
    Ok(core)
}

struct BiasJacobians {
    rotation_gyro: [[f64; 3]; 3],
    velocity_accel: [[f64; 3]; 3],
    velocity_gyro: [[f64; 3]; 3],
    position_accel: [[f64; 3]; 3],
    position_gyro: [[f64; 3]; 3],
}

fn finite_difference_bias_jacobians(
    batch: &ImuBatch,
    bias: &ImuBias,
    noise: &ImuNoiseModel,
) -> Result<BiasJacobians, PreintegrationError> {
    let mut jacobians = BiasJacobians {
        rotation_gyro: [[0.0; 3]; 3],
        velocity_accel: [[0.0; 3]; 3],
        velocity_gyro: [[0.0; 3]; 3],
        position_accel: [[0.0; 3]; 3],
        position_gyro: [[0.0; 3]; 3],
    };
    for axis in 0..3 {
        let mut plus_delta = [0.0; 3];
        let mut minus_delta = [0.0; 3];
        plus_delta[axis] = GYRO_BIAS_FD_STEP_RADPS;
        minus_delta[axis] = -GYRO_BIAS_FD_STEP_RADPS;
        let plus = bias
            .checked_add([0.0; 3], plus_delta)
            .map_err(|source| PreintegrationError::InvalidBiasPerturbation { source })?;
        let minus = bias
            .checked_add([0.0; 3], minus_delta)
            .map_err(|source| PreintegrationError::InvalidBiasPerturbation { source })?;
        let plus_core = integrate_core(batch.samples(), &plus, noise)?;
        let minus_core = integrate_core(batch.samples(), &minus, noise)?;
        let central_left_rotation_delta = so3_log_f64(mat_mul_f64(
            plus_core.delta_rotation,
            transpose3(minus_core.delta_rotation),
        ));
        let denominator = 2.0 * GYRO_BIAS_FD_STEP_RADPS;
        for (row, &rotation_delta) in central_left_rotation_delta.iter().enumerate() {
            jacobians.rotation_gyro[row][axis] = rotation_delta / denominator;
            jacobians.velocity_gyro[row][axis] = (plus_core.delta_velocity_mps[row]
                - minus_core.delta_velocity_mps[row])
                / denominator;
            jacobians.position_gyro[row][axis] =
                (plus_core.delta_position_m[row] - minus_core.delta_position_m[row]) / denominator;
        }
    }

    for axis in 0..3 {
        let mut plus_delta = [0.0; 3];
        let mut minus_delta = [0.0; 3];
        plus_delta[axis] = ACCEL_BIAS_FD_STEP_MPS2;
        minus_delta[axis] = -ACCEL_BIAS_FD_STEP_MPS2;
        let plus = bias
            .checked_add(plus_delta, [0.0; 3])
            .map_err(|source| PreintegrationError::InvalidBiasPerturbation { source })?;
        let minus = bias
            .checked_add(minus_delta, [0.0; 3])
            .map_err(|source| PreintegrationError::InvalidBiasPerturbation { source })?;
        let plus_core = integrate_core(batch.samples(), &plus, noise)?;
        let minus_core = integrate_core(batch.samples(), &minus, noise)?;
        let denominator = 2.0 * ACCEL_BIAS_FD_STEP_MPS2;
        for row in 0..3 {
            jacobians.velocity_accel[row][axis] = (plus_core.delta_velocity_mps[row]
                - minus_core.delta_velocity_mps[row])
                / denominator;
            jacobians.position_accel[row][axis] =
                (plus_core.delta_position_m[row] - minus_core.delta_position_m[row]) / denominator;
        }
    }
    Ok(jacobians)
}

fn identity3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn transpose3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn add_vec3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub_vec3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale_vec3(v: [f64; 3], scale: f64) -> [f64; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

fn error_state_transition(omega_mid: [f64; 3], accel_mid: [f64; 3], dt: f64) -> [[f64; 9]; 9] {
    let mut f = identity9();
    let omega_hat = skew3(omega_mid);
    let accel_hat = skew3(accel_mid);
    for row in 0..3 {
        for col in 0..3 {
            f[row][col] -= omega_hat[row][col] * dt;
            f[3 + row][col] -= accel_hat[row][col] * dt;
            f[6 + row][col] -= accel_hat[row][col] * (0.5 * dt * dt);
        }
        f[6 + row][3 + row] = dt;
    }
    f
}

fn noise_injection(delta_rotation: [[f64; 3]; 3], dt: f64) -> [[f64; 6]; 9] {
    let mut g = [[0.0_f64; 6]; 9];
    for (row, g_row) in g.iter_mut().enumerate().take(3) {
        g_row[row] = dt;
    }
    for row in 0..3 {
        for col in 0..3 {
            g[3 + row][3 + col] = delta_rotation[row][col] * dt;
            g[6 + row][3 + col] = delta_rotation[row][col] * (0.5 * dt * dt);
        }
    }
    g
}

fn propagate_covariance(
    covariance: [[f64; 9]; 9],
    f: [[f64; 9]; 9],
    g: [[f64; 6]; 9],
    q: [[f64; 6]; 6],
) -> [[f64; 9]; 9] {
    let ft = transpose9(f);
    let gt = transpose96(g);
    let predicted = matmul9(matmul9(f, covariance), ft);
    let process = matmul9x6_6x9(matmul9x6_6x6(g, q), gt);
    symmetrize9(add9(predicted, process))
}

fn regularized_residual_information(
    mut covariance: [[f64; 9]; 9],
    regularization: ImuResidualCovarianceRegularization,
) -> Result<RegularizedImuResidualInformation, PreintegrationInformationError> {
    let mut standard_deviations = [0.0_f64; 9];
    for (axis, ((row, added_variance), quantity)) in covariance
        .iter_mut()
        .zip(regularization.diagonal_variances())
        .zip(ImuResidualVarianceQuantity::BY_AXIS)
        .enumerate()
    {
        row[axis] += added_variance;
        let variance = row[axis];
        if !variance.is_finite() || variance <= 0.0 {
            return Err(
                PreintegrationInformationError::InvalidRegularizedResidualVariance {
                    quantity,
                    axis,
                    value: variance,
                },
            );
        }
        standard_deviations[axis] = variance.sqrt();
    }

    let mut correlation = [[0.0_f64; 9]; 9];
    for row in 0..9 {
        for col in 0..9 {
            let value =
                covariance[row][col] / (standard_deviations[row] * standard_deviations[col]);
            if !value.is_finite() {
                return Err(
                    PreintegrationInformationError::NonFiniteResidualCorrelation {
                        row,
                        col,
                        value,
                    },
                );
            }
            correlation[row][col] = value;
        }
    }

    validate_residual_correlation(correlation)?;
    let correlation_inverse = invert_residual_correlation(correlation)?;
    let mut matrix = [[0.0_f64; 9]; 9];
    for row in 0..9 {
        for col in 0..9 {
            let value = correlation_inverse[row][col]
                / (standard_deviations[row] * standard_deviations[col]);
            if !value.is_finite() {
                return Err(
                    PreintegrationInformationError::NonFiniteResidualInformation {
                        row,
                        col,
                        value,
                    },
                );
            }
            matrix[row][col] = value;
        }
    }
    Ok(RegularizedImuResidualInformation {
        matrix,
        regularization,
    })
}

fn validate_residual_correlation(
    matrix: [[f64; 9]; 9],
) -> Result<(), PreintegrationInformationError> {
    for row in 0..9 {
        for col in (row + 1)..9 {
            let upper = matrix[row][col];
            let lower = matrix[col][row];
            if (upper - lower).abs() > CORRELATION_SYMMETRY_TOLERANCE {
                return Err(
                    PreintegrationInformationError::AsymmetricResidualCorrelation {
                        row,
                        col,
                        upper,
                        lower,
                        tolerance: CORRELATION_SYMMETRY_TOLERANCE,
                    },
                );
            }
        }
    }

    let mut lower = [[0.0_f64; 9]; 9];
    for row in 0..9 {
        for col in 0..=row {
            let mut schur_complement = matrix[row][col];
            for k in 0..col {
                schur_complement -= lower[row][k] * lower[col][k];
            }
            if row == col {
                if !schur_complement.is_finite()
                    || schur_complement <= CORRELATION_CHOLESKY_PIVOT_TOLERANCE
                {
                    return Err(
                        PreintegrationInformationError::NonPositiveDefiniteResidualCorrelation {
                            pivot: row,
                            schur_complement,
                            tolerance: CORRELATION_CHOLESKY_PIVOT_TOLERANCE,
                        },
                    );
                }
                lower[row][col] = schur_complement.sqrt();
            } else {
                let value = schur_complement / lower[col][col];
                if !value.is_finite() {
                    return Err(
                        PreintegrationInformationError::NonPositiveDefiniteResidualCorrelation {
                            pivot: col,
                            schur_complement,
                            tolerance: CORRELATION_CHOLESKY_PIVOT_TOLERANCE,
                        },
                    );
                }
                lower[row][col] = value;
            }
        }
    }
    Ok(())
}

fn bias_random_walk_information(
    noise: &ImuNoiseModel,
    duration_seconds: f64,
) -> Result<FlooredBiasRandomWalkInformation, PreintegrationInformationError> {
    let accel_variance = noise.accel_random_walk() * noise.accel_random_walk() * duration_seconds;
    if !accel_variance.is_finite() {
        return Err(
            PreintegrationInformationError::NonFiniteBiasRandomWalkVariance {
                quantity: BiasRandomWalkVarianceQuantity::AccelerometerBiasM2PerS4,
                value: accel_variance,
            },
        );
    }
    let gyro_variance = noise.gyro_random_walk() * noise.gyro_random_walk() * duration_seconds;
    if !gyro_variance.is_finite() {
        return Err(
            PreintegrationInformationError::NonFiniteBiasRandomWalkVariance {
                quantity: BiasRandomWalkVarianceQuantity::GyroscopeBiasRad2PerS2,
                value: gyro_variance,
            },
        );
    }
    let accel_variance_floor_applied =
        accel_variance < FlooredBiasRandomWalkInformation::ACCEL_VARIANCE_FLOOR_M2_PER_S4;
    let gyro_variance_floor_applied =
        gyro_variance < FlooredBiasRandomWalkInformation::GYRO_VARIANCE_FLOOR_RAD2_PER_S2;
    let effective_accel_variance =
        accel_variance.max(FlooredBiasRandomWalkInformation::ACCEL_VARIANCE_FLOOR_M2_PER_S4);
    let effective_gyro_variance =
        gyro_variance.max(FlooredBiasRandomWalkInformation::GYRO_VARIANCE_FLOOR_RAD2_PER_S2);
    let mut diagonal = [0.0_f64; 6];
    for axis in 0..3 {
        diagonal[axis] = 1.0 / effective_accel_variance;
        diagonal[3 + axis] = 1.0 / effective_gyro_variance;
    }
    Ok(FlooredBiasRandomWalkInformation {
        diagonal,
        accel_variance_floor_applied,
        gyro_variance_floor_applied,
    })
}

fn invert_residual_correlation(
    matrix: [[f64; 9]; 9],
) -> Result<[[f64; 9]; 9], PreintegrationInformationError> {
    const N: usize = 9;
    let mut a = matrix;
    let mut inverse = [[0.0_f64; N]; N];
    let mut row_scales = [0.0_f64; N];
    for (row_index, row) in a.iter().enumerate() {
        row_scales[row_index] = row.iter().map(|value| value.abs()).fold(0.0, f64::max);
    }
    for (idx, row) in inverse.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    for pivot in 0..N {
        let mut pivot_row = pivot;
        let mut scaled_pivot = if row_scales[pivot] > 0.0 {
            a[pivot][pivot].abs() / row_scales[pivot]
        } else {
            0.0
        };
        for (row, a_row) in a.iter().enumerate().take(N).skip(pivot + 1) {
            let candidate = if row_scales[row] > 0.0 {
                a_row[pivot].abs() / row_scales[row]
            } else {
                0.0
            };
            if candidate > scaled_pivot {
                scaled_pivot = candidate;
                pivot_row = row;
            }
        }
        if !scaled_pivot.is_finite() || scaled_pivot <= CORRELATION_SCALED_PIVOT_TOLERANCE {
            return Err(
                PreintegrationInformationError::IllConditionedResidualCorrelation {
                    pivot,
                    scaled_pivot,
                    tolerance: CORRELATION_SCALED_PIVOT_TOLERANCE,
                },
            );
        }
        if pivot_row != pivot {
            a.swap(pivot, pivot_row);
            inverse.swap(pivot, pivot_row);
            row_scales.swap(pivot, pivot_row);
        }
        let diag = a[pivot][pivot];
        for col in 0..N {
            a[pivot][col] /= diag;
            inverse[pivot][col] /= diag;
        }
        for row in 0..N {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for col in 0..N {
                a[row][col] -= factor * a[pivot][col];
                inverse[row][col] -= factor * inverse[pivot][col];
            }
        }
    }
    for row in 0..N {
        for col in 0..row {
            let symmetric = 0.5 * (inverse[row][col] + inverse[col][row]);
            inverse[row][col] = symmetric;
            inverse[col][row] = symmetric;
        }
    }
    for (row, values) in inverse.iter().enumerate() {
        for (col, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(
                    PreintegrationInformationError::NonFiniteResidualCorrelationInverse {
                        row,
                        col,
                        value,
                    },
                );
            }
        }
    }

    let mut max_abs_identity_error = 0.0_f64;
    for (row, matrix_row) in matrix.iter().enumerate() {
        for col in 0..N {
            let product = matrix_row
                .iter()
                .zip(&inverse)
                .map(|(matrix_value, inverse_row)| matrix_value * inverse_row[col])
                .sum::<f64>();
            let expected = if row == col { 1.0 } else { 0.0 };
            max_abs_identity_error = max_abs_identity_error.max((product - expected).abs());
        }
    }
    if !max_abs_identity_error.is_finite()
        || max_abs_identity_error > CORRELATION_INVERSE_RESIDUAL_TOLERANCE
    {
        return Err(
            PreintegrationInformationError::InaccurateResidualCorrelationInverse {
                max_abs_identity_error,
                tolerance: CORRELATION_INVERSE_RESIDUAL_TOLERANCE,
            },
        );
    }
    Ok(inverse)
}

fn add9(a: [[f64; 9]; 9], b: [[f64; 9]; 9]) -> [[f64; 9]; 9] {
    let mut out = [[0.0_f64; 9]; 9];
    for row in 0..9 {
        for col in 0..9 {
            out[row][col] = a[row][col] + b[row][col];
        }
    }
    out
}

fn identity9() -> [[f64; 9]; 9] {
    let mut out = [[0.0_f64; 9]; 9];
    for (idx, row) in out.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    out
}

fn matmul9(a: [[f64; 9]; 9], b: [[f64; 9]; 9]) -> [[f64; 9]; 9] {
    let mut out = [[0.0_f64; 9]; 9];
    for i in 0..9 {
        for j in 0..9 {
            let mut value = 0.0_f64;
            for (k, row) in b.iter().enumerate() {
                value += a[i][k] * row[j];
            }
            out[i][j] = value;
        }
    }
    out
}

fn matmul9x6_6x6(a: [[f64; 6]; 9], b: [[f64; 6]; 6]) -> [[f64; 6]; 9] {
    let mut out = [[0.0_f64; 6]; 9];
    for i in 0..9 {
        for j in 0..6 {
            let mut value = 0.0_f64;
            for (k, row) in b.iter().enumerate() {
                value += a[i][k] * row[j];
            }
            out[i][j] = value;
        }
    }
    out
}

fn matmul9x6_6x9(a: [[f64; 6]; 9], b: [[f64; 9]; 6]) -> [[f64; 9]; 9] {
    let mut out = [[0.0_f64; 9]; 9];
    for i in 0..9 {
        for j in 0..9 {
            let mut value = 0.0_f64;
            for (k, row) in b.iter().enumerate() {
                value += a[i][k] * row[j];
            }
            out[i][j] = value;
        }
    }
    out
}

fn skew3(v: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn symmetrize9(matrix: [[f64; 9]; 9]) -> [[f64; 9]; 9] {
    let mut out = matrix;
    for row in 0..9 {
        for col in (row + 1)..9 {
            let value = 0.5 * (matrix[row][col] + matrix[col][row]);
            out[row][col] = value;
            out[col][row] = value;
        }
    }
    out
}

fn transpose9(matrix: [[f64; 9]; 9]) -> [[f64; 9]; 9] {
    let mut out = [[0.0_f64; 9]; 9];
    for (row, matrix_row) in matrix.iter().enumerate() {
        for (col, value) in matrix_row.iter().enumerate() {
            out[col][row] = *value;
        }
    }
    out
}

fn transpose96(matrix: [[f64; 6]; 9]) -> [[f64; 9]; 6] {
    let mut out = [[0.0_f64; 9]; 6];
    for (row, matrix_row) in matrix.iter().enumerate() {
        for (col, value) in matrix_row.iter().enumerate() {
            out[col][row] = *value;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise() -> ImuNoiseModel {
        ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise")
    }

    fn batch(samples: &[(i64, [f64; 3], [f64; 3])]) -> ImuBatch {
        ImuBatch::new(
            samples
                .iter()
                .map(|(timestamp, accel, gyro)| {
                    ImuSample::new(crate::Timestamp::from_nanos(*timestamp), *accel, *gyro)
                        .expect("imu sample")
                })
                .collect(),
        )
        .expect("imu batch")
    }

    fn exact_reintegration(
        batch: &ImuBatch,
        bias: &ImuBias,
        noise: &ImuNoiseModel,
    ) -> CorrectedPreintegration {
        let core = integrate_core(batch.samples(), bias, noise).expect("exact integration core");
        CorrectedPreintegration::try_new(
            core.delta_rotation,
            core.delta_velocity_mps,
            core.delta_position_m,
        )
        .expect("validated exact reintegration")
    }

    fn matrix_trace(matrix: [[f64; 3]; 3]) -> f64 {
        matrix[0][0] + matrix[1][1] + matrix[2][2]
    }

    #[test]
    fn integrate_rejects_too_few_samples() {
        let batch = batch(&[(0, [0.0; 3], [0.0; 3])]);
        let err = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect_err("single sample should fail");
        assert_eq!(err, PreintegrationError::TooFewSamples { len: 1 });
    }

    #[test]
    fn integrate_rejects_nonfinite_derived_outputs() {
        let stationary = batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]);

        let huge_gyro_bias = ImuBias::try_new([0.0; 3], [f64::MAX, 0.0, 0.0]).expect("finite bias");
        let error = PreintegratedImu::integrate(&stationary, &huge_gyro_bias, &noise())
            .expect_err("overflowed rotation must fail");
        assert!(matches!(
            error,
            PreintegrationError::InvalidRotation {
                quantity: PreintegrationQuantity::DeltaRotation,
                source: Pose64Error::NonFiniteRotation { .. },
            }
        ));
        assert!(std::error::Error::source(&error).is_some());

        let huge_accel_bias =
            ImuBias::try_new([f64::MAX, 0.0, 0.0], [0.0; 3]).expect("finite bias");
        let error = PreintegratedImu::integrate(&stationary, &huge_accel_bias, &noise())
            .expect_err("overflowed translation terms must fail");
        assert!(matches!(
            error,
            PreintegrationError::NonFiniteVector {
                quantity: PreintegrationQuantity::DeltaVelocityMps,
                axis: 0,
                value,
            } if !value.is_finite()
        ));

        let huge_noise =
            ImuNoiseModel::new(f64::MAX, 0.01, 0.001, 0.0001).expect("finite noise model");
        let error = PreintegratedImu::integrate(&stationary, &ImuBias::default(), &huge_noise)
            .expect_err("overflowed covariance must fail");
        assert!(matches!(
            error,
            PreintegrationError::NonFiniteMatrix {
                quantity: PreintegrationQuantity::Covariance,
                value,
                ..
            } if !value.is_finite()
        ));
    }

    #[test]
    fn first_order_correction_rejects_nonfinite_derived_rotation() {
        let stationary = batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]);
        let preintegrated = PreintegratedImu::integrate(&stationary, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let huge_gyro_bias = ImuBias::try_new([0.0; 3], [f64::MAX, 0.0, 0.0]).expect("finite bias");

        let error = preintegrated
            .corrected_first_order(&huge_gyro_bias)
            .expect_err("overflowed correction must fail");
        assert!(matches!(
            error,
            PreintegrationError::InvalidRotation {
                quantity: PreintegrationQuantity::DeltaRotation,
                source: Pose64Error::NonFiniteRotation { .. },
            }
        ));
    }

    #[test]
    fn zero_specific_force_and_zero_gyro_stays_identity() {
        let batch = batch(&[
            (0, [0.0; 3], [0.0; 3]),
            (10_000_000, [0.0; 3], [0.0; 3]),
            (20_000_000, [0.0; 3], [0.0; 3]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        assert_eq!(preintegrated.delta_rotation(), identity3());
        assert_eq!(preintegrated.delta_velocity_mps(), [0.0; 3]);
        assert_eq!(preintegrated.delta_position_m(), [0.0; 3]);
        assert!((preintegrated.duration_seconds() - 0.02).abs() < 1e-12);
    }

    #[test]
    fn constant_rotation_matches_so3_exponential() {
        let batch = batch(&[
            (0, [0.0; 3], [0.0, 0.0, 0.5]),
            (10_000_000, [0.0; 3], [0.0, 0.0, 0.5]),
            (20_000_000, [0.0; 3], [0.0, 0.0, 0.5]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let expected = so3_exp_f64([0.0, 0.0, 0.01]);
        let error = so3_log_f64(mat_mul_f64(
            preintegrated.delta_rotation,
            transpose3(expected),
        ));
        let norm = (error[0] * error[0] + error[1] * error[1] + error[2] * error[2]).sqrt();
        assert!(norm < 1e-9, "rotation error={norm}");
    }

    #[test]
    fn covariance_is_symmetric_with_non_negative_diagonal() {
        let batch = batch(&[
            (0, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (10_000_000, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (20_000_000, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
        ]);
        let core = integrate_core(batch.samples(), &ImuBias::default(), &noise())
            .expect("preintegration core");
        for row in 0..9 {
            assert!(core.covariance[row][row] >= 0.0);
            for col in 0..9 {
                assert!((core.covariance[row][col] - core.covariance[col][row]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn residual_information_diagonal_is_finite_and_positive() {
        let batch = batch(&[
            (0, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (10_000_000, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (20_000_000, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        for value in preintegrated.regularized_residual_information().diagonal() {
            assert!(value.is_finite() && value > 0.0);
        }
    }

    #[test]
    fn bias_random_walk_information_decreases_after_variance_floor_is_exceeded() {
        let short = PreintegratedImu::integrate(
            &batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("short");
        let long = PreintegratedImu::integrate(
            &batch(&[
                (0, [0.0; 3], [0.0; 3]),
                (200_000_000_000, [0.0; 3], [0.0; 3]),
            ]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("long");
        let short_info = short.floored_bias_random_walk_information().diagonal();
        let long_info = long.floored_bias_random_walk_information().diagonal();
        for axis in 0..6 {
            assert!(short_info[axis] > long_info[axis]);
        }
    }

    #[test]
    fn bias_random_walk_information_reports_applied_variance_floors() {
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("preintegrated");
        let information = preintegrated.floored_bias_random_walk_information();
        assert!(information.accel_variance_floor_applied());
        assert!(information.gyro_variance_floor_applied());
        for value in information.diagonal() {
            assert!(value <= 1.0e6);
            assert!(value.is_finite() && value > 0.0);
        }
    }

    #[test]
    fn bias_random_walk_information_reports_unfloored_components() {
        let information = bias_random_walk_information(&noise(), 200.0).expect("information");
        assert!(!information.accel_variance_floor_applied());
        assert!(!information.gyro_variance_floor_applied());
        assert!(information.diagonal().into_iter().all(|value| value > 0.0));
    }

    #[test]
    fn bias_random_walk_information_rejects_overflow() {
        let huge = ImuNoiseModel::new(0.1, 0.01, f64::MAX, 0.0001).expect("finite noise");
        let error = bias_random_walk_information(&huge, 1.0)
            .expect_err("overflowed variance must not become zero information");
        assert!(matches!(
            error,
            PreintegrationInformationError::NonFiniteBiasRandomWalkVariance {
                quantity: BiasRandomWalkVarianceQuantity::AccelerometerBiasM2PerS4,
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn corrected_first_order_stays_close_to_exact_reintegration() {
        let batch = batch(&[
            (0, [0.2, 0.1, 0.0], [0.0, 0.0, 0.5]),
            (10_000_000, [0.2, 0.1, 0.0], [0.0, 0.0, 0.5]),
            (20_000_000, [0.2, 0.1, 0.0], [0.0, 0.0, 0.5]),
        ]);
        let base_bias = ImuBias::default();
        let noise = noise();
        let preintegrated =
            PreintegratedImu::integrate(&batch, &base_bias, &noise).expect("preintegrated");
        let new_bias =
            ImuBias::try_new([1e-4, -2e-4, 3e-4], [-1e-4, 2e-4, -1e-4]).expect("finite bias");
        let corrected = preintegrated
            .corrected_first_order(&new_bias)
            .expect("finite first-order correction");
        let exact = exact_reintegration(&batch, &new_bias, &noise);

        let rot_error = so3_log_f64(mat_mul_f64(
            corrected.delta_rotation(),
            transpose3(exact.delta_rotation()),
        ));
        let rot_norm = (rot_error[0] * rot_error[0]
            + rot_error[1] * rot_error[1]
            + rot_error[2] * rot_error[2])
            .sqrt();
        let vel_error = sub_vec3(corrected.delta_velocity_mps(), exact.delta_velocity_mps());
        let pos_error = sub_vec3(corrected.delta_position_m(), exact.delta_position_m());
        let vel_norm = (vel_error[0] * vel_error[0]
            + vel_error[1] * vel_error[1]
            + vel_error[2] * vel_error[2])
            .sqrt();
        let pos_norm = (pos_error[0] * pos_error[0]
            + pos_error[1] * pos_error[1]
            + pos_error[2] * pos_error[2])
            .sqrt();
        assert!(rot_norm < 1e-5, "rotation first-order error={rot_norm}");
        assert!(vel_norm < 1e-6, "velocity first-order error={vel_norm}");
        assert!(pos_norm < 1e-6, "position first-order error={pos_norm}");
    }

    #[test]
    fn gyro_bias_correction_matches_left_tangent_for_noncommuting_rotation() {
        let batch = batch(&[
            (0, [0.0; 3], [1.2, -0.4, 0.3]),
            (250_000_000, [0.0; 3], [0.7, 0.9, -0.2]),
            (500_000_000, [0.0; 3], [-0.5, 0.8, 1.0]),
            (750_000_000, [0.0; 3], [0.3, -1.1, 0.6]),
            (1_000_000_000, [0.0; 3], [0.9, 0.2, -0.7]),
        ]);
        let base_bias = ImuBias::try_new([0.0; 3], [0.05, -0.03, 0.02]).expect("base bias");
        let noise = noise();
        let preintegrated =
            PreintegratedImu::integrate(&batch, &base_bias, &noise).expect("preintegrated");
        let new_bias = base_bias
            .checked_add([0.0; 3], [1e-4, -2e-4, 1.5e-4])
            .expect("perturbed bias");
        let corrected = preintegrated
            .corrected_first_order(&new_bias)
            .expect("first-order correction");
        let exact = exact_reintegration(&batch, &new_bias, &noise);

        let rotation_error = so3_log_f64(mat_mul_f64(
            corrected.delta_rotation(),
            transpose3(exact.delta_rotation()),
        ));
        let error_norm = rotation_error
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!(
            error_norm < 1e-6,
            "left-trivialized gyro-bias correction error={error_norm}"
        );
    }

    #[test]
    fn shared_bias_perturbations_preserve_each_central_difference_jacobian() {
        let batch = batch(&[
            (0, [0.2, -0.1, 0.05], [0.1, -0.2, 0.3]),
            (10_000_000, [0.25, -0.08, 0.04], [0.12, -0.18, 0.28]),
            (20_000_000, [0.3, -0.05, 0.02], [0.15, -0.15, 0.25]),
        ]);
        let bias = ImuBias::try_new([0.01, -0.02, 0.03], [0.001, -0.002, 0.003]).expect("bias");
        let jacobians = finite_difference_bias_jacobians(&batch, &bias, &noise())
            .expect("shared perturbations");
        for axis in 0..3 {
            let mut plus_delta = [0.0; 3];
            let mut minus_delta = [0.0; 3];
            plus_delta[axis] = GYRO_BIAS_FD_STEP_RADPS;
            minus_delta[axis] = -GYRO_BIAS_FD_STEP_RADPS;

            let plus_gyro = bias
                .checked_add([0.0; 3], plus_delta)
                .expect("plus gyro bias");
            let minus_gyro = bias
                .checked_add([0.0; 3], minus_delta)
                .expect("minus gyro bias");
            let plus_gyro_core = integrate_core(batch.samples(), &plus_gyro, &noise())
                .expect("plus gyro integration");
            let minus_gyro_core = integrate_core(batch.samples(), &minus_gyro, &noise())
                .expect("minus gyro integration");
            let central_left_rotation_delta = so3_log_f64(mat_mul_f64(
                plus_gyro_core.delta_rotation,
                transpose3(minus_gyro_core.delta_rotation),
            ));
            let gyro_denominator = 2.0 * GYRO_BIAS_FD_STEP_RADPS;

            plus_delta[axis] = ACCEL_BIAS_FD_STEP_MPS2;
            minus_delta[axis] = -ACCEL_BIAS_FD_STEP_MPS2;
            let plus_accel = bias
                .checked_add(plus_delta, [0.0; 3])
                .expect("plus accel bias");
            let minus_accel = bias
                .checked_add(minus_delta, [0.0; 3])
                .expect("minus accel bias");
            let plus_accel_core = integrate_core(batch.samples(), &plus_accel, &noise())
                .expect("plus accel integration");
            let minus_accel_core = integrate_core(batch.samples(), &minus_accel, &noise())
                .expect("minus accel integration");
            let accel_denominator = 2.0 * ACCEL_BIAS_FD_STEP_MPS2;

            for (row, &rotation_delta) in central_left_rotation_delta.iter().enumerate() {
                assert_eq!(
                    jacobians.rotation_gyro[row][axis],
                    rotation_delta / gyro_denominator
                );
                assert_eq!(
                    jacobians.velocity_gyro[row][axis],
                    (plus_gyro_core.delta_velocity_mps[row]
                        - minus_gyro_core.delta_velocity_mps[row])
                        / gyro_denominator
                );
                assert_eq!(
                    jacobians.position_gyro[row][axis],
                    (plus_gyro_core.delta_position_m[row] - minus_gyro_core.delta_position_m[row])
                        / gyro_denominator
                );
                assert_eq!(
                    jacobians.velocity_accel[row][axis],
                    (plus_accel_core.delta_velocity_mps[row]
                        - minus_accel_core.delta_velocity_mps[row])
                        / accel_denominator
                );
                assert_eq!(
                    jacobians.position_accel[row][axis],
                    (plus_accel_core.delta_position_m[row]
                        - minus_accel_core.delta_position_m[row])
                        / accel_denominator
                );
            }
        }
    }

    #[test]
    fn rotation_matrix_stays_on_so3() {
        let batch = batch(&[
            (0, [0.0; 3], [0.1, -0.2, 0.3]),
            (10_000_000, [0.0; 3], [0.1, -0.2, 0.3]),
            (20_000_000, [0.0; 3], [0.1, -0.2, 0.3]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let rot = preintegrated.delta_rotation;
        let rt_r = mat_mul_f64(transpose3(rot), rot);
        let identity = identity3();
        for row in 0..3 {
            for col in 0..3 {
                assert!((rt_r[row][col] - identity[row][col]).abs() < 1e-10);
            }
        }
        let trace = matrix_trace(rot);
        assert!(trace.is_finite());
    }

    #[test]
    fn coupled_covariance_produces_cross_terms() {
        let batch = batch(&[
            (0, [0.2, 0.1, -0.05], [0.1, -0.2, 0.3]),
            (10_000_000, [0.25, 0.08, -0.04], [0.12, -0.18, 0.28]),
            (20_000_000, [0.3, 0.05, -0.02], [0.15, -0.15, 0.25]),
            (30_000_000, [0.28, 0.02, 0.0], [0.17, -0.1, 0.2]),
        ]);
        let core = integrate_core(batch.samples(), &ImuBias::default(), &noise())
            .expect("preintegration core");
        let mut max_off_diagonal = 0.0_f64;
        for row in 0..9 {
            for col in 0..9 {
                if row != col {
                    max_off_diagonal = max_off_diagonal.max(core.covariance[row][col].abs());
                }
            }
        }
        assert!(max_off_diagonal > 0.0, "cross terms should be present");
    }

    #[test]
    fn residual_information_matrix_is_symmetric_and_positive_on_diagonal() {
        let batch = batch(&[
            (0, [0.1, -0.1, 0.2], [0.0, 0.05, -0.1]),
            (10_000_000, [0.1, -0.1, 0.2], [0.0, 0.05, -0.1]),
            (20_000_000, [0.1, -0.1, 0.2], [0.0, 0.05, -0.1]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let information = preintegrated.regularized_residual_information().matrix();
        for row in 0..9 {
            assert!(information[row][row].is_finite() && information[row][row] > 0.0);
            for col in 0..9 {
                assert!(
                    (information[row][col] - information[col][row]).abs() < 1e-8,
                    "row={row} col={col}"
                );
            }
        }
    }

    #[test]
    fn residual_information_normalizes_mixed_unit_scales_before_inversion() {
        let mut covariance = [[0.0_f64; 9]; 9];
        for axis in 0..3 {
            covariance[axis][axis] = 1e-12;
            covariance[3 + axis][3 + axis] = 1e6;
            covariance[6 + axis][6 + axis] = 1e12;
        }
        let regularization = ImuResidualCovarianceRegularization::DEFAULT;
        let information = regularized_residual_information(covariance, regularization)
            .expect("scaled diagonal covariance");
        for (axis, (covariance_row, added_variance)) in covariance
            .iter()
            .zip(regularization.diagonal_variances())
            .enumerate()
        {
            let regularized_variance = covariance_row[axis] + added_variance;
            let identity_diagonal = regularized_variance * information.matrix()[axis][axis];
            assert!((identity_diagonal - 1.0).abs() < 1e-12);
        }
        assert_eq!(information.regularization().rotation_variance_rad2(), 1e-6);
        assert_eq!(
            information.regularization().velocity_variance_m2_per_s2(),
            1e-6
        );
        assert_eq!(information.regularization().position_variance_m2(), 1e-6);
    }

    #[test]
    fn residual_information_rejects_rank_deficient_correlation_without_fallback() {
        let covariance = [[f64::MAX; 9]; 9];
        let error = regularized_residual_information(
            covariance,
            ImuResidualCovarianceRegularization::DEFAULT,
        )
        .expect_err("rank-one correlation must fail closed");
        assert!(matches!(
            error,
            PreintegrationInformationError::NonPositiveDefiniteResidualCorrelation { .. }
        ));
    }

    #[test]
    fn residual_information_rejects_invertible_but_indefinite_correlation() {
        let regularization = ImuResidualCovarianceRegularization::DEFAULT;
        let mut covariance = [[0.0_f64; 9]; 9];
        for (axis, added_variance) in regularization.diagonal_variances().into_iter().enumerate() {
            covariance[axis][axis] = 1.0 - added_variance;
        }
        covariance[0][1] = 2.0;
        covariance[1][0] = 2.0;

        let error = regularized_residual_information(covariance, regularization)
            .expect_err("indefinite correlation must not become factor information");
        assert!(matches!(
            error,
            PreintegrationInformationError::NonPositiveDefiniteResidualCorrelation {
                pivot: 1,
                schur_complement,
                ..
            } if schur_complement < 0.0
        ));
    }

    #[test]
    fn residual_information_rejects_asymmetric_covariance() {
        let regularization = ImuResidualCovarianceRegularization::DEFAULT;
        let mut covariance = [[0.0_f64; 9]; 9];
        for (axis, added_variance) in regularization.diagonal_variances().into_iter().enumerate() {
            covariance[axis][axis] = 1.0 - added_variance;
        }
        covariance[0][1] = 0.25;

        let error = regularized_residual_information(covariance, regularization)
            .expect_err("asymmetric covariance must not be silently symmetrized");
        assert!(matches!(
            error,
            PreintegrationInformationError::AsymmetricResidualCorrelation {
                row: 0,
                col: 1,
                upper,
                lower: 0.0,
                ..
            } if (upper - 0.25).abs() < 1e-12
        ));
    }
}
