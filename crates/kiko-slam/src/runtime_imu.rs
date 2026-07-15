use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::dataset::{Calibration, ImuCalibration, ImuExtrinsicsMeta, ImuNoiseMeta};
use crate::{Pose64, Pose64Error};

const IMU_CALIBRATION_FILE_ENV: &str = "KIKO_IMU_CALIBRATION_FILE";
const IMU_ROTATION_ENV: &str = "KIKO_IMU_ROTATION";
const IMU_TRANSLATION_ENV: &str = "KIKO_IMU_TRANSLATION";
const IMU_ACCEL_NOISE_ENV: &str = "KIKO_IMU_ACCEL_NOISE_DENSITY";
const IMU_GYRO_NOISE_ENV: &str = "KIKO_IMU_GYRO_NOISE_DENSITY";
const IMU_ACCEL_RW_ENV: &str = "KIKO_IMU_ACCEL_RANDOM_WALK";
const IMU_GYRO_RW_ENV: &str = "KIKO_IMU_GYRO_RANDOM_WALK";
const IMU_TIME_OFFSET_ENV: &str = "KIKO_IMU_TIME_OFFSET_NS";
const IMU_GRAVITY_ENV: &str = "KIKO_IMU_GRAVITY_MPS2";
const IMU_INITIAL_ACCEL_BIAS_ENV: &str = "KIKO_IMU_INITIAL_ACCEL_BIAS";
const IMU_INITIAL_GYRO_BIAS_ENV: &str = "KIKO_IMU_INITIAL_GYRO_BIAS";

#[derive(Debug)]
pub enum RuntimeImuCalibrationError {
    ConflictingSources {
        file_env: &'static str,
        direct_keys_present: Vec<&'static str>,
    },
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    UnsupportedFileFormat {
        path: PathBuf,
    },
    MissingImuCalibrationBlock {
        path: PathBuf,
    },
    MissingEnv {
        key: &'static str,
    },
    Environment {
        source: crate::env::EnvError,
    },
    InvalidEnvFloat {
        key: &'static str,
        value: String,
        element_index: Option<usize>,
        source: std::num::ParseFloatError,
    },
    InvalidEnvInteger {
        key: &'static str,
        value: String,
        source: std::num::ParseIntError,
    },
    InvalidEnvVector {
        key: &'static str,
        expected: usize,
        actual: usize,
    },
    BasaltMissingLeftCamera {
        path: PathBuf,
    },
    BasaltNonIsotropicNoise {
        path: PathBuf,
        field: &'static str,
        values: [f64; 3],
    },
    BasaltInvalidQuaternion {
        path: PathBuf,
        field: &'static str,
    },
    BasaltInvalidPose {
        path: PathBuf,
        source: Pose64Error,
    },
    BasaltUnsupportedBiasModel {
        path: PathBuf,
        field: &'static str,
        trailing_terms: Vec<f64>,
    },
}

impl std::fmt::Display for RuntimeImuCalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConflictingSources {
                file_env,
                direct_keys_present,
            } => write!(
                f,
                "{file_env} cannot be combined with direct IMU override env vars: {}",
                direct_keys_present.join(", ")
            ),
            Self::Io { path, source } => {
                write!(
                    f,
                    "failed to read runtime IMU calibration override `{}`: {source}",
                    path.display()
                )
            }
            Self::Json { path, source } => {
                write!(
                    f,
                    "failed to parse runtime IMU calibration override `{}`: {source}",
                    path.display()
                )
            }
            Self::UnsupportedFileFormat { path } => write!(
                f,
                "runtime IMU calibration override `{}` is neither an IMU block, a dataset calibration, nor a Basalt calibration",
                path.display()
            ),
            Self::MissingImuCalibrationBlock { path } => write!(
                f,
                "runtime IMU calibration override `{}` does not contain an `imu` block",
                path.display()
            ),
            Self::MissingEnv { key } => {
                write!(
                    f,
                    "{key} is required when runtime IMU env calibration is configured"
                )
            }
            Self::Environment { source } => {
                write!(f, "failed to read runtime IMU environment: {source}")
            }
            Self::InvalidEnvFloat {
                key,
                value,
                element_index: None,
                source,
            } => write!(f, "failed to parse {key}={value:?} as a float: {source}"),
            Self::InvalidEnvFloat {
                key,
                value,
                element_index: Some(element_index),
                source,
            } => write!(
                f,
                "failed to parse zero-based element {element_index} of {key} ({value:?}) as a float: {source}"
            ),
            Self::InvalidEnvInteger { key, value, source } => write!(
                f,
                "failed to parse {key}={value:?} as an i64 integer: {source}"
            ),
            Self::InvalidEnvVector {
                key,
                expected,
                actual,
            } => write!(
                f,
                "{key} must contain {expected} comma-separated values, got {actual}"
            ),
            Self::BasaltMissingLeftCamera { path } => write!(
                f,
                "Basalt calibration `{}` does not contain a left-camera T_imu_cam entry",
                path.display()
            ),
            Self::BasaltNonIsotropicNoise {
                path,
                field,
                values,
            } => write!(
                f,
                "Basalt calibration `{}` has anisotropic `{field}` = [{}, {}, {}], but the runtime model currently requires a single isotropic scalar",
                path.display(),
                values[0],
                values[1],
                values[2]
            ),
            Self::BasaltInvalidQuaternion { path, field } => write!(
                f,
                "Basalt calibration `{}` has an invalid quaternion in `{field}`",
                path.display()
            ),
            Self::BasaltInvalidPose { path, source } => write!(
                f,
                "Basalt calibration `{}` has an invalid camera pose: {source}",
                path.display()
            ),
            Self::BasaltUnsupportedBiasModel {
                path,
                field,
                trailing_terms,
            } => write!(
                f,
                "Basalt calibration `{}` has non-zero higher-order `{field}` terms {:?}; refusing to collapse them into a simple 3-vector bias",
                path.display(),
                trailing_terms
            ),
        }
    }
}

impl std::error::Error for RuntimeImuCalibrationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Environment { source } => Some(source),
            Self::InvalidEnvFloat { source, .. } => Some(source),
            Self::InvalidEnvInteger { source, .. } => Some(source),
            Self::BasaltInvalidPose { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub fn apply_runtime_imu_calibration_override(
    calibration: &Calibration,
) -> Result<Calibration, RuntimeImuCalibrationError> {
    let Some(override_imu) = load_runtime_imu_calibration_from_env()? else {
        return Ok(calibration.clone());
    };
    let mut merged = calibration.clone();
    merged.imu = Some(override_imu);
    Ok(merged)
}

pub fn load_runtime_imu_calibration_from_env()
-> Result<Option<ImuCalibration>, RuntimeImuCalibrationError> {
    let RuntimeImuEnvironment {
        calibration_file,
        direct,
    } = RuntimeImuEnvironment::read()?;
    let has_direct = direct.has_any();

    if calibration_file.is_some() && has_direct {
        return Err(RuntimeImuCalibrationError::ConflictingSources {
            file_env: IMU_CALIBRATION_FILE_ENV,
            direct_keys_present: direct.present_keys(),
        });
    }

    if let Some(path) = calibration_file {
        return load_runtime_imu_calibration_from_path(&path).map(Some);
    }

    if has_direct {
        return direct.into_calibration().map(Some);
    }

    Ok(None)
}

fn load_runtime_imu_calibration_from_path(
    path: &Path,
) -> Result<ImuCalibration, RuntimeImuCalibrationError> {
    let bytes = std::fs::read(path).map_err(|source| RuntimeImuCalibrationError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let value = serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|source| {
        RuntimeImuCalibrationError::Json {
            path: path.to_path_buf(),
            source,
        }
    })?;

    if value.get("noise").is_some() && value.get("extrinsics").is_some() {
        return serde_json::from_value(value).map_err(|source| RuntimeImuCalibrationError::Json {
            path: path.to_path_buf(),
            source,
        });
    }

    if value.get("left").is_some() && value.get("right").is_some() {
        let calibration: Calibration =
            serde_json::from_value(value).map_err(|source| RuntimeImuCalibrationError::Json {
                path: path.to_path_buf(),
                source,
            })?;
        return calibration.imu.ok_or_else(|| {
            RuntimeImuCalibrationError::MissingImuCalibrationBlock {
                path: path.to_path_buf(),
            }
        });
    }

    if value.get("value0").is_some() {
        let basalt: BasaltCalibrationRoot =
            serde_json::from_value(value).map_err(|source| RuntimeImuCalibrationError::Json {
                path: path.to_path_buf(),
                source,
            })?;
        return basalt_into_imu_calibration(path, &basalt);
    }

    Err(RuntimeImuCalibrationError::UnsupportedFileFormat {
        path: path.to_path_buf(),
    })
}

#[derive(Debug)]
struct RuntimeImuEnvironment {
    calibration_file: Option<PathBuf>,
    direct: DirectImuEnvironment,
}

impl RuntimeImuEnvironment {
    fn read() -> Result<Self, RuntimeImuCalibrationError> {
        Ok(Self {
            calibration_file: std::env::var_os(IMU_CALIBRATION_FILE_ENV).map(PathBuf::from),
            direct: DirectImuEnvironment::read()?,
        })
    }
}

#[derive(Debug)]
struct DirectImuEnvironment {
    rotation: CapturedEnvValue,
    translation: CapturedEnvValue,
    accel_noise_density: CapturedEnvValue,
    gyro_noise_density: CapturedEnvValue,
    accel_random_walk: CapturedEnvValue,
    gyro_random_walk: CapturedEnvValue,
    time_offset_ns: CapturedEnvValue,
    gravity_magnitude_mps2: CapturedEnvValue,
    initial_accel_bias: CapturedEnvValue,
    initial_gyro_bias: CapturedEnvValue,
}

impl DirectImuEnvironment {
    fn read() -> Result<Self, RuntimeImuCalibrationError> {
        Ok(Self {
            rotation: CapturedEnvValue::read(IMU_ROTATION_ENV)?,
            translation: CapturedEnvValue::read(IMU_TRANSLATION_ENV)?,
            accel_noise_density: CapturedEnvValue::read(IMU_ACCEL_NOISE_ENV)?,
            gyro_noise_density: CapturedEnvValue::read(IMU_GYRO_NOISE_ENV)?,
            accel_random_walk: CapturedEnvValue::read(IMU_ACCEL_RW_ENV)?,
            gyro_random_walk: CapturedEnvValue::read(IMU_GYRO_RW_ENV)?,
            time_offset_ns: CapturedEnvValue::read(IMU_TIME_OFFSET_ENV)?,
            gravity_magnitude_mps2: CapturedEnvValue::read(IMU_GRAVITY_ENV)?,
            initial_accel_bias: CapturedEnvValue::read(IMU_INITIAL_ACCEL_BIAS_ENV)?,
            initial_gyro_bias: CapturedEnvValue::read(IMU_INITIAL_GYRO_BIAS_ENV)?,
        })
    }

    fn entries(&self) -> [&CapturedEnvValue; 10] {
        [
            &self.rotation,
            &self.translation,
            &self.accel_noise_density,
            &self.gyro_noise_density,
            &self.accel_random_walk,
            &self.gyro_random_walk,
            &self.time_offset_ns,
            &self.gravity_magnitude_mps2,
            &self.initial_accel_bias,
            &self.initial_gyro_bias,
        ]
    }

    fn has_any(&self) -> bool {
        self.entries().into_iter().any(|value| value.is_present())
    }

    fn present_keys(&self) -> Vec<&'static str> {
        self.entries()
            .into_iter()
            .filter_map(|value| value.is_present().then_some(value.key()))
            .collect()
    }

    fn into_calibration(self) -> Result<ImuCalibration, RuntimeImuCalibrationError> {
        let rotation = parse_required_matrix3(&self.rotation)?;
        let translation = parse_required_vec3(&self.translation)?;
        let accel_noise_density = parse_required_f64(&self.accel_noise_density)?;
        let gyro_noise_density = parse_required_f64(&self.gyro_noise_density)?;
        let accel_random_walk = parse_required_f64(&self.accel_random_walk)?;
        let gyro_random_walk = parse_required_f64(&self.gyro_random_walk)?;
        let time_offset_ns = parse_optional_i64(&self.time_offset_ns)?.unwrap_or(0);
        let gravity_magnitude_mps2 =
            parse_optional_f64(&self.gravity_magnitude_mps2)?.unwrap_or(9.81);
        let initial_accel_bias = parse_optional_vec3(&self.initial_accel_bias)?;
        let initial_gyro_bias = parse_optional_vec3(&self.initial_gyro_bias)?;
        if initial_accel_bias.is_some() != initial_gyro_bias.is_some() {
            let missing = if initial_accel_bias.is_none() {
                IMU_INITIAL_ACCEL_BIAS_ENV
            } else {
                IMU_INITIAL_GYRO_BIAS_ENV
            };
            return Err(RuntimeImuCalibrationError::MissingEnv { key: missing });
        }

        Ok(ImuCalibration {
            noise: ImuNoiseMeta {
                accel_noise_density,
                gyro_noise_density,
                accel_random_walk,
                gyro_random_walk,
            },
            extrinsics: ImuExtrinsicsMeta {
                rotation,
                translation,
                time_offset_ns,
            },
            gravity_magnitude_mps2,
            initial_accel_bias,
            initial_gyro_bias,
        })
    }
}

#[derive(Debug)]
struct CapturedEnvValue {
    key: &'static str,
    raw: Option<String>,
}

impl CapturedEnvValue {
    fn read(key: &'static str) -> Result<Self, RuntimeImuCalibrationError> {
        let raw = crate::env::try_env_string(key)
            .map_err(|source| RuntimeImuCalibrationError::Environment { source })?;
        Ok(Self { key, raw })
    }

    fn key(&self) -> &'static str {
        self.key
    }

    fn is_present(&self) -> bool {
        self.raw.is_some()
    }

    fn raw(&self) -> Option<&str> {
        self.raw.as_deref()
    }

    fn required(&self) -> Result<&str, RuntimeImuCalibrationError> {
        self.raw()
            .ok_or(RuntimeImuCalibrationError::MissingEnv { key: self.key })
    }
}

fn parse_required_f64(value: &CapturedEnvValue) -> Result<f64, RuntimeImuCalibrationError> {
    parse_f64(value.key(), value.required()?, None)
}

fn parse_optional_f64(value: &CapturedEnvValue) -> Result<Option<f64>, RuntimeImuCalibrationError> {
    value
        .raw()
        .map(|raw| parse_f64(value.key(), raw, None))
        .transpose()
}

fn parse_optional_i64(value: &CapturedEnvValue) -> Result<Option<i64>, RuntimeImuCalibrationError> {
    value
        .raw()
        .map(|raw| {
            raw.parse::<i64>()
                .map_err(|source| RuntimeImuCalibrationError::InvalidEnvInteger {
                    key: value.key(),
                    value: raw.to_string(),
                    source,
                })
        })
        .transpose()
}

fn parse_f64(
    key: &'static str,
    raw: &str,
    element_index: Option<usize>,
) -> Result<f64, RuntimeImuCalibrationError> {
    raw.parse::<f64>()
        .map_err(|source| RuntimeImuCalibrationError::InvalidEnvFloat {
            key,
            value: raw.to_string(),
            element_index,
            source,
        })
}

fn parse_csv_f64<const N: usize>(
    key: &'static str,
    raw: &str,
) -> Result<[f64; N], RuntimeImuCalibrationError> {
    let mut values = [0.0; N];
    let mut actual = 0_usize;
    for (element_index, part) in raw.split(',').map(str::trim).enumerate() {
        let value = parse_f64(key, part, Some(element_index))?;
        if let Some(slot) = values.get_mut(element_index) {
            *slot = value;
        }
        actual = element_index.saturating_add(1);
    }
    if actual != N {
        return Err(RuntimeImuCalibrationError::InvalidEnvVector {
            key,
            expected: N,
            actual,
        });
    }
    Ok(values)
}

fn parse_required_vec3(value: &CapturedEnvValue) -> Result<[f64; 3], RuntimeImuCalibrationError> {
    parse_csv_f64(value.key(), value.required()?)
}

fn parse_optional_vec3(
    value: &CapturedEnvValue,
) -> Result<Option<[f64; 3]>, RuntimeImuCalibrationError> {
    value
        .raw()
        .map(|raw| parse_csv_f64(value.key(), raw))
        .transpose()
}

fn parse_required_matrix3(
    value: &CapturedEnvValue,
) -> Result<[[f64; 3]; 3], RuntimeImuCalibrationError> {
    let [m00, m01, m02, m10, m11, m12, m20, m21, m22] =
        parse_csv_f64(value.key(), value.required()?)?;
    Ok([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
}

#[derive(Deserialize)]
struct BasaltCalibrationRoot {
    value0: BasaltCalibrationValue,
}

#[derive(Deserialize)]
struct BasaltCalibrationValue {
    #[serde(rename = "T_imu_cam")]
    t_imu_cam: Vec<BasaltPose>,
    accel_noise_std: [f64; 3],
    gyro_noise_std: [f64; 3],
    accel_bias_std: [f64; 3],
    gyro_bias_std: [f64; 3],
    calib_accel_bias: Vec<f64>,
    calib_gyro_bias: Vec<f64>,
    cam_time_offset_ns: i64,
}

#[derive(Deserialize)]
struct BasaltPose {
    px: f64,
    py: f64,
    pz: f64,
    qx: f64,
    qy: f64,
    qz: f64,
    qw: f64,
}

fn basalt_into_imu_calibration(
    path: &Path,
    basalt: &BasaltCalibrationRoot,
) -> Result<ImuCalibration, RuntimeImuCalibrationError> {
    let left_camera = basalt.value0.t_imu_cam.first().ok_or_else(|| {
        RuntimeImuCalibrationError::BasaltMissingLeftCamera {
            path: path.to_path_buf(),
        }
    })?;
    let rotation = rotation_from_quaternion(
        path,
        "value0.T_imu_cam[0]",
        left_camera.qx,
        left_camera.qy,
        left_camera.qz,
        left_camera.qw,
    )?;
    let t_imu_cam = Pose64::try_from_rt(rotation, [left_camera.px, left_camera.py, left_camera.pz])
        .map_err(|source| RuntimeImuCalibrationError::BasaltInvalidPose {
            path: path.to_path_buf(),
            source,
        })?;
    let t_cam_imu = t_imu_cam.try_inverse().map_err(|source| {
        RuntimeImuCalibrationError::BasaltInvalidPose {
            path: path.to_path_buf(),
            source,
        }
    })?;

    Ok(ImuCalibration {
        noise: ImuNoiseMeta {
            accel_noise_density: isotropic_basalt_noise(
                path,
                "value0.accel_noise_std",
                basalt.value0.accel_noise_std,
            )?,
            gyro_noise_density: isotropic_basalt_noise(
                path,
                "value0.gyro_noise_std",
                basalt.value0.gyro_noise_std,
            )?,
            accel_random_walk: isotropic_basalt_noise(
                path,
                "value0.accel_bias_std",
                basalt.value0.accel_bias_std,
            )?,
            gyro_random_walk: isotropic_basalt_noise(
                path,
                "value0.gyro_bias_std",
                basalt.value0.gyro_bias_std,
            )?,
        },
        extrinsics: ImuExtrinsicsMeta {
            rotation: t_cam_imu.rotation(),
            translation: t_cam_imu.translation(),
            time_offset_ns: basalt.value0.cam_time_offset_ns,
        },
        gravity_magnitude_mps2: 9.81,
        initial_accel_bias: Some(basalt_bias_vector(
            path,
            "value0.calib_accel_bias",
            &basalt.value0.calib_accel_bias,
        )?),
        initial_gyro_bias: Some(basalt_bias_vector(
            path,
            "value0.calib_gyro_bias",
            &basalt.value0.calib_gyro_bias,
        )?),
    })
}

fn isotropic_basalt_noise(
    path: &Path,
    field: &'static str,
    values: [f64; 3],
) -> Result<f64, RuntimeImuCalibrationError> {
    let max_diff = values
        .iter()
        .map(|value| (value - values[0]).abs())
        .fold(0.0_f64, f64::max);
    if max_diff > 1e-12 {
        return Err(RuntimeImuCalibrationError::BasaltNonIsotropicNoise {
            path: path.to_path_buf(),
            field,
            values,
        });
    }
    Ok(values[0])
}

fn basalt_bias_vector(
    path: &Path,
    field: &'static str,
    values: &[f64],
) -> Result<[f64; 3], RuntimeImuCalibrationError> {
    if values.len() < 3 {
        return Err(RuntimeImuCalibrationError::BasaltUnsupportedBiasModel {
            path: path.to_path_buf(),
            field,
            trailing_terms: values.to_vec(),
        });
    }
    let trailing = values[3..].to_vec();
    if trailing.iter().any(|value| value.abs() > 1e-12) {
        return Err(RuntimeImuCalibrationError::BasaltUnsupportedBiasModel {
            path: path.to_path_buf(),
            field,
            trailing_terms: trailing,
        });
    }
    Ok([values[0], values[1], values[2]])
}

fn rotation_from_quaternion(
    path: &Path,
    field: &'static str,
    qx: f64,
    qy: f64,
    qz: f64,
    qw: f64,
) -> Result<[[f64; 3]; 3], RuntimeImuCalibrationError> {
    let norm = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(RuntimeImuCalibrationError::BasaltInvalidQuaternion {
            path: path.to_path_buf(),
            field,
        });
    }
    let qx = qx / norm;
    let qy = qy / norm;
    let qz = qz / norm;
    let qw = qw / norm;

    Ok([
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ])
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use std::error::Error as _;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn clear_runtime_imu_env() {
        for key in [
            IMU_CALIBRATION_FILE_ENV,
            IMU_ROTATION_ENV,
            IMU_TRANSLATION_ENV,
            IMU_ACCEL_NOISE_ENV,
            IMU_GYRO_NOISE_ENV,
            IMU_ACCEL_RW_ENV,
            IMU_GYRO_RW_ENV,
            IMU_TIME_OFFSET_ENV,
            IMU_GRAVITY_ENV,
            IMU_INITIAL_ACCEL_BIAS_ENV,
            IMU_INITIAL_GYRO_BIAS_ENV,
        ] {
            // Tests serialize environment mutation with a process-wide mutex.
            unsafe { std::env::remove_var(key) };
        }
    }

    #[test]
    fn numeric_env_errors_preserve_the_parser_source() {
        let invalid_float = CapturedEnvValue {
            key: IMU_ACCEL_NOISE_ENV,
            raw: Some("not-a-float".to_string()),
        };
        let float_error = parse_required_f64(&invalid_float).expect_err("invalid float must fail");
        assert!(matches!(
            &float_error,
            RuntimeImuCalibrationError::InvalidEnvFloat {
                key: IMU_ACCEL_NOISE_ENV,
                element_index: None,
                ..
            }
        ));
        assert!(float_error.source().is_some());

        let invalid_integer = CapturedEnvValue {
            key: IMU_TIME_OFFSET_ENV,
            raw: Some("not-an-integer".to_string()),
        };
        let integer_error =
            parse_optional_i64(&invalid_integer).expect_err("invalid integer must fail");
        assert!(matches!(
            &integer_error,
            RuntimeImuCalibrationError::InvalidEnvInteger {
                key: IMU_TIME_OFFSET_ENV,
                ..
            }
        ));
        assert!(integer_error.source().is_some());
    }

    #[test]
    fn fixed_vector_parser_rejects_empty_elements_with_exact_provenance() {
        let invalid_vector = CapturedEnvValue {
            key: IMU_TRANSLATION_ENV,
            raw: Some("1,,3".to_string()),
        };
        let error = parse_required_vec3(&invalid_vector).expect_err("empty component must fail");
        assert!(matches!(
            &error,
            RuntimeImuCalibrationError::InvalidEnvFloat {
                key: IMU_TRANSLATION_ENV,
                value,
                element_index: Some(1),
                ..
            } if value.is_empty()
        ));
        assert!(error.source().is_some());
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_direct_env_is_not_treated_as_absent() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt as _;

        let _guard = env_lock().lock().expect("env lock");
        clear_runtime_imu_env();
        // Tests serialize environment mutation with a process-wide mutex.
        unsafe { std::env::set_var(IMU_ROTATION_ENV, OsString::from_vec(vec![0xff])) };

        let error = load_runtime_imu_calibration_from_env().expect_err("non-Unicode direct value");
        assert!(matches!(
            &error,
            RuntimeImuCalibrationError::Environment {
                source: crate::env::EnvError::NonUnicode {
                    key: IMU_ROTATION_ENV,
                    ..
                }
            }
        ));
        let environment = error.source().expect("environment source");
        assert!(environment.source().is_some());
        clear_runtime_imu_env();
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_calibration_file_is_preserved_as_a_path() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt as _;

        let _guard = env_lock().lock().expect("env lock");
        clear_runtime_imu_env();
        let raw_path = OsString::from_vec(b"/tmp/kiko-imu-\xff.json".to_vec());
        let expected_path = PathBuf::from(&raw_path);
        // Tests serialize environment mutation with a process-wide mutex.
        unsafe { std::env::set_var(IMU_CALIBRATION_FILE_ENV, raw_path) };

        let error = load_runtime_imu_calibration_from_env().expect_err("missing non-Unicode path");
        assert!(matches!(
            error,
            RuntimeImuCalibrationError::Io { path, .. } if path == expected_path
        ));
        clear_runtime_imu_env();
    }

    #[test]
    fn direct_env_override_requires_complete_block() {
        let _guard = env_lock().lock().expect("env lock");
        clear_runtime_imu_env();
        // Tests serialize environment mutation with a process-wide mutex.
        unsafe { std::env::set_var(IMU_ROTATION_ENV, "1,0,0,0,1,0,0,0,1") };
        let err = load_runtime_imu_calibration_from_env().expect_err("missing env block");
        assert!(matches!(err, RuntimeImuCalibrationError::MissingEnv { .. }));
        clear_runtime_imu_env();
    }

    #[test]
    fn direct_env_override_builds_imu_calibration() {
        let _guard = env_lock().lock().expect("env lock");
        clear_runtime_imu_env();
        unsafe {
            std::env::set_var(IMU_ROTATION_ENV, "1,0,0,0,1,0,0,0,1");
            std::env::set_var(IMU_TRANSLATION_ENV, "0.01,-0.02,0.03");
            std::env::set_var(IMU_ACCEL_NOISE_ENV, "0.003");
            std::env::set_var(IMU_GYRO_NOISE_ENV, "0.0017");
            std::env::set_var(IMU_ACCEL_RW_ENV, "0.0004");
            std::env::set_var(IMU_GYRO_RW_ENV, "0.00002");
            std::env::set_var(IMU_TIME_OFFSET_ENV, "42");
            std::env::set_var(IMU_GRAVITY_ENV, "9.81");
            std::env::set_var(IMU_INITIAL_ACCEL_BIAS_ENV, "0.1,0.2,0.3");
            std::env::set_var(IMU_INITIAL_GYRO_BIAS_ENV, "0.01,0.02,0.03");
        }

        let imu = load_runtime_imu_calibration_from_env()
            .expect("load env calibration")
            .expect("imu override");
        assert_eq!(imu.extrinsics.translation, [0.01, -0.02, 0.03]);
        assert_eq!(imu.extrinsics.time_offset_ns, 42);
        assert_eq!(imu.initial_accel_bias, Some([0.1, 0.2, 0.3]));
        assert_eq!(imu.initial_gyro_bias, Some([0.01, 0.02, 0.03]));
        clear_runtime_imu_env();
    }

    #[test]
    fn file_and_direct_env_sources_conflict() {
        let _guard = env_lock().lock().expect("env lock");
        clear_runtime_imu_env();
        unsafe {
            std::env::set_var(IMU_CALIBRATION_FILE_ENV, "/tmp/imu.json");
            std::env::set_var(IMU_ROTATION_ENV, "1,0,0,0,1,0,0,0,1");
        }
        let err = load_runtime_imu_calibration_from_env().expect_err("conflicting sources");
        assert!(matches!(
            err,
            RuntimeImuCalibrationError::ConflictingSources { .. }
        ));
        clear_runtime_imu_env();
    }

    #[test]
    fn basalt_override_inverts_left_camera_extrinsics_and_extracts_biases() {
        let path = Path::new("/tmp/basalt.json");
        let json = serde_json::json!({
            "value0": {
                "T_imu_cam": [{
                    "px": 0.1,
                    "py": -0.2,
                    "pz": 0.3,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0
                }],
                "accel_noise_std": [0.003, 0.003, 0.003],
                "gyro_noise_std": [0.0017, 0.0017, 0.0017],
                "accel_bias_std": [0.0004, 0.0004, 0.0004],
                "gyro_bias_std": [0.00002, 0.00002, 0.00002],
                "calib_accel_bias": [0.5, -0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "calib_gyro_bias": [0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "cam_time_offset_ns": 7
            }
        });
        let basalt: BasaltCalibrationRoot =
            serde_json::from_value(json).expect("basalt calibration");
        let imu = basalt_into_imu_calibration(path, &basalt).expect("basalt override");

        assert_eq!(imu.extrinsics.translation, [-0.1, 0.2, -0.3]);
        assert_eq!(imu.extrinsics.time_offset_ns, 7);
        assert_eq!(imu.initial_accel_bias, Some([0.5, -0.1, 0.2]));
        assert_eq!(imu.initial_gyro_bias, Some([0.01, -0.02, 0.03]));
    }

    #[test]
    fn apply_override_preserves_visual_calibration_and_replaces_imu_block() {
        let _guard = env_lock().lock().expect("env lock");
        clear_runtime_imu_env();
        unsafe {
            std::env::set_var(IMU_ROTATION_ENV, "1,0,0,0,1,0,0,0,1");
            std::env::set_var(IMU_TRANSLATION_ENV, "0.01,-0.02,0.03");
            std::env::set_var(IMU_ACCEL_NOISE_ENV, "0.003");
            std::env::set_var(IMU_GYRO_NOISE_ENV, "0.0017");
            std::env::set_var(IMU_ACCEL_RW_ENV, "0.0004");
            std::env::set_var(IMU_GYRO_RW_ENV, "0.00002");
        }

        let calibration = Calibration {
            left: crate::dataset::CameraIntrinsics {
                fx: 10.0,
                fy: 11.0,
                cx: 12.0,
                cy: 13.0,
                width: 640,
                height: 480,
            },
            right: crate::dataset::CameraIntrinsics {
                fx: 14.0,
                fy: 15.0,
                cx: 16.0,
                cy: 17.0,
                width: 640,
                height: 480,
            },
            baseline_m: 0.075,
            rectified: true,
            imu: None,
        };
        let merged =
            apply_runtime_imu_calibration_override(&calibration).expect("merged calibration");
        assert_eq!(merged.left.fx, calibration.left.fx);
        assert_eq!(merged.right.fx, calibration.right.fx);
        assert!(merged.imu.is_some());
        clear_runtime_imu_env();
    }
}
