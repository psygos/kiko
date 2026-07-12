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

const DIRECT_ENV_KEYS: [&str; 10] = [
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
];

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
    InvalidEnvScalar {
        key: &'static str,
        value: String,
        message: String,
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
            Self::InvalidEnvScalar {
                key,
                value,
                message,
            } => {
                write!(f, "failed to parse {key}={value}: {message}")
            }
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
    let calibration_file = std::env::var(IMU_CALIBRATION_FILE_ENV).ok();
    let direct_keys_present = DIRECT_ENV_KEYS
        .into_iter()
        .filter(|key| std::env::var(key).is_ok())
        .collect::<Vec<_>>();

    if calibration_file.is_some() && !direct_keys_present.is_empty() {
        return Err(RuntimeImuCalibrationError::ConflictingSources {
            file_env: IMU_CALIBRATION_FILE_ENV,
            direct_keys_present,
        });
    }

    if let Some(path) = calibration_file {
        return load_runtime_imu_calibration_from_path(Path::new(&path)).map(Some);
    }

    if !direct_keys_present.is_empty() {
        return load_runtime_imu_calibration_from_direct_env().map(Some);
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

fn load_runtime_imu_calibration_from_direct_env()
-> Result<ImuCalibration, RuntimeImuCalibrationError> {
    let rotation = parse_matrix3_env(IMU_ROTATION_ENV)?;
    let translation = parse_vec3_env(IMU_TRANSLATION_ENV)?;
    let accel_noise_density = parse_scalar_env(IMU_ACCEL_NOISE_ENV)?;
    let gyro_noise_density = parse_scalar_env(IMU_GYRO_NOISE_ENV)?;
    let accel_random_walk = parse_scalar_env(IMU_ACCEL_RW_ENV)?;
    let gyro_random_walk = parse_scalar_env(IMU_GYRO_RW_ENV)?;

    let initial_accel_bias = match std::env::var(IMU_INITIAL_ACCEL_BIAS_ENV).ok() {
        Some(_) => Some(parse_vec3_env(IMU_INITIAL_ACCEL_BIAS_ENV)?),
        None => None,
    };
    let initial_gyro_bias = match std::env::var(IMU_INITIAL_GYRO_BIAS_ENV).ok() {
        Some(_) => Some(parse_vec3_env(IMU_INITIAL_GYRO_BIAS_ENV)?),
        None => None,
    };
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
            time_offset_ns: parse_optional_i64_env(IMU_TIME_OFFSET_ENV)?.unwrap_or(0),
        },
        gravity_magnitude_mps2: parse_optional_scalar_env(IMU_GRAVITY_ENV)?.unwrap_or(9.81),
        initial_accel_bias,
        initial_gyro_bias,
    })
}

fn parse_scalar_env(key: &'static str) -> Result<f64, RuntimeImuCalibrationError> {
    let raw = std::env::var(key).map_err(|_| RuntimeImuCalibrationError::MissingEnv { key })?;
    parse_scalar_value(key, &raw)
}

fn parse_optional_scalar_env(key: &'static str) -> Result<Option<f64>, RuntimeImuCalibrationError> {
    match std::env::var(key) {
        Ok(raw) => parse_scalar_value(key, &raw).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(raw)) => {
            Err(RuntimeImuCalibrationError::InvalidEnvScalar {
                key,
                value: format!("{raw:?}"),
                message: "value is not valid unicode".to_string(),
            })
        }
    }
}

fn parse_optional_i64_env(key: &'static str) -> Result<Option<i64>, RuntimeImuCalibrationError> {
    match std::env::var(key) {
        Ok(raw) => raw.parse::<i64>().map(Some).map_err(|err| {
            RuntimeImuCalibrationError::InvalidEnvScalar {
                key,
                value: raw,
                message: err.to_string(),
            }
        }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(raw)) => {
            Err(RuntimeImuCalibrationError::InvalidEnvScalar {
                key,
                value: format!("{raw:?}"),
                message: "value is not valid unicode".to_string(),
            })
        }
    }
}

fn parse_scalar_value(key: &'static str, raw: &str) -> Result<f64, RuntimeImuCalibrationError> {
    raw.parse::<f64>()
        .map_err(|err| RuntimeImuCalibrationError::InvalidEnvScalar {
            key,
            value: raw.to_string(),
            message: err.to_string(),
        })
}

fn parse_vec3_env(key: &'static str) -> Result<[f64; 3], RuntimeImuCalibrationError> {
    let raw = std::env::var(key).map_err(|_| RuntimeImuCalibrationError::MissingEnv { key })?;
    let values = parse_csv_f64(key, &raw, 3)?;
    Ok([values[0], values[1], values[2]])
}

fn parse_matrix3_env(key: &'static str) -> Result<[[f64; 3]; 3], RuntimeImuCalibrationError> {
    let raw = std::env::var(key).map_err(|_| RuntimeImuCalibrationError::MissingEnv { key })?;
    let values = parse_csv_f64(key, &raw, 9)?;
    Ok([
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
    ])
}

fn parse_csv_f64(
    key: &'static str,
    raw: &str,
    expected: usize,
) -> Result<Vec<f64>, RuntimeImuCalibrationError> {
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<f64>()
                .map_err(|err| RuntimeImuCalibrationError::InvalidEnvScalar {
                    key,
                    value: raw.to_string(),
                    message: err.to_string(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != expected {
        return Err(RuntimeImuCalibrationError::InvalidEnvVector {
            key,
            expected,
            actual: values.len(),
        });
    }
    Ok(values)
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
    let t_cam_imu = t_imu_cam.inverse();

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
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn clear_runtime_imu_env() {
        for key in [
            IMU_CALIBRATION_FILE_ENV,
            DIRECT_ENV_KEYS[0],
            DIRECT_ENV_KEYS[1],
            DIRECT_ENV_KEYS[2],
            DIRECT_ENV_KEYS[3],
            DIRECT_ENV_KEYS[4],
            DIRECT_ENV_KEYS[5],
            DIRECT_ENV_KEYS[6],
            DIRECT_ENV_KEYS[7],
            DIRECT_ENV_KEYS[8],
            DIRECT_ENV_KEYS[9],
        ] {
            // Tests serialize environment mutation with a process-wide mutex.
            unsafe { std::env::remove_var(key) };
        }
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
