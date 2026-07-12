#[derive(Debug)]
pub enum EnvError {
    NonUnicode {
        key: &'static str,
        source: std::env::VarError,
    },
    InvalidInteger {
        key: &'static str,
        value: String,
        source: std::num::ParseIntError,
    },
    InvalidFloat {
        key: &'static str,
        value: String,
        source: std::num::ParseFloatError,
    },
    InvalidBool {
        key: &'static str,
        value: String,
    },
}

impl std::fmt::Display for EnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonUnicode { key, .. } => write!(f, "environment variable {key} is not Unicode"),
            Self::InvalidInteger { key, value, .. } => {
                write!(f, "environment variable {key} is not an integer: {value:?}")
            }
            Self::InvalidFloat { key, value, .. } => {
                write!(f, "environment variable {key} is not a float: {value:?}")
            }
            Self::InvalidBool { key, value } => {
                write!(f, "environment variable {key} is not a boolean: {value:?}")
            }
        }
    }
}

impl std::error::Error for EnvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NonUnicode { source, .. } => Some(source),
            Self::InvalidInteger { source, .. } => Some(source),
            Self::InvalidFloat { source, .. } => Some(source),
            Self::InvalidBool { .. } => None,
        }
    }
}

fn raw_env(key: &'static str) -> Result<Option<String>, EnvError> {
    match std::env::var(key) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(source) => Err(EnvError::NonUnicode { key, source }),
    }
}

pub fn try_env_usize(key: &'static str) -> Result<Option<usize>, EnvError> {
    let Some(value) = raw_env(key)? else {
        return Ok(None);
    };
    value
        .parse()
        .map(Some)
        .map_err(|source| EnvError::InvalidInteger { key, value, source })
}

pub fn try_env_f32(key: &'static str) -> Result<Option<f32>, EnvError> {
    let Some(value) = raw_env(key)? else {
        return Ok(None);
    };
    value
        .parse()
        .map(Some)
        .map_err(|source| EnvError::InvalidFloat { key, value, source })
}

pub fn try_env_f64(key: &'static str) -> Result<Option<f64>, EnvError> {
    let Some(value) = raw_env(key)? else {
        return Ok(None);
    };
    value
        .parse()
        .map(Some)
        .map_err(|source| EnvError::InvalidFloat { key, value, source })
}

pub fn try_env_bool(key: &'static str) -> Result<Option<bool>, EnvError> {
    let Some(value) = raw_env(key)? else {
        return Ok(None);
    };
    match value.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => Err(EnvError::InvalidBool { key, value }),
    }
}

fn env_parse<T>(key: &str) -> Option<T>
where
    T: std::str::FromStr,
{
    let raw = std::env::var(key).ok()?;
    match raw.parse::<T>() {
        Ok(value) => Some(value),
        Err(_) => {
            eprintln!("invalid {key}={raw}, ignoring");
            None
        }
    }
}

pub fn env_usize(key: &str) -> Option<usize> {
    env_parse(key)
}

pub fn env_f32(key: &str) -> Option<f32> {
    env_parse(key)
}

pub fn env_f64(key: &str) -> Option<f64> {
    env_parse(key)
}

pub fn env_i64(key: &str) -> Option<i64> {
    env_parse(key)
}

pub fn env_bool(key: &str) -> Option<bool> {
    let raw = std::env::var(key).ok()?;
    match raw.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => {
            eprintln!("invalid {key}={raw}, ignoring");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn strict_environment_parse_errors_preserve_sources() {
        let source = "not-an-integer"
            .parse::<usize>()
            .expect_err("invalid integer");
        let error = EnvError::InvalidInteger {
            key: "KIKO_TEST",
            value: "not-an-integer".to_string(),
            source,
        };
        assert!(error.source().is_some());

        let source = "not-a-float".parse::<f32>().expect_err("invalid float");
        let error = EnvError::InvalidFloat {
            key: "KIKO_TEST",
            value: "not-a-float".to_string(),
            source,
        };
        assert!(error.source().is_some());
    }
}
