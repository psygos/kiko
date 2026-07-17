use std::ffi::OsString;
use std::num::{ParseFloatError, ParseIntError};

#[derive(Debug)]
pub enum EnvError {
    NonUnicode {
        key: String,
        value: OsString,
    },
    InvalidBool {
        key: String,
        value: String,
    },
    InvalidF32 {
        key: String,
        value: String,
        source: ParseFloatError,
    },
    InvalidF64 {
        key: String,
        value: String,
        source: ParseFloatError,
    },
    InvalidU32 {
        key: String,
        value: String,
        source: ParseIntError,
    },
    InvalidU64 {
        key: String,
        value: String,
        source: ParseIntError,
    },
    InvalidUsize {
        key: String,
        value: String,
        source: ParseIntError,
    },
}

impl std::fmt::Display for EnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonUnicode { key, .. } => {
                write!(f, "environment variable {key} contains non-Unicode data")
            }
            Self::InvalidBool { key, .. } => {
                write!(f, "environment variable {key} is not a recognized boolean")
            }
            Self::InvalidF32 { key, source, .. } => {
                write!(f, "environment variable {key} is not a valid f32: {source}")
            }
            Self::InvalidF64 { key, source, .. } => {
                write!(f, "environment variable {key} is not a valid f64: {source}")
            }
            Self::InvalidU32 { key, source, .. } => {
                write!(f, "environment variable {key} is not a valid u32: {source}")
            }
            Self::InvalidU64 { key, source, .. } => {
                write!(f, "environment variable {key} is not a valid u64: {source}")
            }
            Self::InvalidUsize { key, source, .. } => {
                write!(
                    f,
                    "environment variable {key} is not a valid usize: {source}"
                )
            }
        }
    }
}

impl std::error::Error for EnvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidF32 { source, .. } | Self::InvalidF64 { source, .. } => Some(source),
            Self::InvalidU32 { source, .. }
            | Self::InvalidU64 { source, .. }
            | Self::InvalidUsize { source, .. } => Some(source),
            Self::NonUnicode { .. } | Self::InvalidBool { .. } => None,
        }
    }
}

pub fn env_string(key: &str) -> Result<Option<String>, EnvError> {
    parse_string_result(key, std::env::var(key))
}

fn parse_string_result(
    key: &str,
    result: Result<String, std::env::VarError>,
) -> Result<Option<String>, EnvError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(value)) => Err(EnvError::NonUnicode {
            key: key.to_owned(),
            value,
        }),
    }
}

fn env_parse<T>(
    key: &str,
    invalid: impl FnOnce(String, String, T::Err) -> EnvError,
) -> Result<Option<T>, EnvError>
where
    T: std::str::FromStr,
{
    let Some(raw) = env_string(key)? else {
        return Ok(None);
    };
    parse_value(key, raw, invalid).map(Some)
}

fn parse_value<T>(
    key: &str,
    raw: String,
    invalid: impl FnOnce(String, String, T::Err) -> EnvError,
) -> Result<T, EnvError>
where
    T: std::str::FromStr,
{
    match raw.trim().parse::<T>() {
        Ok(value) => Ok(value),
        Err(source) => Err(invalid(key.to_owned(), raw, source)),
    }
}

pub fn env_usize(key: &str) -> Result<Option<usize>, EnvError> {
    env_parse(key, |key, value, source| EnvError::InvalidUsize {
        key,
        value,
        source,
    })
}

pub fn env_u64(key: &str) -> Result<Option<u64>, EnvError> {
    env_parse(key, |key, value, source| EnvError::InvalidU64 {
        key,
        value,
        source,
    })
}

pub fn env_u32(key: &str) -> Result<Option<u32>, EnvError> {
    env_parse(key, |key, value, source| EnvError::InvalidU32 {
        key,
        value,
        source,
    })
}

pub fn env_f32(key: &str) -> Result<Option<f32>, EnvError> {
    env_parse(key, |key, value, source| EnvError::InvalidF32 {
        key,
        value,
        source,
    })
}

pub fn env_f64(key: &str) -> Result<Option<f64>, EnvError> {
    env_parse(key, |key, value, source| EnvError::InvalidF64 {
        key,
        value,
        source,
    })
}

pub fn env_bool(key: &str) -> Result<Option<bool>, EnvError> {
    let Some(raw) = env_string(key)? else {
        return Ok(None);
    };
    parse_bool_value(key, raw).map(Some)
}

fn parse_bool_value(key: &str, raw: String) -> Result<bool, EnvError> {
    let value = raw.trim();
    if value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("on")
    {
        return Ok(true);
    }
    if value == "0"
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("no")
        || value.eq_ignore_ascii_case("off")
    {
        return Ok(false);
    }
    Err(EnvError::InvalidBool {
        key: key.to_owned(),
        value: raw,
    })
}

#[cfg(test)]
mod tests {
    use super::{EnvError, parse_bool_value, parse_string_result, parse_value};
    use std::ffi::OsString;

    #[test]
    fn string_boundary_distinguishes_absent_and_non_unicode_values() {
        assert!(matches!(
            parse_string_result("TEST_STRING", Err(std::env::VarError::NotPresent)),
            Ok(None)
        ));
        assert!(matches!(
            parse_string_result(
                "TEST_STRING",
                Err(std::env::VarError::NotUnicode(OsString::from("bytes"))),
            ),
            Err(EnvError::NonUnicode { key, value })
                if key == "TEST_STRING" && value == "bytes"
        ));
    }

    #[test]
    fn boolean_parser_accepts_documented_spellings_without_normalizing() {
        for raw in ["1", "true", "TRUE", " yes ", "On"] {
            assert!(matches!(
                parse_bool_value("TEST_BOOL", raw.to_owned()),
                Ok(true)
            ));
        }
        for raw in ["0", "false", "FALSE", " no ", "Off"] {
            assert!(matches!(
                parse_bool_value("TEST_BOOL", raw.to_owned()),
                Ok(false)
            ));
        }
    }

    #[test]
    fn boolean_parser_rejects_ambiguous_values() {
        assert!(matches!(
            parse_bool_value("TEST_BOOL", "enabled".to_owned()),
            Err(EnvError::InvalidBool { key, value })
                if key == "TEST_BOOL" && value == "enabled"
        ));
    }

    #[test]
    fn numeric_parsers_distinguish_types_and_retain_sources() {
        let errors = [
            parse_value::<f32>("F32", "invalid".to_owned(), |key, value, source| {
                EnvError::InvalidF32 { key, value, source }
            })
            .unwrap_err(),
            parse_value::<f64>("F64", "invalid".to_owned(), |key, value, source| {
                EnvError::InvalidF64 { key, value, source }
            })
            .unwrap_err(),
            parse_value::<u32>("U32", "invalid".to_owned(), |key, value, source| {
                EnvError::InvalidU32 { key, value, source }
            })
            .unwrap_err(),
            parse_value::<u64>("U64", "invalid".to_owned(), |key, value, source| {
                EnvError::InvalidU64 { key, value, source }
            })
            .unwrap_err(),
            parse_value::<usize>("USIZE", "invalid".to_owned(), |key, value, source| {
                EnvError::InvalidUsize { key, value, source }
            })
            .unwrap_err(),
        ];

        assert!(matches!(&errors[0], EnvError::InvalidF32 { key, .. } if key == "F32"));
        assert!(matches!(&errors[1], EnvError::InvalidF64 { key, .. } if key == "F64"));
        assert!(matches!(&errors[2], EnvError::InvalidU32 { key, .. } if key == "U32"));
        assert!(matches!(&errors[3], EnvError::InvalidU64 { key, .. } if key == "U64"));
        assert!(matches!(&errors[4], EnvError::InvalidUsize { key, .. } if key == "USIZE"));
        for error in &errors {
            assert!(std::error::Error::source(error).is_some());
        }
    }

    #[test]
    fn generic_parser_preserves_whitespace_in_diagnostics() {
        // Exercise the value-to-error mapping without mutating the process-wide
        // environment, which would race the test harness's other threads.
        let result = parse_value::<u32>(
            "TEST_U32",
            "  invalid  ".to_owned(),
            |key, value, source| EnvError::InvalidU32 { key, value, source },
        );
        assert!(matches!(
            result,
            Err(EnvError::InvalidU32 { value, .. }) if value == "  invalid  "
        ));
    }
}
