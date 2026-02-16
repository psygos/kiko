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
