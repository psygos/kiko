use std::env;
use std::fs;
use std::path::PathBuf;

const BUILD_ID_ENV: &str = "KIKO_EYE_FIRMWARE_BUILD_ID_HEX";
const RIGHT_SIGN_ENV: &str = "KIKO_EYE_RIGHT_X_SIGN";

fn main() {
    if let Err(error) = run() {
        panic!("cannot configure KEP2 RP2350 firmware: {error}");
    }
}

fn run() -> Result<(), String> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-env-changed={BUILD_ID_ENV}");
    println!("cargo:rerun-if-env-changed={RIGHT_SIGN_ENV}");

    let out = PathBuf::from(
        env::var_os("OUT_DIR").ok_or_else(|| "Cargo did not provide OUT_DIR".to_owned())?,
    );
    fs::write(out.join("memory.x"), include_bytes!("memory.x"))
        .map_err(|error| format!("could not stage memory.x: {error}"))?;
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-arg-bins=--nmagic");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");

    if env::var_os("CARGO_FEATURE_RP2350").is_none() {
        return Ok(());
    }

    let build_id = required_hex::<32>(BUILD_ID_ENV)?;
    let right_eye_mounting = match env::var(RIGHT_SIGN_ENV).as_deref() {
        Ok("1") => "SameDirection",
        Ok("-1") => "Mirrored",
        Ok(value) => {
            return Err(format!(
                "{RIGHT_SIGN_ENV} must be exactly 1 or -1, got {value:?}"
            ));
        }
        Err(env::VarError::NotPresent) => {
            return Err(format!(
                "{RIGHT_SIGN_ENV} is required; record the physically verified right-panel mounting polarity"
            ));
        }
        Err(env::VarError::NotUnicode(_)) => {
            return Err(format!("{RIGHT_SIGN_ENV} is not valid UTF-8"));
        }
    };

    let bytes = build_id
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let generated = format!(
        "pub const FIRMWARE_BUILD_ID_BYTES: [u8; 32] = [{bytes}];\n\
         pub const RIGHT_EYE_MOUNTING: MountingSign = MountingSign::{right_eye_mounting};\n"
    );
    fs::write(out.join("provisioning.rs"), generated)
        .map_err(|error| format!("could not write generated provisioning: {error}"))?;
    Ok(())
}

fn required_hex<const N: usize>(name: &str) -> Result<[u8; N], String> {
    let value = env::var(name).map_err(|error| match error {
        env::VarError::NotPresent => format!(
            "{name} is required and must be the {N}-byte immutable release-build identity in hexadecimal"
        ),
        env::VarError::NotUnicode(_) => format!("{name} is not valid UTF-8"),
    })?;
    if value.len() != N * 2 {
        return Err(format!(
            "{name} must contain exactly {} hexadecimal characters, got {}",
            N * 2,
            value.len()
        ));
    }

    let mut output = [0_u8; N];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0]).ok_or_else(|| invalid_hex(name, index * 2))?;
        let low = hex_nibble(pair[1]).ok_or_else(|| invalid_hex(name, index * 2 + 1))?;
        output[index] = (high << 4) | low;
    }
    if output == [0; N] {
        return Err(format!("{name} must not be all zero"));
    }
    Ok(output)
}

fn invalid_hex(name: &str, offset: usize) -> String {
    format!("{name} contains a non-hexadecimal byte at offset {offset}")
}

const fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}
