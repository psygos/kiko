use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let output_directory = PathBuf::from(
        env::var_os("OUT_DIR")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "OUT_DIR is not set"))?,
    );
    fs::copy("memory.x", output_directory.join("memory.x"))?;

    println!("cargo:rustc-link-search={}", output_directory.display());
    println!("cargo:rustc-link-arg-bin=embedded=-Tlink.x");
    println!("cargo:rerun-if-changed=memory.x");
    Ok(())
}
