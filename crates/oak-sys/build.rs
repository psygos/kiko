fn main() {
    let mut build = cxx_build::bridge("src/lib.rs");

    build
        .file("cpp/oak_device.cpp")
        .include("cpp")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-missing-template-arg-list-after-template-kw");

    let (depthai_include, depthai_lib) = if std::env::consts::OS == "macos" {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/ttrb".to_string());
        let depthai_root = format!("{}/depthai-core", home);
        (format!("{}/include", depthai_root), format!("{}/build", depthai_root))
    } else {
        (
            std::env::var("DEPTHAI_INCLUDE").unwrap_or_else(|_| "/usr/local/include".to_string()),
            std::env::var("DEPTHAI_LIB").unwrap_or_else(|_| "/usr/local/lib".to_string()),
        )
    };

    build.include(&depthai_include);

    if std::env::consts::OS == "macos" {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/ttrb".to_string());
        let depthai_root = format!("{}/depthai-core", home);

        build.include(format!("{}/build", depthai_root));
        build.include(format!("{}/build/_deps/xlink-src/include", depthai_root));
        build.include(format!("{}/shared", depthai_root));
        build.include(format!("{}/shared/depthai-shared/include", depthai_root));
        build.include(format!("{}/shared/depthai-bootloader-shared/include", depthai_root));
        build.include(format!("{}/build/vcpkg_installed/arm64-osx/include", depthai_root));
        build.include(format!("{}/build/_deps/libnop-src/include", depthai_root));
        build.include("/opt/homebrew/opt/opencv/include/opencv4");
    }

    println!("cargo:rustc-link-search=native={}", depthai_lib);
    println!("cargo:rustc-link-lib=dylib=depthai-core");

    if std::env::consts::OS == "macos" {
        println!("cargo:rustc-link-search=native=/opt/homebrew/opt/opencv/lib");
        println!("cargo:rustc-link-lib=dylib=opencv_core");
        println!("cargo:rustc-link-lib=dylib=opencv_imgproc");
    } else {
        println!("cargo:rustc-link-lib=usb-1.0");
    }

    build.compile("oak_bridge");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cpp/oak_device.hpp");
    println!("cargo:rerun-if-changed=cpp/oak_device.cpp");
    println!("cargo:rerun-if-env-changed=DEPTHAI_INCLUDE");
    println!("cargo:rerun-if-env-changed=DEPTHAI_LIB");
}
