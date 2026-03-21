fn main() {
    let nvblox_root =
        std::env::var("NVBLOX_ROOT").unwrap_or_else(|_| {
            "/home/makerspace/work/isaac_ros_nvblox/nvblox_ros/nvblox_core".to_string()
        });
    let nvblox_build =
        std::env::var("NVBLOX_BUILD").unwrap_or_else(|_| format!("{nvblox_root}/build"));

    // Compile the C wrapper
    cc::Build::new()
        .cpp(true)
        .file("nvblox_c.cpp")
        .include(format!("{nvblox_root}/nvblox/include"))
        .include(format!("{nvblox_build}/nvblox/eigen/include/eigen3"))
        .include(format!("{nvblox_build}/_deps/ext_stdgpu-src/src"))
        .include(format!("{nvblox_build}/_deps/ext_stdgpu-build/src"))
        .include(format!("{nvblox_build}/_deps/ext_glog-src/src"))
        .include(format!("{nvblox_build}/_deps/ext_glog-build"))
        .include("/usr/local/cuda/include")
        .std("c++17")
        .flag("-Wno-deprecated-declarations")
        .compile("nvblox_c");

    // Link against nvblox shared library
    println!("cargo:rustc-link-search=native={nvblox_build}/nvblox");
    println!("cargo:rustc-link-lib=dylib=nvblox_lib");

    // Link vendored glog and gflags (static)
    println!("cargo:rustc-link-search=native={nvblox_build}/_deps/ext_glog-build");
    println!("cargo:rustc-link-lib=static=glog");
    println!("cargo:rustc-link-search=native={nvblox_build}/_deps/ext_gflags-build");
    println!("cargo:rustc-link-lib=static=gflags_nothreads");

    // Link CUDA runtime
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Link C++ stdlib + unwind
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=unwind");
}
