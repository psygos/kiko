use std::env;
use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};

const INPUT_ENV_VARS: &[&str] = &[
    "CARGO_CFG_TARGET_OS",
    "CARGO_CFG_TARGET_ARCH",
    "CARGO_MANIFEST_DIR",
    "CARGO_FEATURE_OPENCV_FACE_DETECTOR",
    "DEPTHAI_INCLUDE",
    "DEPTHAI_LIB",
    "HOMEBREW_PREFIX",
    "OPENCV_INCLUDE",
    "OPENCV_LIB",
    "OAK_SYS_CHECK_ONLY",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetOs {
    Linux,
    Macos,
}

impl TargetOs {
    fn parse(value: &str) -> Result<Self, DiscoveryError> {
        match value {
            "linux" => Ok(Self::Linux),
            "macos" => Ok(Self::Macos),
            _ => Err(DiscoveryError::new(
                "CARGO_CFG_TARGET_OS",
                format!("oak-sys supports only Linux and macOS targets, got {value:?}"),
            )),
        }
    }

    fn dynamic_library(self, link_name: &str) -> String {
        match self {
            Self::Linux => format!("lib{link_name}.so"),
            Self::Macos => format!("lib{link_name}.dylib"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DiscoveryError {
    variable: &'static str,
    detail: String,
}

impl DiscoveryError {
    fn new(variable: &'static str, detail: impl Into<String>) -> Self {
        Self {
            variable,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for DiscoveryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}: {}; discovery inputs: {}",
            self.variable,
            self.detail,
            INPUT_ENV_VARS.join(", ")
        )
    }
}

impl std::error::Error for DiscoveryError {}

#[derive(Debug)]
struct BuildInputs {
    target_os: TargetOs,
    target_arch: String,
    opencv_face_detector: bool,
    depthai_include: Option<Vec<PathBuf>>,
    depthai_lib: Option<Vec<PathBuf>>,
    homebrew_prefix: Option<Vec<PathBuf>>,
    opencv_include: Option<Vec<PathBuf>>,
    opencv_lib: Option<Vec<PathBuf>>,
}

impl BuildInputs {
    fn from_process() -> Result<Self, DiscoveryError> {
        let target_os_value = env::var("CARGO_CFG_TARGET_OS")
            .map_err(|error| DiscoveryError::new("CARGO_CFG_TARGET_OS", error.to_string()))?;
        let target_os = TargetOs::parse(&target_os_value)?;
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH")
            .map_err(|error| DiscoveryError::new("CARGO_CFG_TARGET_ARCH", error.to_string()))?;
        let manifest_dir = env::var_os("CARGO_MANIFEST_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .ok_or_else(|| {
                DiscoveryError::new(
                    "CARGO_MANIFEST_DIR",
                    "required Cargo path is unset or empty",
                )
            })?;
        let parse = |variable| parse_path_list(variable, env::var_os(variable), &manifest_dir);
        let opencv_face_detector = env::var_os("CARGO_FEATURE_OPENCV_FACE_DETECTOR").is_some();
        let macos_path = |variable| {
            if target_os == TargetOs::Macos {
                parse(variable)
            } else {
                Ok(None)
            }
        };
        let opencv_path = |variable| {
            if target_os == TargetOs::Macos || opencv_face_detector {
                parse(variable)
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            target_os,
            target_arch,
            opencv_face_detector,
            depthai_include: parse("DEPTHAI_INCLUDE")?,
            depthai_lib: parse("DEPTHAI_LIB")?,
            homebrew_prefix: macos_path("HOMEBREW_PREFIX")?,
            opencv_include: opencv_path("OPENCV_INCLUDE")?,
            opencv_lib: opencv_path("OPENCV_LIB")?,
        })
    }
}

#[derive(Debug)]
struct NativePaths {
    depthai_include: Vec<PathBuf>,
    depthai_lib: Vec<PathBuf>,
    opencv_include: Vec<PathBuf>,
    opencv_lib: Vec<PathBuf>,
}

#[derive(Debug)]
struct CandidatePaths {
    include: Vec<PathBuf>,
    lib: Vec<PathBuf>,
    opencv_include: Vec<PathBuf>,
    opencv_lib: Vec<PathBuf>,
}

fn parse_path_list(
    variable: &'static str,
    value: Option<OsString>,
    relative_to: &Path,
) -> Result<Option<Vec<PathBuf>>, DiscoveryError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let raw = env::split_paths(&value).collect::<Vec<_>>();
    if value.is_empty() || raw.is_empty() || raw.iter().any(|path| path.as_os_str().is_empty()) {
        return Err(DiscoveryError::new(
            variable,
            "path list is empty or contains an empty segment",
        ));
    }
    let paths = raw
        .into_iter()
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                relative_to.join(path)
            }
        })
        .collect::<Vec<_>>();
    if paths.iter().any(|path| {
        path.to_str()
            .is_none_or(|path| path.contains('\n') || path.contains('\r'))
    }) {
        return Err(DiscoveryError::new(
            variable,
            "Cargo directive paths must be Unicode and contain no line breaks",
        ));
    }
    Ok(Some(paths))
}

fn display_paths(paths: impl IntoIterator<Item = PathBuf>) -> String {
    paths
        .into_iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn supplies(paths: &[PathBuf], required: &[Vec<String>]) -> bool {
    required.iter().all(|alternatives| {
        paths.iter().any(|path| {
            alternatives
                .iter()
                .any(|artifact| path.join(artifact).is_file())
        })
    })
}

fn checked_artifacts(paths: &[PathBuf], required: &[Vec<String>]) -> String {
    display_paths(paths.iter().flat_map(|path| {
        required
            .iter()
            .flat_map(|alternatives| alternatives.iter().map(|artifact| path.join(artifact)))
    }))
}

fn resolve_paths(
    variable: &'static str,
    explicit: Option<Vec<PathBuf>>,
    candidates: &[PathBuf],
    required: &[Vec<String>],
    guidance: &str,
) -> Result<Vec<PathBuf>, DiscoveryError> {
    if let Some(paths) = explicit {
        let missing = paths
            .iter()
            .filter(|path| !path.is_dir())
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(DiscoveryError::new(
                variable,
                format!("directories do not exist: {}", display_paths(missing)),
            ));
        }
        if supplies(&paths, required) {
            return Ok(paths);
        }
        return Err(DiscoveryError::new(
            variable,
            format!(
                "explicit path list is authoritative but incomplete; {guidance}; checked: {}",
                checked_artifacts(&paths, required)
            ),
        ));
    }
    candidates
        .iter()
        .find(|path| supplies(std::slice::from_ref(path), required))
        .cloned()
        .map(|path| vec![path])
        .ok_or_else(|| {
            DiscoveryError::new(
                variable,
                format!(
                    "override is unset; {guidance}; checked: {}",
                    checked_artifacts(candidates, required)
                ),
            )
        })
}

fn header_requirement(header: &str) -> Vec<Vec<String>> {
    vec![vec![header.to_string()]]
}

fn header_requirements(headers: &[&str]) -> Vec<Vec<String>> {
    headers
        .iter()
        .map(|header| vec![(*header).to_string()])
        .collect()
}

fn library_requirements(target_os: TargetOs, link_names: &[&str]) -> Vec<Vec<String>> {
    link_names
        .iter()
        .map(|link_name| {
            vec![
                target_os.dynamic_library(link_name),
                format!("lib{link_name}.a"),
            ]
        })
        .collect()
}

fn dynamic_library_requirements(target_os: TargetOs, link_names: &[&str]) -> Vec<Vec<String>> {
    link_names
        .iter()
        .map(|link_name| vec![target_os.dynamic_library(link_name)])
        .collect()
}

fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.contains(&path) {
        paths.push(path);
    }
}

fn candidate_paths(inputs: &BuildInputs) -> CandidatePaths {
    let mut prefixes = inputs.homebrew_prefix.clone().unwrap_or_default();
    if inputs.target_os == TargetOs::Macos {
        push_unique(&mut prefixes, PathBuf::from("/opt/homebrew"));
        push_unique(&mut prefixes, PathBuf::from("/usr/local"));
    }
    let mut include = Vec::new();
    let mut lib = Vec::new();
    let mut opencv_include = Vec::new();
    let mut opencv_lib = Vec::new();
    for prefix in &prefixes {
        push_unique(&mut include, prefix.join("include"));
        push_unique(&mut lib, prefix.join("lib"));
        push_unique(
            &mut opencv_include,
            prefix.join("opt/opencv/include/opencv4"),
        );
        push_unique(&mut opencv_include, prefix.join("include/opencv4"));
        push_unique(&mut opencv_lib, prefix.join("opt/opencv/lib"));
        push_unique(&mut opencv_lib, prefix.join("lib"));
    }
    push_unique(&mut include, PathBuf::from("/usr/local/include"));
    push_unique(&mut lib, PathBuf::from("/usr/local/lib"));
    push_unique(
        &mut opencv_include,
        PathBuf::from("/usr/local/include/opencv4"),
    );
    push_unique(&mut opencv_lib, PathBuf::from("/usr/local/lib"));
    if inputs.target_os == TargetOs::Linux {
        push_unique(&mut include, PathBuf::from("/usr/include"));
        push_unique(&mut opencv_include, PathBuf::from("/usr/include/opencv4"));
        push_unique(&mut lib, PathBuf::from("/usr/local/lib64"));
        push_unique(&mut opencv_lib, PathBuf::from("/usr/local/lib64"));
        if let Some(multiarch) = match inputs.target_arch.as_str() {
            "aarch64" => Some("aarch64-linux-gnu"),
            "arm" => Some("arm-linux-gnueabihf"),
            "x86_64" => Some("x86_64-linux-gnu"),
            _ => None,
        } {
            let multiarch_lib = PathBuf::from("/usr/lib").join(multiarch);
            push_unique(&mut lib, multiarch_lib.clone());
            push_unique(&mut opencv_lib, multiarch_lib);
        }
        push_unique(&mut lib, PathBuf::from("/usr/lib64"));
        push_unique(&mut lib, PathBuf::from("/usr/lib"));
        push_unique(&mut opencv_lib, PathBuf::from("/usr/lib64"));
        push_unique(&mut opencv_lib, PathBuf::from("/usr/lib"));
    }
    CandidatePaths {
        include,
        lib,
        opencv_include,
        opencv_lib,
    }
}

fn discover(inputs: &BuildInputs) -> Result<NativePaths, DiscoveryError> {
    if inputs.opencv_face_detector
        && inputs.target_os == TargetOs::Linux
        && inputs.target_arch != "aarch64"
    {
        return Err(DiscoveryError::new(
            "CARGO_CFG_TARGET_ARCH",
            format!(
                "the opencv-face-detector feature supports Linux aarch64 and macOS, got Linux {:?}",
                inputs.target_arch
            ),
        ));
    }
    let candidates = candidate_paths(inputs);
    let depthai_include = resolve_paths(
        "DEPTHAI_INCLUDE",
        inputs.depthai_include.clone(),
        &candidates.include,
        &header_requirement("depthai/depthai.hpp"),
        "use an installed include root or an OS path list containing every source-tree/transitive include root",
    )?;
    let depthai_lib = resolve_paths(
        "DEPTHAI_LIB",
        inputs.depthai_lib.clone(),
        &candidates.lib,
        &library_requirements(inputs.target_os, &["depthai-core"]),
        "each link name requires a platform dynamic library or static archive",
    )?;
    let (opencv_include, opencv_lib) = if inputs.target_os == TargetOs::Macos
        || inputs.opencv_face_detector
    {
        let headers = if inputs.opencv_face_detector {
            header_requirements(&[
                "opencv2/core.hpp",
                "opencv2/imgproc.hpp",
                "opencv2/objdetect.hpp",
            ])
        } else {
            header_requirement("opencv2/core.hpp")
        };
        let link_names = if inputs.opencv_face_detector {
            &["opencv_core", "opencv_imgproc", "opencv_objdetect"][..]
        } else {
            &["opencv_core", "opencv_imgproc"][..]
        };
        (
                resolve_paths(
                    "OPENCV_INCLUDE",
                    inputs.opencv_include.clone(),
                    &candidates.opencv_include,
                    &headers,
                    if inputs.opencv_face_detector {
                        "use an installed OpenCV include root containing core, imgproc, and objdetect headers"
                    } else {
                        "use an installed OpenCV include root containing the core header"
                    },
                )?,
                resolve_paths(
                    "OPENCV_LIB",
                    inputs.opencv_lib.clone(),
                    &candidates.opencv_lib,
                    &dynamic_library_requirements(inputs.target_os, link_names),
                    "dynamic OpenCV libraries are required; static archives are unsupported because their transitive link closure is not discovered",
                )?,
            )
    } else {
        (Vec::new(), Vec::new())
    };
    Ok(NativePaths {
        depthai_include,
        depthai_lib,
        opencv_include,
        opencv_lib,
    })
}

fn emit_rerun_directives() {
    println!("cargo:rustc-check-cfg=cfg(oak_sys_check_only)");
    for path in [
        "src/lib.rs",
        "src/opencv_face_ffi.rs",
        "cpp/oak_device.hpp",
        "cpp/oak_device.cpp",
        "cpp/opencv_face_detector.hpp",
        "cpp/opencv_face_detector.cpp",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    for variable in INPUT_ENV_VARS {
        println!("cargo:rerun-if-env-changed={variable}");
    }
}

fn bridge_sources(opencv_face_detector: bool) -> &'static [&'static str] {
    if opencv_face_detector {
        &["src/lib.rs", "src/opencv_face_ffi.rs"]
    } else {
        &["src/lib.rs"]
    }
}

fn link_directives(target_os: TargetOs, opencv_face_detector: bool) -> Vec<&'static str> {
    let mut directives = match target_os {
        TargetOs::Macos => vec![
            "cargo:rustc-link-lib=depthai-core",
            "cargo:rustc-link-lib=opencv_core",
            "cargo:rustc-link-lib=opencv_imgproc",
        ],
        TargetOs::Linux => vec![
            "cargo:rustc-link-lib=depthai-core",
            "cargo:rustc-link-lib=usb-1.0",
        ],
    };
    if opencv_face_detector {
        match target_os {
            TargetOs::Macos => {
                directives.push("cargo:rustc-link-lib=opencv_objdetect");
            }
            TargetOs::Linux => directives.extend([
                "cargo:rustc-link-lib=opencv_objdetect",
                "cargo:rustc-link-lib=opencv_imgproc",
                "cargo:rustc-link-lib=opencv_core",
            ]),
        }
    }
    directives
}

#[cfg(not(test))]
fn main() {
    emit_rerun_directives();
    match env::var("OAK_SYS_CHECK_ONLY") {
        Ok(value) if value == "1" => {
            println!("cargo:rustc-cfg=oak_sys_check_only");
            println!(
                "cargo:warning=oak-sys native bridge skipped for compile-only host validation"
            );
            return;
        }
        Ok(value) => panic!("OAK_SYS_CHECK_ONLY must be exactly `1` when set, got {value:?}"),
        Err(env::VarError::NotPresent) => {}
        Err(env::VarError::NotUnicode(_)) => {
            panic!("OAK_SYS_CHECK_ONLY must contain valid UTF-8")
        }
    }
    let inputs = BuildInputs::from_process()
        .unwrap_or_else(|error| panic!("oak-sys native dependency discovery failed: {error}"));
    let paths = discover(&inputs)
        .unwrap_or_else(|error| panic!("oak-sys native dependency discovery failed: {error}"));
    let mut build = cxx_build::bridges(bridge_sources(inputs.opencv_face_detector).iter().copied());
    build
        .file("cpp/oak_device.cpp")
        .include("cpp")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-missing-template-arg-list-after-template-kw");
    if inputs.opencv_face_detector {
        build.file("cpp/opencv_face_detector.cpp");
    }
    for include in paths.depthai_include.iter().chain(&paths.opencv_include) {
        build.include(include);
    }
    for library in &paths.depthai_lib {
        println!("cargo:rustc-link-search=native={}", library.display());
    }
    if !paths.opencv_lib.is_empty() {
        for library in &paths.opencv_lib {
            println!("cargo:rustc-link-search=native={}", library.display());
        }
    }
    for directive in link_directives(inputs.target_os, inputs.opencv_face_detector) {
        println!("{directive}");
    }
    build.compile("oak_bridge");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let suffix = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path =
                env::temp_dir().join(format!("kiko-oak-build-{}-{suffix}", std::process::id()));
            fs::create_dir(&path).expect("create test directory");
            Self(path)
        }

        fn directory(&self, relative: &str) -> PathBuf {
            let path = self.0.join(relative);
            fs::create_dir_all(&path).expect("create test directory");
            path
        }

        fn artifact(&self, relative: &str) {
            let path = self.0.join(relative);
            fs::create_dir_all(path.parent().expect("artifact parent"))
                .expect("create artifact parent");
            fs::write(path, []).expect("create artifact");
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove test directory");
        }
    }

    #[test]
    fn face_feature_selects_its_complete_cxx_bridge_surface() {
        assert_eq!(bridge_sources(false), ["src/lib.rs"]);
        assert_eq!(
            bridge_sources(true),
            ["src/lib.rs", "src/opencv_face_ffi.rs"]
        );
    }

    #[test]
    fn explicit_path_lists_are_aggregated_authoritative_and_static_aware() {
        let tree = TestDirectory::new();
        let primary = tree.directory("primary");
        let transitive = tree.directory("transitive");
        let library = tree.directory("library");
        let fallback = tree.directory("fallback");
        tree.artifact("primary/depthai/depthai.hpp");
        tree.artifact("library/libdepthai-core.a");
        tree.artifact("fallback/libdepthai-core.dylib");
        let joined = env::join_paths([&primary, &transitive]).expect("join paths");
        let includes = parse_path_list("DEPTHAI_INCLUDE", Some(joined), &tree.0)
            .expect("parse paths")
            .expect("explicit paths");

        assert_eq!(
            resolve_paths(
                "DEPTHAI_INCLUDE",
                Some(includes),
                &[],
                &header_requirement("depthai/depthai.hpp"),
                "include guidance",
            )
            .expect("resolve includes"),
            vec![primary, transitive]
        );
        assert_eq!(
            resolve_paths(
                "DEPTHAI_LIB",
                Some(vec![library.clone()]),
                &[fallback.clone()],
                &library_requirements(TargetOs::Macos, &["depthai-core"]),
                "library guidance",
            )
            .expect("resolve static library"),
            vec![library.clone()]
        );

        fs::remove_file(library.join("libdepthai-core.a")).expect("remove static fixture");
        let error = resolve_paths(
            "DEPTHAI_LIB",
            Some(vec![library.clone()]),
            &[fallback.clone()],
            &library_requirements(TargetOs::Macos, &["depthai-core"]),
            "library guidance",
        )
        .unwrap_err();
        assert!(error.detail.contains(&library.display().to_string()));
        assert!(!error.detail.contains(&fallback.display().to_string()));
    }

    #[test]
    fn homebrew_opencv_opt_layout_is_a_first_class_candidate() {
        let prefix = PathBuf::from("/custom/homebrew");
        let inputs = BuildInputs {
            target_os: TargetOs::Macos,
            target_arch: "aarch64".to_string(),
            opencv_face_detector: true,
            depthai_include: None,
            depthai_lib: None,
            homebrew_prefix: Some(vec![prefix.clone()]),
            opencv_include: None,
            opencv_lib: None,
        };
        let candidates = candidate_paths(&inputs);
        assert_eq!(
            candidates.opencv_include[0],
            prefix.join("opt/opencv/include/opencv4")
        );
        assert_eq!(candidates.opencv_lib[0], prefix.join("opt/opencv/lib"));
        assert_eq!(
            link_directives(TargetOs::Macos, true),
            [
                "cargo:rustc-link-lib=depthai-core",
                "cargo:rustc-link-lib=opencv_core",
                "cargo:rustc-link-lib=opencv_imgproc",
                "cargo:rustc-link-lib=opencv_objdetect",
            ]
        );
        assert_eq!(
            link_directives(TargetOs::Linux, false),
            [
                "cargo:rustc-link-lib=depthai-core",
                "cargo:rustc-link-lib=usb-1.0",
            ]
        );
    }

    #[test]
    fn macos_preserves_baseline_opencv_and_feature_adds_dynamic_objdetect() {
        let tree = TestDirectory::new();
        let depthai_include = tree.directory("depthai-include");
        let depthai_lib = tree.directory("depthai-lib");
        tree.artifact("depthai-include/depthai/depthai.hpp");
        tree.artifact("depthai-lib/libdepthai-core.dylib");
        let opencv_include = tree.directory("opencv-include");
        let opencv_lib = tree.directory("opencv-lib");
        tree.artifact("opencv-include/opencv2/core.hpp");
        tree.artifact("opencv-lib/libopencv_core.dylib");
        tree.artifact("opencv-lib/libopencv_imgproc.dylib");

        let disabled = BuildInputs {
            target_os: TargetOs::Macos,
            target_arch: "aarch64".to_string(),
            opencv_face_detector: false,
            depthai_include: Some(vec![depthai_include.clone()]),
            depthai_lib: Some(vec![depthai_lib.clone()]),
            homebrew_prefix: None,
            opencv_include: Some(vec![opencv_include.clone()]),
            opencv_lib: Some(vec![opencv_lib.clone()]),
        };
        let paths = discover(&disabled).expect("feature-off macOS keeps baseline OpenCV");
        assert_eq!(paths.opencv_include, vec![opencv_include.clone()]);
        assert_eq!(paths.opencv_lib, vec![opencv_lib.clone()]);
        assert_eq!(
            link_directives(TargetOs::Macos, false),
            [
                "cargo:rustc-link-lib=depthai-core",
                "cargo:rustc-link-lib=opencv_core",
                "cargo:rustc-link-lib=opencv_imgproc",
            ]
        );

        for header in ["opencv2/imgproc.hpp", "opencv2/objdetect.hpp"] {
            tree.artifact(&format!("opencv-include/{header}"));
        }
        tree.artifact("opencv-lib/libopencv_objdetect.dylib");
        let enabled = BuildInputs {
            opencv_face_detector: true,
            opencv_include: Some(vec![opencv_include.clone()]),
            opencv_lib: Some(vec![opencv_lib.clone()]),
            ..disabled
        };
        let paths = discover(&enabled).expect("complete OpenCV feature discovery");
        assert_eq!(paths.opencv_include, vec![opencv_include]);
        assert_eq!(paths.opencv_lib, vec![opencv_lib.clone()]);

        fs::remove_file(opencv_lib.join("libopencv_objdetect.dylib"))
            .expect("remove objdetect fixture");
        tree.artifact("opencv-lib/libopencv_objdetect.a");
        let error = discover(&enabled).unwrap_err();
        assert_eq!(error.variable, "OPENCV_LIB");
        assert!(error.detail.contains("libopencv_objdetect"));
        assert!(error.detail.contains("static archives are unsupported"));
    }

    #[test]
    fn feature_off_linux_does_not_discover_or_link_opencv() {
        let tree = TestDirectory::new();
        let depthai_include = tree.directory("depthai-include");
        let depthai_lib = tree.directory("depthai-lib");
        tree.artifact("depthai-include/depthai/depthai.hpp");
        tree.artifact("depthai-lib/libdepthai-core.so");
        let inputs = BuildInputs {
            target_os: TargetOs::Linux,
            target_arch: "aarch64".to_string(),
            opencv_face_detector: false,
            depthai_include: Some(vec![depthai_include]),
            depthai_lib: Some(vec![depthai_lib]),
            homebrew_prefix: None,
            opencv_include: Some(vec![tree.0.join("missing-opencv-include")]),
            opencv_lib: Some(vec![tree.0.join("missing-opencv-lib")]),
        };
        let paths = discover(&inputs).expect("feature-off Linux ignores OpenCV");
        assert!(paths.opencv_include.is_empty());
        assert!(paths.opencv_lib.is_empty());
        assert_eq!(
            link_directives(TargetOs::Linux, false),
            [
                "cargo:rustc-link-lib=depthai-core",
                "cargo:rustc-link-lib=usb-1.0",
            ]
        );
    }

    #[test]
    fn linux_aarch64_opencv_candidates_and_links_are_explicit() {
        let inputs = BuildInputs {
            target_os: TargetOs::Linux,
            target_arch: "aarch64".to_string(),
            opencv_face_detector: true,
            depthai_include: None,
            depthai_lib: None,
            homebrew_prefix: None,
            opencv_include: None,
            opencv_lib: None,
        };
        let candidates = candidate_paths(&inputs);
        assert!(candidates
            .opencv_include
            .contains(&PathBuf::from("/usr/include/opencv4")));
        assert!(candidates
            .opencv_lib
            .contains(&PathBuf::from("/usr/lib/aarch64-linux-gnu")));
        assert_eq!(
            link_directives(TargetOs::Linux, true),
            [
                "cargo:rustc-link-lib=depthai-core",
                "cargo:rustc-link-lib=usb-1.0",
                "cargo:rustc-link-lib=opencv_objdetect",
                "cargo:rustc-link-lib=opencv_imgproc",
                "cargo:rustc-link-lib=opencv_core",
            ]
        );
    }

    #[test]
    fn feature_rejects_non_aarch64_linux_native_builds() {
        let tree = TestDirectory::new();
        let depthai_include = tree.directory("depthai-include");
        let depthai_lib = tree.directory("depthai-lib");
        tree.artifact("depthai-include/depthai/depthai.hpp");
        tree.artifact("depthai-lib/libdepthai-core.so");
        let inputs = BuildInputs {
            target_os: TargetOs::Linux,
            target_arch: "x86_64".to_string(),
            opencv_face_detector: true,
            depthai_include: Some(vec![depthai_include]),
            depthai_lib: Some(vec![depthai_lib]),
            homebrew_prefix: None,
            opencv_include: None,
            opencv_lib: None,
        };

        let error = discover(&inputs).unwrap_err();
        assert_eq!(error.variable, "CARGO_CFG_TARGET_ARCH");
        assert!(error.detail.contains("Linux aarch64 and macOS"));
    }
}
