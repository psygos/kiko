//! Feature-specific CXX surface for the host OpenCV face detector.
//!
//! Keeping this bridge in its own source file lets `build.rs` select the
//! complete native surface from the already-parsed Cargo feature. CXX build
//! generation therefore does not need to reinterpret a hyphenated
//! `cfg(feature = "...")` inside the primary OAK bridge.

#[cxx::bridge(namespace = "kiko::oak")]
#[cfg_attr(oak_sys_check_only, allow(dead_code))]
pub(crate) mod ffi {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum RawHaarFaceDetectionSource {
        Frontal,
        Profile,
        MirroredProfile,
    }

    #[derive(Debug, Clone, Copy)]
    struct RawOpenCvHaarFaceDetectorConfig {
        scale_factor: f64,
        frontal_minimum_neighbors: u32,
        profile_minimum_neighbors: u32,
        minimum_face_width: u32,
        minimum_face_height: u32,
    }

    #[derive(Debug, Clone, Copy)]
    struct RawHaarFaceDetection {
        x: i64,
        y: i64,
        width: i64,
        height: i64,
        detector_level_weight: f64,
        source: RawHaarFaceDetectionSource,
    }

    #[derive(Debug)]
    struct RawHaarFaceDetectionBatch {
        detections: Vec<RawHaarFaceDetection>,
    }

    #[cfg(not(oak_sys_check_only))]
    unsafe extern "C++" {
        include!("opencv_face_detector.hpp");

        type OpenCvHaarFaceDetector;

        fn create_opencv_haar_face_detector(
            frontal_cascade_xml: &[u8],
            profile_cascade_xml: &[u8],
            config: &RawOpenCvHaarFaceDetectorConfig,
        ) -> Result<UniquePtr<OpenCvHaarFaceDetector>>;
        fn detect(
            self: Pin<&mut OpenCvHaarFaceDetector>,
            tightly_packed_bgr: &[u8],
            width: u32,
            height: u32,
            stride_bytes: u32,
        ) -> Result<RawHaarFaceDetectionBatch>;
    }
}
