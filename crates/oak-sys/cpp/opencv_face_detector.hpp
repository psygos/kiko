#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"

namespace kiko {
namespace oak {

struct RawOpenCvHaarFaceDetectorConfig;
struct RawHaarFaceDetectionBatch;

class OpenCvHaarFaceDetector {
public:
    OpenCvHaarFaceDetector(
        rust::Slice<const uint8_t> frontal_cascade_xml,
        rust::Slice<const uint8_t> profile_cascade_xml,
        const RawOpenCvHaarFaceDetectorConfig& config
    );
    ~OpenCvHaarFaceDetector() noexcept;

    OpenCvHaarFaceDetector(const OpenCvHaarFaceDetector&) = delete;
    OpenCvHaarFaceDetector& operator=(const OpenCvHaarFaceDetector&) = delete;

    RawHaarFaceDetectionBatch detect(
        rust::Slice<const uint8_t> tightly_packed_bgr,
        uint32_t width,
        uint32_t height,
        uint32_t stride_bytes
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<OpenCvHaarFaceDetector> create_opencv_haar_face_detector(
    rust::Slice<const uint8_t> frontal_cascade_xml,
    rust::Slice<const uint8_t> profile_cascade_xml,
    const RawOpenCvHaarFaceDetectorConfig& config
);

}  // namespace oak
}  // namespace kiko
