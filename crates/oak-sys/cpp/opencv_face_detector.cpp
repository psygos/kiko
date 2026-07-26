#include "oak-sys/src/opencv_face_ffi.rs.h"
#include "opencv_face_detector.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kiko {
namespace oak {

namespace {

constexpr uint32_t MAXIMUM_HAAR_FACE_FRAME_DIMENSION = 4096;
constexpr uint64_t MAXIMUM_HAAR_FACE_FRAME_PIXELS = 8'388'608;
constexpr size_t MAXIMUM_NATIVE_HAAR_FACE_DETECTIONS = 4096;
constexpr size_t MAXIMUM_OPENCV_HAAR_CASCADE_XML_BYTES = 4 * 1024 * 1024;

std::string copied_cascade_xml(
    rust::Slice<const uint8_t> xml,
    const char* cascade_kind
) {
    if (xml.empty()) {
        throw std::invalid_argument(
            std::string(cascade_kind) + " Haar cascade XML must not be empty"
        );
    }
    if (xml.size() > MAXIMUM_OPENCV_HAAR_CASCADE_XML_BYTES) {
        throw std::length_error(
            std::string(cascade_kind) + " Haar cascade XML exceeds "
            + std::to_string(MAXIMUM_OPENCV_HAAR_CASCADE_XML_BYTES)
            + "-byte hard limit"
        );
    }
    // Copy before constructing FileStorage. No OpenCV object receives a view
    // into, or can retain, the borrowed Rust slice.
    std::string result(
        reinterpret_cast<const char*>(xml.data()),
        xml.size()
    );
    if (result.find('\0') != std::string::npos) {
        throw std::invalid_argument(
            std::string(cascade_kind) + " Haar cascade XML contains an embedded NUL"
        );
    }
    return result;
}

void read_cascade_from_memory(
    cv::CascadeClassifier& classifier,
    const std::string& xml,
    const char* cascade_kind
) {
    try {
        cv::FileStorage storage(
            xml,
            cv::FileStorage::READ | cv::FileStorage::MEMORY
        );
        if (!storage.isOpened()) {
            throw std::runtime_error(
                std::string("OpenCV could not open the supplied ")
                + cascade_kind + " Haar cascade XML from memory"
            );
        }
        const cv::FileNode classifier_node = storage.getFirstTopLevelNode();
        if (classifier_node.empty()) {
            throw std::runtime_error(
                std::string("supplied ") + cascade_kind
                + " Haar cascade XML has no top-level classifier node"
            );
        }
        if (!classifier.read(classifier_node) || classifier.empty()) {
            throw std::runtime_error(
                std::string("OpenCV could not read a nonempty ")
                + cascade_kind
                + " Haar classifier from the first top-level XML node"
            );
        }
    } catch (const cv::Exception& error) {
        throw std::runtime_error(
            std::string("OpenCV rejected supplied ") + cascade_kind
            + " Haar cascade XML: " + error.what()
        );
    }
}

int checked_positive_int(uint32_t value, const char* field) {
    if (value == 0 || value > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::string("OpenCV Haar ") + field
            + " must be nonzero and fit a signed int"
        );
    }
    return static_cast<int>(value);
}

size_t checked_bgr_size(uint32_t stride_bytes, uint32_t height) {
    const auto stride = static_cast<size_t>(stride_bytes);
    const auto rows = static_cast<size_t>(height);
    if (rows != 0 && stride > std::numeric_limits<size_t>::max() / rows) {
        throw std::invalid_argument("tightly packed BGR frame size overflows size_t");
    }
    return stride * rows;
}

}  // namespace

struct OpenCvHaarFaceDetector::Impl {
    Impl(
        rust::Slice<const uint8_t> frontal_cascade_xml,
        rust::Slice<const uint8_t> profile_cascade_xml,
        const RawOpenCvHaarFaceDetectorConfig& config
    )
        : scale_factor(config.scale_factor)
        , frontal_minimum_neighbors(checked_positive_int(
              config.frontal_minimum_neighbors,
              "frontal_minimum_neighbors"
          ))
        , profile_minimum_neighbors(checked_positive_int(
              config.profile_minimum_neighbors,
              "profile_minimum_neighbors"
          ))
        , minimum_face_size(
              checked_positive_int(config.minimum_face_width, "minimum_face_width"),
              checked_positive_int(config.minimum_face_height, "minimum_face_height")
          )
    {
        if (!std::isfinite(scale_factor) || scale_factor <= 1.0) {
            throw std::invalid_argument(
                "OpenCV Haar scale_factor must be finite and greater than 1"
            );
        }
        const std::string frontal_xml =
            copied_cascade_xml(frontal_cascade_xml, "frontal");
        const std::string profile_xml =
            copied_cascade_xml(profile_cascade_xml, "profile");
        read_cascade_from_memory(frontal, frontal_xml, "frontal");
        read_cascade_from_memory(profile, profile_xml, "profile");
    }

    bool run(
        cv::CascadeClassifier& classifier,
        const cv::Mat& grayscale,
        int minimum_neighbors
    ) {
        rectangles.clear();
        reject_levels.clear();
        level_weights.clear();
        classifier.detectMultiScale(
            grayscale,
            rectangles,
            reject_levels,
            level_weights,
            scale_factor,
            minimum_neighbors,
            0,
            minimum_face_size,
            cv::Size(),
            true
        );
        if (rectangles.size() != reject_levels.size()
            || rectangles.size() != level_weights.size()) {
            throw std::runtime_error(
                "OpenCV Haar detector returned mismatched rectangles and level weights"
            );
        }
        // OpenCV has already performed its internal search by this point.
        // This limit bounds only candidates crossing into the CXX result.
        if (rectangles.size() > MAXIMUM_NATIVE_HAAR_FACE_DETECTIONS) {
            throw std::length_error(
                "OpenCV Haar returned " + std::to_string(rectangles.size())
                + " candidates, exceeding absolute native bridge limit "
                + std::to_string(MAXIMUM_NATIVE_HAAR_FACE_DETECTIONS)
                + "; this limit applies after OpenCV internal search"
            );
        }
        return !rectangles.empty();
    }

    cv::CascadeClassifier frontal;
    cv::CascadeClassifier profile;
    double scale_factor;
    int frontal_minimum_neighbors;
    int profile_minimum_neighbors;
    cv::Size minimum_face_size;
    cv::Mat grayscale;
    cv::Mat equalized_grayscale;
    cv::Mat mirrored_equalized_grayscale;
    std::vector<cv::Rect> rectangles;
    std::vector<int> reject_levels;
    std::vector<double> level_weights;
};

OpenCvHaarFaceDetector::OpenCvHaarFaceDetector(
    rust::Slice<const uint8_t> frontal_cascade_xml,
    rust::Slice<const uint8_t> profile_cascade_xml,
    const RawOpenCvHaarFaceDetectorConfig& config
)
    : impl_(std::make_unique<Impl>(
          frontal_cascade_xml,
          profile_cascade_xml,
          config
      ))
{}

OpenCvHaarFaceDetector::~OpenCvHaarFaceDetector() noexcept = default;

RawHaarFaceDetectionBatch OpenCvHaarFaceDetector::detect(
    rust::Slice<const uint8_t> tightly_packed_bgr,
    uint32_t width,
    uint32_t height,
    uint32_t stride_bytes
) {
    const auto frame_pixels = static_cast<uint64_t>(width)
        * static_cast<uint64_t>(height);
    if (width == 0 || height == 0
        || width > MAXIMUM_HAAR_FACE_FRAME_DIMENSION
        || height > MAXIMUM_HAAR_FACE_FRAME_DIMENSION
        || frame_pixels > MAXIMUM_HAAR_FACE_FRAME_PIXELS
        || width > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || height > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || width > std::numeric_limits<uint32_t>::max() / 3
        || stride_bytes != width * 3
        || tightly_packed_bgr.size() != checked_bgr_size(stride_bytes, height)) {
        throw std::invalid_argument(
            "OpenCV Haar detector requires tightly packed BGR888 within the "
            "4096-axis and 8388608-pixel input limits"
        );
    }

    // This local Mat is only a header over the Rust-owned slice. Its automatic
    // lifetime ends on success and every exception path; no detector member
    // can retain the borrowed pixels.
    const cv::Mat bgr_view(
        static_cast<int>(height),
        static_cast<int>(width),
        CV_8UC3,
        const_cast<uint8_t*>(tightly_packed_bgr.data()),
        static_cast<size_t>(stride_bytes)
    );
    cv::cvtColor(bgr_view, impl_->grayscale, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(impl_->grayscale, impl_->equalized_grayscale);

    RawHaarFaceDetectionSource source = RawHaarFaceDetectionSource::Frontal;
    bool mirrored = false;
    if (!impl_->run(
            impl_->frontal,
            impl_->equalized_grayscale,
            impl_->frontal_minimum_neighbors
        )) {
        source = RawHaarFaceDetectionSource::Profile;
        if (!impl_->run(
                impl_->profile,
                impl_->equalized_grayscale,
                impl_->profile_minimum_neighbors
            )) {
            cv::flip(
                impl_->equalized_grayscale,
                impl_->mirrored_equalized_grayscale,
                1
            );
            source = RawHaarFaceDetectionSource::MirroredProfile;
            mirrored = true;
            impl_->run(
                impl_->profile,
                impl_->mirrored_equalized_grayscale,
                impl_->profile_minimum_neighbors
            );
        }
    }

    // Return the complete first-nonempty source set. Rust validates every
    // rectangle and weight before applying its deterministic configured cap.
    RawHaarFaceDetectionBatch output;
    output.detections.reserve(impl_->rectangles.size());
    for (size_t index = 0; index < impl_->rectangles.size(); ++index) {
        const auto& rectangle = impl_->rectangles[index];
        RawHaarFaceDetection detection{};
        detection.x = mirrored
            ? static_cast<int64_t>(width)
                - static_cast<int64_t>(rectangle.x)
                - static_cast<int64_t>(rectangle.width)
            : static_cast<int64_t>(rectangle.x);
        detection.y = static_cast<int64_t>(rectangle.y);
        detection.width = static_cast<int64_t>(rectangle.width);
        detection.height = static_cast<int64_t>(rectangle.height);
        detection.detector_level_weight = impl_->level_weights[index];
        detection.source = source;
        output.detections.push_back(std::move(detection));
    }
    return output;
}

std::unique_ptr<OpenCvHaarFaceDetector> create_opencv_haar_face_detector(
    rust::Slice<const uint8_t> frontal_cascade_xml,
    rust::Slice<const uint8_t> profile_cascade_xml,
    const RawOpenCvHaarFaceDetectorConfig& config
) {
    return std::make_unique<OpenCvHaarFaceDetector>(
        frontal_cascade_xml,
        profile_cascade_xml,
        config
    );
}

}  // namespace oak
}  // namespace kiko
