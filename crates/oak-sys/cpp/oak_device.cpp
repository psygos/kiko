// oak_device.cpp - OAK-D FFI bridge implementation

#include "oak-sys/src/lib.rs.h"
#include "oak_device.hpp"

#include <depthai/build/version.hpp>

#include <chrono>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace kiko {
namespace oak {

namespace {

bool checked_payload_size(
    uint32_t width,
    uint32_t height,
    uint32_t bytes_per_pixel,
    size_t& size
) noexcept {
    if (width == 0 || height == 0 || bytes_per_pixel == 0) return false;

    const auto w = static_cast<size_t>(width);
    const auto h = static_cast<size_t>(height);
    const auto bpp = static_cast<size_t>(bytes_per_pixel);
    const auto max = std::numeric_limits<size_t>::max();
    if (w > max / h) return false;
    const auto pixels = w * h;
    if (pixels > max / bpp) return false;
    size = pixels * bpp;
    return true;
}

bool checked_row_size(uint32_t width, uint32_t channels, uint32_t& size) noexcept {
    if (width == 0 || channels == 0) return false;
    if (width > std::numeric_limits<uint32_t>::max() / channels) return false;
    size = width * channels;
    return true;
}

Intrinsics frame_intrinsics(const std::shared_ptr<dai::ImgFrame>& frame) {
    const auto matrix = frame->getTransformation().getIntrinsicMatrix();
    Intrinsics intrinsics{};
    intrinsics.m00 = matrix[0][0];
    intrinsics.m01 = matrix[0][1];
    intrinsics.m02 = matrix[0][2];
    intrinsics.m10 = matrix[1][0];
    intrinsics.m11 = matrix[1][1];
    intrinsics.m12 = matrix[1][2];
    intrinsics.m20 = matrix[2][0];
    intrinsics.m21 = matrix[2][1];
    intrinsics.m22 = matrix[2][2];
    intrinsics.width = frame->getWidth();
    intrinsics.height = frame->getHeight();
    return intrinsics;
}

template <typename BridgeFrame>
void set_camera_capture_metadata(
    BridgeFrame& output,
    const std::shared_ptr<dai::ImgFrame>& input
) {
    output.device_capture_sequence = input->getSequenceNum();
    output.timestamp.device_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        input->getTimestampDevice(dai::CameraExposureOffset::MIDDLE).time_since_epoch()
    ).count();
    output.exposure_time_us = input->getExposureTime().count();
}

std::optional<ImuAccuracy> imu_accuracy(dai::IMUReport::Accuracy accuracy) noexcept {
    switch (accuracy) {
        case dai::IMUReport::Accuracy::UNRELIABLE:
            return ImuAccuracy::Unreliable;
        case dai::IMUReport::Accuracy::LOW:
            return ImuAccuracy::Low;
        case dai::IMUReport::Accuracy::MEDIUM:
            return ImuAccuracy::Medium;
        case dai::IMUReport::Accuracy::HIGH:
            return ImuAccuracy::High;
        default:
            return std::nullopt;
    }
}

dai::UsbSpeed to_depthai_usb_speed(UsbSpeed speed) {
    switch (speed) {
        case UsbSpeed::Low:
            return dai::UsbSpeed::LOW;
        case UsbSpeed::Full:
            return dai::UsbSpeed::FULL;
        case UsbSpeed::High:
            return dai::UsbSpeed::HIGH;
        case UsbSpeed::Super:
            return dai::UsbSpeed::SUPER;
        case UsbSpeed::SuperPlus:
            return dai::UsbSpeed::SUPER_PLUS;
        case UsbSpeed::Unknown:
        case UsbSpeed::Unrecognized:
        default:
            throw std::invalid_argument("UNKNOWN is not a valid requested OAK USB speed");
    }
}

UsbSpeed from_depthai_usb_speed(dai::UsbSpeed speed) {
    switch (speed) {
        case dai::UsbSpeed::UNKNOWN:
            return UsbSpeed::Unknown;
        case dai::UsbSpeed::LOW:
            return UsbSpeed::Low;
        case dai::UsbSpeed::FULL:
            return UsbSpeed::Full;
        case dai::UsbSpeed::HIGH:
            return UsbSpeed::High;
        case dai::UsbSpeed::SUPER:
            return UsbSpeed::Super;
        case dai::UsbSpeed::SUPER_PLUS:
            return UsbSpeed::SuperPlus;
        default:
            return UsbSpeed::Unrecognized;
    }
}

CameraSocket from_depthai_camera_socket(dai::CameraBoardSocket socket) noexcept {
    if (socket == dai::CameraBoardSocket::CAM_A) return CameraSocket::CameraA;
    if (socket == dai::CameraBoardSocket::CAM_B) return CameraSocket::CameraB;
    if (socket == dai::CameraBoardSocket::CAM_C) return CameraSocket::CameraC;
    return CameraSocket::Unrecognized;
}

template <typename T>
T next_sequence(std::atomic<T>& sequence) {
    auto current = sequence.load(std::memory_order_relaxed);
    while (true) {
        if (current == std::numeric_limits<T>::max()) {
            throw std::overflow_error("OAK host-delivery sequence exhausted");
        }
        if (sequence.compare_exchange_weak(
                current,
                static_cast<T>(current + 1),
                std::memory_order_relaxed,
                std::memory_order_relaxed
            )) {
            return current;
        }
    }
}

template <std::size_t Rows, std::size_t Columns>
rust::Vec<float> flatten_calibration_matrix(
    const std::vector<std::vector<float>>& matrix,
    const char* matrix_name
) {
    if (matrix.size() != Rows) {
        throw std::runtime_error(
            std::string("DepthAI ") + matrix_name + " has "
            + std::to_string(matrix.size()) + " rows; expected "
            + std::to_string(Rows)
        );
    }

    rust::Vec<float> flattened;
    flattened.reserve(Rows * Columns);
    for (std::size_t row = 0; row < Rows; ++row) {
        if (matrix[row].size() != Columns) {
            throw std::runtime_error(
                std::string("DepthAI ") + matrix_name + " row "
                + std::to_string(row) + " has "
                + std::to_string(matrix[row].size()) + " columns; expected "
                + std::to_string(Columns)
            );
        }
        for (std::size_t column = 0; column < Columns; ++column) {
            flattened.push_back(matrix[row][column]);
        }
    }
    return flattened;
}

}  // namespace

DepthAiBuildMetadata depthai_build_metadata() {
    DepthAiBuildMetadata metadata{};
    metadata.sdk_version = rust::String(dai::build::VERSION);
    metadata.sdk_commit = rust::String(dai::build::COMMIT);
    metadata.embedded_device_artifact_version = rust::String(dai::build::DEVICE_VERSION);
    metadata.embedded_bootloader_artifact_version = rust::String(
        dai::build::BOOTLOADER_VERSION
    );
    return metadata;
}

rust::Vec<DeviceInfo> list_devices() {
    rust::Vec<DeviceInfo> devices;
    for (const auto& info : dai::Device::getAllConnectedDevices()) {
        DeviceInfo dev;
        dev.device_id = rust::String(info.deviceId);
        dev.name = rust::String(info.name);
        switch (info.state) {
            case X_LINK_UNBOOTED:
                dev.state = DeviceState::Available;
                break;
            case X_LINK_BOOTED:
                dev.state = DeviceState::InUse;
                break;
            case X_LINK_BOOTLOADER:
                dev.state = DeviceState::Bootloader;
                break;
            default:
                dev.state = DeviceState::Unknown;
                break;
        }
        devices.push_back(std::move(dev));
    }
    return devices;
}

std::unique_ptr<OakDevice> create_device(rust::Str selector, const DeviceConfig& config) {
    return std::make_unique<OakDevice>(config, std::string(selector.data(), selector.size()));
}

OakDevice::OakDevice(const DeviceConfig& config, const std::string& selector)
    : rgb_enabled_(config.rgb_enabled)
    , rgb_width_(config.rgb_width)
    , rgb_height_(config.rgb_height)
    , mono_enabled_(config.mono_enabled)
    , mono_width_(config.mono_width)
    , mono_height_(config.mono_height)
    , mono_rectified_(config.mono_rectified)
    , depth_enabled_(config.depth_enabled)
    , imu_enabled_(config.imu_enabled)
{
    if (selector.empty()) {
        throw std::invalid_argument("OAK device selector must be a nonempty exact MXID");
    }

    const auto lookup = dai::Device::getDeviceById(selector);
    if (!std::get<0>(lookup)) {
        throw std::invalid_argument("no DepthAI device has exact id '" + selector + "'");
    }
    const auto& device_info = std::get<1>(lookup);
    if (device_info.deviceId != selector) {
        throw std::runtime_error(
            "DepthAI exact-id lookup returned '" + device_info.deviceId
            + "' for requested id '" + selector + "'"
        );
    }
    // DepthAI v3's `(DeviceInfo, UsbSpeed)` constructor binds the requested
    // maximum speed to this exact MXID. Passing the selected Device into the
    // Pipeline prevents an implicit first/default-device reopen.
    auto selected_device = std::make_shared<dai::Device>(
        device_info,
        to_depthai_usb_speed(config.maximum_usb_speed)
    );
    pipeline_ = std::make_unique<dai::Pipeline>(std::move(selected_device));

    if (rgb_enabled_) {
        auto cam = pipeline_->create<dai::node::Camera>();
        cam->build(dai::CameraBoardSocket::CAM_A);
        auto* output = cam->requestOutput(
            std::make_pair(rgb_width_, rgb_height_),
            dai::ImgFrame::Type::BGR888i,
            dai::ImgResizeMode::CROP,
            static_cast<float>(config.rgb_fps)
        );
        rgb_queue_ = output->createOutputQueue(config.queue_size, config.queue_blocking);
    }

    if (mono_enabled_ || depth_enabled_) {
        auto left = pipeline_->create<dai::node::Camera>();
        auto right = pipeline_->create<dai::node::Camera>();
        left->build(dai::CameraBoardSocket::CAM_B);
        right->build(dai::CameraBoardSocket::CAM_C);

        auto fps = static_cast<float>(mono_enabled_ ? config.mono_fps : config.depth_fps);
        auto w = mono_enabled_ ? mono_width_ : config.depth_width;
        auto h = mono_enabled_ ? mono_height_ : config.depth_height;

        auto* leftOut = left->requestOutput({w, h}, dai::ImgFrame::Type::GRAY8, dai::ImgResizeMode::CROP, fps);
        auto* rightOut = right->requestOutput({w, h}, dai::ImgFrame::Type::GRAY8, dai::ImgResizeMode::CROP, fps);

        if (mono_enabled_ && !mono_rectified_) {
            mono_left_queue_ = leftOut->createOutputQueue(config.queue_size, config.queue_blocking);
            mono_right_queue_ = rightOut->createOutputQueue(config.queue_size, config.queue_blocking);
        }

        if (depth_enabled_ || mono_rectified_) {
            auto stereo = pipeline_->create<dai::node::StereoDepth>();
            leftOut->link(stereo->left);
            rightOut->link(stereo->right);
            stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::DEFAULT);
            stereo->initialConfig->setDepthUnit(dai::DepthUnit::MILLIMETER);
            stereo->enableDistortionCorrection(true);
            stereo->setRectifyEdgeFillColor(0);
            // Depth pixels need an explicit optical-frame contract. Do not rely on
            // DepthAI's default alignment because the selected calibration and pose
            // must describe the same pixel grid.
            switch (config.depth_alignment) {
                case DepthAlignment::RectifiedLeft:
                    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_B);
                    break;
                case DepthAlignment::RectifiedRight:
                    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_C);
                    break;
                case DepthAlignment::Rgb:
                    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);
                    break;
                default:
                    throw std::invalid_argument("unsupported depth alignment");
            }

            if (mono_rectified_) {
                mono_left_queue_ = stereo->rectifiedLeft.createOutputQueue(config.queue_size, config.queue_blocking);
                mono_right_queue_ = stereo->rectifiedRight.createOutputQueue(config.queue_size, config.queue_blocking);
            }

            if (depth_enabled_) {
                depth_queue_ = stereo->depth.createOutputQueue(config.queue_size, config.queue_blocking);
            }
        }
    }

    if (imu_enabled_) {
        auto imu = pipeline_->create<dai::node::IMU>();
        imu->enableIMUSensor({dai::IMUSensor::ACCELEROMETER_RAW, dai::IMUSensor::GYROSCOPE_RAW}, config.imu_rate_hz);
        imu->setBatchReportThreshold(1);
        imu->setMaxBatchReports(50);
        imu_queue_ = imu->out.createOutputQueue(config.queue_size, config.queue_blocking);
    }

    pipeline_->start();

    const auto device = pipeline_->getDefaultDevice();
    if (!device) throw std::runtime_error("DepthAI pipeline started without a default device");
    const auto actual_device_id = device->getDeviceId();
    if (actual_device_id != selector) {
        throw std::runtime_error(
            "DepthAI pipeline opened device '" + actual_device_id
            + "' instead of requested exact id '" + selector + "'"
        );
    }
    connected_mxid_ = actual_device_id;
    discovery_transport_name_ = device_info.name;
    eeprom_device_name_ = device->getDeviceName();
    product_name_ = device->getProductName();
    calibration_ = device->readCalibration();

    connected_ = true;
}

OakDevice::~OakDevice() noexcept {
    if (closed_) return;
    try {
        close();
    } catch (...) {
        // An explicit Rust `Device::close` reports this error. Destruction is
        // necessarily best-effort and must not throw across the C++ ABI.
    }
}

bool OakDevice::is_connected() const noexcept {
    return connected_ && !closed_;
}

UsbSpeed OakDevice::get_usb_speed() const {
    if (!is_connected()) {
        throw std::runtime_error("cannot read USB speed from a disconnected OAK pipeline");
    }
    const auto device = pipeline_->getDefaultDevice();
    if (!device) {
        throw std::runtime_error("DepthAI pipeline has no selected device for USB-speed readback");
    }
    return from_depthai_usb_speed(device->getUsbSpeed());
}

void OakDevice::close() {
    if (closed_.exchange(true)) return;
    connected_ = false;
    if (pipeline_) pipeline_->stop();
}

ImageFrameResult OakDevice::try_get_image(
    StreamId stream,
    bool enabled,
    const std::shared_ptr<dai::MessageQueue>& queue,
    std::atomic<uint64_t>& host_delivery_sequence,
    dai::ImgFrame::Type expected_type,
    uint32_t channels,
    uint32_t timeout_ms
) {
    ImageFrameResult result{};
    result.frame.stream = stream;

    if (!is_connected()) { result.status = FrameStatus::Disconnected; return result; }
    if (!enabled) { result.status = FrameStatus::StreamNotEnabled; return result; }
    if (!queue) throw std::logic_error("enabled DepthAI image stream has no output queue");

    bool timedout = false;
    std::shared_ptr<dai::ImgFrame> msg;
    try {
        msg = queue->get<dai::ImgFrame>(std::chrono::milliseconds(timeout_ms), timedout);
    } catch (const dai::MessageQueue::QueueException&) {
        connected_ = false;
        result.status = FrameStatus::Disconnected;
        return result;
    }
    if (timedout) {
        result.status = timeout_ms == 0 ? FrameStatus::QueueEmpty : FrameStatus::Timeout;
        return result;
    }
    if (!msg) { result.status = FrameStatus::Corrupt; return result; }

    if (msg->getType() != expected_type || !msg->validateTransformations()) {
        result.status = FrameStatus::Corrupt;
        return result;
    }

    set_camera_capture_metadata(result.frame, msg);
    result.frame.width = msg->getWidth();
    result.frame.height = msg->getHeight();

    size_t packed_size = 0;
    if (!checked_payload_size(result.frame.width, result.frame.height, channels, packed_size)
        || !checked_row_size(result.frame.width, channels, result.frame.stride_bytes)) {
        result.status = FrameStatus::Corrupt;
        return result;
    }

    const auto source_stride = msg->getStride();
    size_t source_size = 0;
    if (source_stride < result.frame.stride_bytes
        || !checked_payload_size(source_stride, result.frame.height, 1, source_size)) {
        result.status = FrameStatus::Corrupt;
        return result;
    }

    const auto data = msg->getData();
    if (data.size() != source_size) {
        result.status = FrameStatus::Corrupt;
        return result;
    }

    result.frame.intrinsics = frame_intrinsics(msg);
    result.frame.data.reserve(packed_size);
    if (source_stride == result.frame.stride_bytes) {
        for (auto byte : data) result.frame.data.push_back(byte);
    } else {
        for (uint32_t row = 0; row < result.frame.height; ++row) {
            const auto row_offset = static_cast<size_t>(row) * source_stride;
            for (uint32_t column = 0; column < result.frame.stride_bytes; ++column) {
                result.frame.data.push_back(data[row_offset + column]);
            }
        }
    }
    result.frame.host_delivery_sequence = next_sequence(host_delivery_sequence);
    result.status = FrameStatus::Ok;

    return result;
}

ImageFrameResult OakDevice::try_get_rgb(uint32_t timeout_ms) {
    return try_get_image(
        StreamId::Rgb,
        rgb_enabled_,
        rgb_queue_,
        rgb_seq_,
        dai::ImgFrame::Type::BGR888i,
        3,
        timeout_ms
    );
}

ImageFrameResult OakDevice::try_get_mono_left(uint32_t timeout_ms) {
    // StereoDepth documents both rectified outputs as RAW8. Direct camera
    // grayscale output is GRAY8, so the expected wire type must follow the
    // graph selected at the parsed DeviceConfig boundary.
    const auto expected_type = mono_rectified_
        ? dai::ImgFrame::Type::RAW8
        : dai::ImgFrame::Type::GRAY8;
    return try_get_image(
        StreamId::MonoLeft,
        mono_enabled_,
        mono_left_queue_,
        mono_left_seq_,
        expected_type,
        1,
        timeout_ms
    );
}

ImageFrameResult OakDevice::try_get_mono_right(uint32_t timeout_ms) {
    const auto expected_type = mono_rectified_
        ? dai::ImgFrame::Type::RAW8
        : dai::ImgFrame::Type::GRAY8;
    return try_get_image(
        StreamId::MonoRight,
        mono_enabled_,
        mono_right_queue_,
        mono_right_seq_,
        expected_type,
        1,
        timeout_ms
    );
}

DepthFrameResult OakDevice::try_get_depth(uint32_t timeout_ms) {
    DepthFrameResult result{};

    if (!is_connected()) { result.status = FrameStatus::Disconnected; return result; }
    if (!depth_enabled_) { result.status = FrameStatus::StreamNotEnabled; return result; }
    if (!depth_queue_) throw std::logic_error("enabled DepthAI depth stream has no output queue");

    bool timedout = false;
    std::shared_ptr<dai::ImgFrame> msg;
    try {
        msg = depth_queue_->get<dai::ImgFrame>(std::chrono::milliseconds(timeout_ms), timedout);
    } catch (const dai::MessageQueue::QueueException&) {
        connected_ = false;
        result.status = FrameStatus::Disconnected;
        return result;
    }
    if (timedout) {
        result.status = timeout_ms == 0 ? FrameStatus::QueueEmpty : FrameStatus::Timeout;
        return result;
    }
    if (!msg) { result.status = FrameStatus::Corrupt; return result; }

    if (msg->getType() != dai::ImgFrame::Type::RAW16 || !msg->validateTransformations()) {
        result.status = FrameStatus::Corrupt;
        return result;
    }

    set_camera_capture_metadata(result.frame, msg);
    result.frame.width = msg->getWidth();
    result.frame.height = msg->getHeight();

    size_t packed_size = 0;
    uint32_t packed_stride = 0;
    if (!checked_payload_size(result.frame.width, result.frame.height, 2, packed_size)
        || !checked_row_size(result.frame.width, 2, packed_stride)) {
        result.status = FrameStatus::Corrupt;
        return result;
    }

    const auto source_stride = msg->getStride();
    size_t source_size = 0;
    if (source_stride < packed_stride
        || !checked_payload_size(source_stride, result.frame.height, 1, source_size)) {
        result.status = FrameStatus::Corrupt;
        return result;
    }

    const auto data = msg->getData();
    if (data.size() != source_size) {
        result.status = FrameStatus::Corrupt;
        return result;
    }

    result.frame.intrinsics = frame_intrinsics(msg);
    result.frame.data.reserve(packed_size / 2);
    for (uint32_t row = 0; row < result.frame.height; ++row) {
        const auto row_offset = static_cast<size_t>(row) * source_stride;
        for (uint32_t column = 0; column < packed_stride; column += 2) {
            const auto offset = row_offset + column;
            result.frame.data.push_back(
                static_cast<uint16_t>(data[offset])
                | (static_cast<uint16_t>(data[offset + 1]) << 8)
            );
        }
    }
    result.frame.host_delivery_sequence = next_sequence(depth_seq_);
    result.status = FrameStatus::Ok;

    return result;
}

ImuBatchResult OakDevice::get_imu_batch() {
    ImuBatchResult result{};

    if (!is_connected()) { result.status = ImuStatus::Disconnected; return result; }
    if (!imu_enabled_) { result.status = ImuStatus::Empty; return result; }
    if (!imu_queue_) throw std::logic_error("enabled DepthAI IMU stream has no output queue");

    std::vector<std::shared_ptr<dai::IMUData>> packets;
    try {
        packets = imu_queue_->tryGetAll<dai::IMUData>();
    } catch (const dai::MessageQueue::QueueException&) {
        connected_ = false;
        result.status = ImuStatus::Disconnected;
        return result;
    }
    if (packets.empty()) { result.status = ImuStatus::Empty; return result; }

    // tryGetAll<T> returns nullptr only when a queued message fails its
    // dynamic cast to T. With a batch threshold of one, a typed IMUData
    // message with no reports is likewise outside this pipeline contract.
    for (const auto& imuData : packets) {
        if (!imuData || imuData->packets.empty()) {
            result.status = ImuStatus::Corrupt;
            return result;
        }
        for (const auto& packet : imuData->packets) {
            if (!imu_accuracy(packet.acceleroMeter.accuracy).has_value()
                || !imu_accuracy(packet.gyroscope.accuracy).has_value()) {
                result.status = ImuStatus::Corrupt;
                return result;
            }
        }
    }

    for (const auto& imuData : packets) {
        for (const auto& p : imuData->packets) {
            ImuSample s{};
            s.accel_timestamp.device_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                p.acceleroMeter.getTimestampDevice().time_since_epoch()).count();
            s.gyro_timestamp.device_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                p.gyroscope.getTimestampDevice().time_since_epoch()).count();
            s.accel_x = p.acceleroMeter.x;
            s.accel_y = p.acceleroMeter.y;
            s.accel_z = p.acceleroMeter.z;
            s.accel_accuracy = imu_accuracy(p.acceleroMeter.accuracy).value();
            s.gyro_x = p.gyroscope.x;
            s.gyro_y = p.gyroscope.y;
            s.gyro_z = p.gyroscope.z;
            s.gyro_accuracy = imu_accuracy(p.gyroscope.accuracy).value();
            s.sequence = next_sequence(imu_seq_);
            result.batch.samples.push_back(s);
        }
    }

    result.status = result.batch.samples.empty() ? ImuStatus::Empty : ImuStatus::Ok;
    return result;
}

float OakDevice::get_stereo_baseline_m() const {
    return calibration_.getBaselineDistance(
        dai::CameraBoardSocket::CAM_B,
        dai::CameraBoardSocket::CAM_C,
        false,
        dai::LengthUnit::METER
    );
}

EepromCalibrationEvidence OakDevice::get_eeprom_calibration_evidence() const {
    const auto device = pipeline_->getDefaultDevice();
    if (!device) {
        throw std::runtime_error(
            "cannot read EEPROM calibration without the connected OAK device"
        );
    }
    // Unlike readCalibration(), readCalibration2() propagates EEPROM access
    // failures instead of substituting an empty/default handler.
    const auto eeprom_calibration = device->readCalibration2();
    EepromCalibrationEvidence evidence{};
    evidence.stereo_left_camera_socket = from_depthai_camera_socket(
        eeprom_calibration.getStereoLeftCameraId()
    );
    evidence.stereo_right_camera_socket = from_depthai_camera_socket(
        eeprom_calibration.getStereoRightCameraId()
    );
    evidence.imu_to_camera_b_m = flatten_calibration_matrix<4, 4>(
        eeprom_calibration.getImuToCameraExtrinsics(
            dai::CameraBoardSocket::CAM_B,
            false,
            dai::LengthUnit::METER
        ),
        "IMU-to-CAM_B extrinsics"
    );
    evidence.stereo_left_rectification_rotation_raw =
        flatten_calibration_matrix<3, 3>(
            eeprom_calibration.getStereoLeftRectificationRotation(),
            "stereo-left rectification rotation"
        );
    return evidence;
}

ConnectedDeviceIdentity OakDevice::get_connected_device_identity() const {
    ConnectedDeviceIdentity identity{};
    identity.mxid = rust::String(connected_mxid_);
    identity.discovery_transport_name = rust::String(discovery_transport_name_);
    identity.eeprom_device_name = rust::String(eeprom_device_name_);
    identity.product_name = rust::String(product_name_);
    return identity;
}

} // namespace oak
} // namespace kiko
