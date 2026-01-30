// oak_device.cpp - OAK-D FFI bridge implementation

#include "oak-sys/src/lib.rs.h"
#include "oak_device.hpp"

namespace kiko {
namespace oak {

rust::Vec<DeviceInfo> list_devices() {
    rust::Vec<DeviceInfo> devices;
    for (const auto& info : dai::Device::getAllAvailableDevices()) {
        DeviceInfo dev;
        dev.device_id = rust::String(info.deviceId);
        dev.name = rust::String(info.name);
        switch (info.state) {
            case X_LINK_UNBOOTED:
            case X_LINK_BOOTED:
                dev.state = DeviceState::Available;
                break;
            case X_LINK_BOOTLOADER:
                dev.state = DeviceState::Bootloader;
                break;
            default:
                dev.state = DeviceState::Disconnected;
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
    pipeline_ = std::make_unique<dai::Pipeline>();

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
            stereo->enableDistortionCorrection(true);
            stereo->setRectifyEdgeFillColor(0);

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

    if (auto device = pipeline_->getDefaultDevice()) {
        calibration_ = device->readCalibration();
    }

    connected_ = true;
}

OakDevice::~OakDevice() {
    if (!closed_) close();
}

bool OakDevice::is_connected() const {
    return connected_ && !closed_;
}

void OakDevice::close() {
    closed_ = true;
    connected_ = false;
    if (pipeline_) pipeline_->stop();
}

ImageFrameResult OakDevice::try_get_rgb(uint32_t timeout_ms) {
    ImageFrameResult result{};
    result.frame.stream = StreamId::Rgb;

    if (!is_connected()) { result.status = FrameStatus::Disconnected; return result; }
    if (!rgb_enabled_) { result.status = FrameStatus::StreamNotEnabled; return result; }

    bool timedout = false;
    auto msg = rgb_queue_->get<dai::ImgFrame>(std::chrono::milliseconds(timeout_ms), timedout);
    if (timedout || !msg) { result.status = FrameStatus::Timeout; return result; }

    result.status = FrameStatus::Ok;
    result.frame.stream = StreamId::Rgb;
    result.frame.sequence = rgb_seq_.fetch_add(1);
    result.frame.timestamp.device_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        msg->getTimestampDevice().time_since_epoch()).count();
    result.frame.width = msg->getWidth();
    result.frame.height = msg->getHeight();
    result.frame.stride_bytes = msg->getWidth() * 3;

    auto data = msg->getData();
    result.frame.data.reserve(data.size());
    for (auto b : data) result.frame.data.push_back(b);

    return result;
}

ImageFrameResult OakDevice::try_get_mono_left(uint32_t timeout_ms) {
    ImageFrameResult result{};
    result.frame.stream = StreamId::MonoLeft;

    if (!is_connected()) { result.status = FrameStatus::Disconnected; return result; }
    if (!mono_enabled_) { result.status = FrameStatus::StreamNotEnabled; return result; }

    bool timedout = false;
    auto msg = mono_left_queue_->get<dai::ImgFrame>(std::chrono::milliseconds(timeout_ms), timedout);
    if (timedout || !msg) { result.status = FrameStatus::Timeout; return result; }

    result.status = FrameStatus::Ok;
    result.frame.sequence = mono_left_seq_.fetch_add(1);
    result.frame.timestamp.device_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        msg->getTimestampDevice().time_since_epoch()).count();
    result.frame.width = msg->getWidth();
    result.frame.height = msg->getHeight();
    result.frame.stride_bytes = msg->getWidth();

    auto data = msg->getData();
    result.frame.data.reserve(data.size());
    for (auto b : data) result.frame.data.push_back(b);

    return result;
}

ImageFrameResult OakDevice::try_get_mono_right(uint32_t timeout_ms) {
    ImageFrameResult result{};
    result.frame.stream = StreamId::MonoRight;

    if (!is_connected()) { result.status = FrameStatus::Disconnected; return result; }
    if (!mono_enabled_) { result.status = FrameStatus::StreamNotEnabled; return result; }

    bool timedout = false;
    auto msg = mono_right_queue_->get<dai::ImgFrame>(std::chrono::milliseconds(timeout_ms), timedout);
    if (timedout || !msg) { result.status = FrameStatus::Timeout; return result; }

    result.status = FrameStatus::Ok;
    result.frame.sequence = mono_right_seq_.fetch_add(1);
    result.frame.timestamp.device_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        msg->getTimestampDevice().time_since_epoch()).count();
    result.frame.width = msg->getWidth();
    result.frame.height = msg->getHeight();
    result.frame.stride_bytes = msg->getWidth();

    auto data = msg->getData();
    result.frame.data.reserve(data.size());
    for (auto b : data) result.frame.data.push_back(b);

    return result;
}

DepthFrameResult OakDevice::try_get_depth(uint32_t timeout_ms) {
    DepthFrameResult result{};
    result.frame.depth_scale = 0.001f;
    result.frame.min_depth_mm = 200;
    result.frame.max_depth_mm = 10000;

    if (!is_connected()) { result.status = FrameStatus::Disconnected; return result; }
    if (!depth_enabled_) { result.status = FrameStatus::StreamNotEnabled; return result; }

    bool timedout = false;
    auto msg = depth_queue_->get<dai::ImgFrame>(std::chrono::milliseconds(timeout_ms), timedout);
    if (timedout || !msg) { result.status = FrameStatus::Timeout; return result; }

    result.status = FrameStatus::Ok;
    result.frame.sequence = depth_seq_.fetch_add(1);
    result.frame.timestamp.device_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        msg->getTimestampDevice().time_since_epoch()).count();
    result.frame.width = msg->getWidth();
    result.frame.height = msg->getHeight();

    auto data = msg->getData();
    result.frame.data.reserve(data.size() / 2);
    for (size_t i = 0; i < data.size(); i += 2) {
        result.frame.data.push_back(static_cast<uint16_t>(data[i]) | (static_cast<uint16_t>(data[i+1]) << 8));
    }

    return result;
}

ImuBatchResult OakDevice::get_imu_batch() {
    ImuBatchResult result{};

    if (!is_connected()) { result.status = ImuStatus::Disconnected; return result; }
    if (!imu_enabled_) { result.status = ImuStatus::Empty; return result; }

    auto packets = imu_queue_->tryGetAll<dai::IMUData>();
    if (packets.empty()) { result.status = ImuStatus::Empty; return result; }

    result.status = ImuStatus::Ok;
    for (const auto& imuData : packets) {
        if (!imuData) continue;
        for (const auto& p : imuData->packets) {
            ImuSample s;
            s.sequence = imu_seq_.fetch_add(1);
            s.timestamp.device_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                p.acceleroMeter.getTimestampDevice().time_since_epoch()).count();
            s.accel_x = p.acceleroMeter.x;
            s.accel_y = p.acceleroMeter.y;
            s.accel_z = p.acceleroMeter.z;
            s.accel_accuracy = static_cast<ImuAccuracy>(static_cast<uint8_t>(p.acceleroMeter.accuracy));
            s.gyro_x = p.gyroscope.x;
            s.gyro_y = p.gyroscope.y;
            s.gyro_z = p.gyroscope.z;
            s.gyro_accuracy = static_cast<ImuAccuracy>(static_cast<uint8_t>(p.gyroscope.accuracy));
            result.batch.samples.push_back(s);
        }
    }

    return result;
}

Intrinsics OakDevice::get_rgb_intrinsics() const {
    Intrinsics intr{};
    intr.width = rgb_width_;
    intr.height = rgb_height_;
    try {
        auto i = calibration_.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, rgb_width_, rgb_height_);
        intr.fx = i[0][0]; intr.fy = i[1][1]; intr.cx = i[0][2]; intr.cy = i[1][2];
    } catch (...) {
        float scale = static_cast<float>(rgb_width_) / 640.0f;
        intr.fx = 517.22f * scale; intr.fy = 517.18f * scale;
        intr.cx = 316.91f * scale; intr.cy = 242.63f * scale;
    }
    return intr;
}

Intrinsics OakDevice::get_left_intrinsics() const {
    Intrinsics intr{};
    intr.width = mono_width_;
    intr.height = mono_height_;
    try {
        auto i = calibration_.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, mono_width_, mono_height_);
        if (mono_rectified_) {
            auto j = calibration_.getCameraIntrinsics(dai::CameraBoardSocket::CAM_C, mono_width_, mono_height_);
            intr.fx = 0.5f * (i[0][0] + j[0][0]);
            intr.fy = 0.5f * (i[1][1] + j[1][1]);
            intr.cx = 0.5f * (i[0][2] + j[0][2]);
            intr.cy = 0.5f * (i[1][2] + j[1][2]);
        } else {
            intr.fx = i[0][0]; intr.fy = i[1][1]; intr.cx = i[0][2]; intr.cy = i[1][2];
        }
    } catch (...) {
        float scale = static_cast<float>(mono_width_) / 640.0f;
        intr.fx = 398.17f * scale; intr.fy = 398.19f * scale;
        intr.cx = 308.64f * scale; intr.cy = 239.88f * scale;
    }
    return intr;
}

Intrinsics OakDevice::get_right_intrinsics() const {
    Intrinsics intr{};
    intr.width = mono_width_;
    intr.height = mono_height_;
    try {
        auto i = calibration_.getCameraIntrinsics(dai::CameraBoardSocket::CAM_C, mono_width_, mono_height_);
        if (mono_rectified_) {
            auto j = calibration_.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, mono_width_, mono_height_);
            intr.fx = 0.5f * (i[0][0] + j[0][0]);
            intr.fy = 0.5f * (i[1][1] + j[1][1]);
            intr.cx = 0.5f * (i[0][2] + j[0][2]);
            intr.cy = 0.5f * (i[1][2] + j[1][2]);
        } else {
            intr.fx = i[0][0]; intr.fy = i[1][1]; intr.cx = i[0][2]; intr.cy = i[1][2];
        }
    } catch (...) {
        float scale = static_cast<float>(mono_width_) / 640.0f;
        intr.fx = 396.99f * scale; intr.fy = 397.00f * scale;
        intr.cx = 326.85f * scale; intr.cy = 234.89f * scale;
    }
    return intr;
}

float OakDevice::get_stereo_baseline_m() const {
    try {
        return calibration_.getBaselineDistance(dai::CameraBoardSocket::CAM_B, dai::CameraBoardSocket::CAM_C) / 100.0f;
    } catch (...) {
        return 0.075f;
    }
}

} // namespace oak
} // namespace kiko
