#pragma once
// oak_device.hpp - OAK-D FFI bridge header

#include <cstdint>
#include <memory>
#include <string>
#include <atomic>

#include "rust/cxx.h"
#include "depthai/depthai.hpp"

namespace kiko {
namespace oak {

// Forward declarations from cxx-generated code
struct DeviceConfig;
struct DeviceInfo;
struct Timestamp;
struct ImageFrame;
struct DepthFrame;
struct ImuSample;
struct ImuBatch;
struct Intrinsics;
struct ImageFrameResult;
struct DepthFrameResult;
struct ImuBatchResult;

enum class DeviceState : uint8_t;
enum class StreamId : uint8_t;
enum class FrameStatus : uint8_t;
enum class ImuStatus : uint8_t;
enum class ImuAccuracy : uint8_t;

class OakDevice {
public:
    explicit OakDevice(const DeviceConfig& config, const std::string& selector);
    ~OakDevice();

    OakDevice(const OakDevice&) = delete;
    OakDevice& operator=(const OakDevice&) = delete;

    bool is_connected() const;

    ImageFrameResult try_get_rgb(uint32_t timeout_ms);
    ImageFrameResult try_get_mono_left(uint32_t timeout_ms);
    ImageFrameResult try_get_mono_right(uint32_t timeout_ms);
    DepthFrameResult try_get_depth(uint32_t timeout_ms);
    ImuBatchResult get_imu_batch();

    Intrinsics get_rgb_intrinsics() const;
    Intrinsics get_left_intrinsics() const;
    Intrinsics get_right_intrinsics() const;
    float get_stereo_baseline_m() const;

    void close();

private:
    bool rgb_enabled_;
    uint32_t rgb_width_;
    uint32_t rgb_height_;

    bool mono_enabled_;
    uint32_t mono_width_;
    uint32_t mono_height_;
    bool mono_rectified_;

    bool depth_enabled_;
    bool imu_enabled_;

    std::atomic<bool> connected_{false};
    std::atomic<bool> closed_{false};

    std::atomic<uint64_t> rgb_seq_{0};
    std::atomic<uint64_t> mono_left_seq_{0};
    std::atomic<uint64_t> mono_right_seq_{0};
    std::atomic<uint64_t> depth_seq_{0};
    std::atomic<uint32_t> imu_seq_{0};

    std::unique_ptr<dai::Pipeline> pipeline_;
    std::shared_ptr<dai::MessageQueue> rgb_queue_;
    std::shared_ptr<dai::MessageQueue> mono_left_queue_;
    std::shared_ptr<dai::MessageQueue> mono_right_queue_;
    std::shared_ptr<dai::MessageQueue> depth_queue_;
    std::shared_ptr<dai::MessageQueue> imu_queue_;
    dai::CalibrationHandler calibration_;
};

rust::Vec<DeviceInfo> list_devices();
std::unique_ptr<OakDevice> create_device(rust::Str selector, const DeviceConfig& config);

} // namespace oak
} // namespace kiko
