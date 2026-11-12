/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Kinect recordings: writing them, and playing one back as a camera.
// From Kixenia by Davi Carlos (github.com/davicarlos/kixenia), BSD.
//
// A recording is a directory:
// - calibration.bin: "XKCAL001", the 116 bytes of registration the camera
//   reports, then dcmos_emitter_distance, dcmos_rcmos_distance,
//   reference_distance and reference_pixel_size (host floats) and const_shift
//   (host uint16), all that PsCam codes 14 and 24 are built from;
// - depth_NNNNN.bin: a depth frame as the camera sends it, 11 bits per pixel
//   packed (kDepthFrameBytes);
// - color_NNNNN.bin: the color frame taken with it, Bayer (kColorFrameBytes).
//   Optional: a recording made before the first color frame plays black.
// Frames are numbered from 0 without gaps.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/mutex.h"
#include "xenia/base/threading.h"
#include "xenia/hid/kinect/usb_camera.h"

namespace xe {
namespace hid {
namespace kinect {

namespace {

constexpr char kCalibrationMagic[8] = {'X', 'K', 'C', 'A', 'L', '0', '0', '1'};
// The camera's 30 frames per second.
constexpr auto kFramePeriod = std::chrono::microseconds(33333);

std::filesystem::path FramePath(const std::filesystem::path& directory,
                                const char* kind, uint32_t index) {
  return directory / fmt::format("{}_{:05}.bin", kind, index);
}

bool WriteFile(const std::filesystem::path& path, const void* data,
               size_t size) {
  FILE* file = xe::filesystem::OpenFile(path, "wb");
  if (!file) {
    return false;
  }
  const bool ok = fwrite(data, 1, size, file) == size;
  fclose(file);
  return ok;
}

bool ReadFile(const std::filesystem::path& path, std::vector<uint8_t>& out,
              size_t size) {
  FILE* file = xe::filesystem::OpenFile(path, "rb");
  if (!file) {
    return false;
  }
  out.resize(size);
  const bool ok = fread(out.data(), 1, size, file) == size;
  fclose(file);
  return ok;
}

// Plays the frames in order at 30 per second, over and over, as a camera would
// deliver them.
class RecordingCamera : public UsbCamera {
 public:
  ~RecordingCamera() override {
    running_ = false;
    if (thread_) {
      xe::threading::Wait(thread_.get(), false);
    }
  }

  bool Initialize(const std::filesystem::path& directory,
                  std::function<void()> on_frame) {
    directory_ = directory;
    on_frame_ = std::move(on_frame);
    std::vector<uint8_t> calibration_bytes;
    constexpr size_t kCalibrationBytes =
        sizeof(kCalibrationMagic) + 116 + 4 * sizeof(float) + sizeof(uint16_t);
    if (!ReadFile(directory / "calibration.bin", calibration_bytes,
                  kCalibrationBytes) ||
        std::memcmp(calibration_bytes.data(), kCalibrationMagic,
                    sizeof(kCalibrationMagic)) != 0) {
      XELOGE("Kinect: {} holds no recording calibration",
             xe::path_to_utf8(directory));
      return false;
    }
    const uint8_t* in = calibration_bytes.data() + sizeof(kCalibrationMagic);
    std::memcpy(calibration_.registration.data(), in, 116);
    in += 116;
    std::memcpy(&calibration_.dcmos_emitter_distance, in, 4);
    std::memcpy(&calibration_.dcmos_rcmos_distance, in + 4, 4);
    std::memcpy(&calibration_.reference_distance, in + 8, 4);
    std::memcpy(&calibration_.reference_pixel_size, in + 12, 4);
    std::memcpy(&calibration_.const_shift, in + 16, 2);
    // The frames run up to the first number missing.
    while (std::filesystem::exists(FramePath(directory, "depth", frame_count_))) {
      ++frame_count_;
    }
    if (!frame_count_) {
      XELOGE("Kinect: {} holds no recorded frames",
             xe::path_to_utf8(directory));
      return false;
    }
    running_ = true;
    thread_ = xe::threading::Thread::Create({}, [this]() { Play(); });
    if (!thread_) {
      return false;
    }
    thread_->set_name("Kinect Recording");
    XELOGI("Kinect: playing {} recorded frames from {}", frame_count_,
           xe::path_to_utf8(directory));
    return true;
  }

  const UsbCameraCalibration& calibration() const override {
    return calibration_;
  }

  uint32_t CopyDepthFrame(uint32_t after,
                          std::vector<uint8_t>& packed_depth) override {
    std::lock_guard<xe_mutex> lock(mutex_);
    if (!sequence_ || sequence_ == after || depth_.empty()) {
      return 0;
    }
    packed_depth = depth_;
    return sequence_;
  }

  uint32_t CopyColorFrame(uint32_t after,
                          std::vector<uint8_t>& bayer) override {
    std::lock_guard<xe_mutex> lock(mutex_);
    if (!sequence_ || sequence_ == after || color_.empty()) {
      return 0;
    }
    bayer = color_;
    return sequence_;
  }

 private:
  void Play() {
    uint32_t index = 0;
    auto next = std::chrono::steady_clock::now();
    std::vector<uint8_t> depth, color;
    while (running_) {
      const bool depth_ok =
          ReadFile(FramePath(directory_, "depth", index), depth,
                   kDepthFrameBytes);
      if (!ReadFile(FramePath(directory_, "color", index), color,
                    kColorFrameBytes)) {
        color.assign(kColorFrameBytes, 0);
      }
      // A frame becomes the newest one only with its depth, which titles
      // can't do without.
      if (depth_ok) {
        std::lock_guard<xe_mutex> lock(mutex_);
        depth_.swap(depth);
        color_.swap(color);
        ++sequence_;
      }
      if (on_frame_) {
        on_frame_();
      }
      // Back to the start after the last frame; on a fixed schedule, so slow
      // file reads don't stretch the recording.
      index = (index + 1) % frame_count_;
      next += kFramePeriod;
      std::this_thread::sleep_until(next);
    }
  }

  std::filesystem::path directory_;
  std::function<void()> on_frame_;
  UsbCameraCalibration calibration_;
  uint32_t frame_count_ = 0;
  std::atomic<bool> running_ = false;
  std::unique_ptr<xe::threading::Thread> thread_;
  xe_mutex mutex_;
  std::vector<uint8_t> depth_;
  std::vector<uint8_t> color_;
  uint32_t sequence_ = 0;
};

}  // namespace

std::unique_ptr<UsbCamera> UsbCamera::OpenRecording(
    const std::filesystem::path& directory, std::function<void()> on_frame) {
  auto camera = std::make_unique<RecordingCamera>();
  if (!camera->Initialize(directory, std::move(on_frame))) {
    return nullptr;
  }
  return camera;
}

bool WriteRecordingCalibration(const std::filesystem::path& directory,
                               const UsbCameraCalibration& calibration) {
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  std::vector<uint8_t> bytes(kCalibrationMagic,
                             kCalibrationMagic + sizeof(kCalibrationMagic));
  bytes.insert(bytes.end(), calibration.registration.begin(),
               calibration.registration.end());
  auto append = [&bytes](const void* data, size_t size) {
    auto begin = static_cast<const uint8_t*>(data);
    bytes.insert(bytes.end(), begin, begin + size);
  };
  append(&calibration.dcmos_emitter_distance, 4);
  append(&calibration.dcmos_rcmos_distance, 4);
  append(&calibration.reference_distance, 4);
  append(&calibration.reference_pixel_size, 4);
  append(&calibration.const_shift, 2);
  return WriteFile(directory / "calibration.bin", bytes.data(), bytes.size());
}

bool WriteRecordingFrame(const std::filesystem::path& directory,
                         uint32_t index,
                         const std::vector<uint8_t>& packed_depth,
                         const std::vector<uint8_t>& bayer) {
  bool ok = WriteFile(FramePath(directory, "depth", index), packed_depth.data(),
                      packed_depth.size());
  if (!bayer.empty()) {
    ok &= WriteFile(FramePath(directory, "color", index), bayer.data(),
                    bayer.size());
  }
  return ok;
}

}  // namespace kinect
}  // namespace hid
}  // namespace xe
