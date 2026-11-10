/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_HID_KINECT_FAKE_SKELETON_SOURCE_H_
#define XENIA_HID_KINECT_FAKE_SKELETON_SOURCE_H_

#include <atomic>
#include <cstdint>
#include <functional>

#include "xenia/hid/kinect/skeleton_source.h"

namespace xe {
namespace hid {
namespace kinect {

enum class FakeHandMotion { kStill, kCircle, kSweep };

struct FakeSkeletonConfig {
  float distance_m = 2.5f;       // User distance from the sensor (+z).
  float sensor_height_m = 1.0f;  // Sensor height above the floor.
  float lateral_m = 0.0f;        // Sideways offset of the user.
  float frame_rate_hz = 30.0f;
  // Applied to the user's RIGHT hand once raised.
  FakeHandMotion hand_motion = FakeHandMotion::kStill;
  float motion_period_s = 4.0f;
  float motion_radius_m = 0.25f;
};

// Deterministic fake user: standing facing the sensor, arms at the sides. The
// right hand can be raised and steered (for menu cursors) either by a scripted
// motion or manually via SetRightHandOffset (e.g. from keyboard/mouse input).
class FakeSkeletonSource : public SkeletonSource {
 public:
  using ClockFn = std::function<uint64_t()>;  // Microseconds, monotonic.

  explicit FakeSkeletonSource(FakeSkeletonConfig config = FakeSkeletonConfig(),
                              ClockFn clock_us = nullptr);

  bool Start() override;
  void Stop() override;
  bool TryGetLatestFrame(SkeletonFrame* out) override;

  // Raises the right hand to shoulder height plus this offset (meters, +x =
  // toward the user's right, +y = up). Thread-safe.
  void SetRightHandOffset(float x, float y);
  // Lowers the right hand back to the side (scripted motion still raises it).
  void ClearRightHand();

  // Pure function of its arguments and current settings, for tests.
  SkeletonFrame GenerateFrame(uint64_t frame_number,
                              uint64_t timestamp_us) const;

 private:
  FakeSkeletonConfig config_;
  ClockFn clock_us_;
  uint64_t start_us_ = 0;
  uint64_t last_frame_number_ = 0;
  bool running_ = false;
  std::atomic<bool> manual_raised_{false};
  std::atomic<float> manual_x_{0.0f};
  std::atomic<float> manual_y_{0.0f};
};

}  // namespace kinect
}  // namespace hid
}  // namespace xe

#endif  // XENIA_HID_KINECT_FAKE_SKELETON_SOURCE_H_
