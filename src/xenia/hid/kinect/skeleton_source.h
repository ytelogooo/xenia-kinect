/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_HID_KINECT_SKELETON_SOURCE_H_
#define XENIA_HID_KINECT_SKELETON_SOURCE_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace xe {
namespace hid {
namespace kinect {

// Host-side, canonical skeleton model. This is NOT the guest-visible layout:
// the HLE layer will serialize it into whatever the title expects once that
// format has been recovered from real traces.
//
// Conventions (ASSUMED from the public Kinect for Windows v1 skeleton space,
// still to be verified against what titles expect):
//   - Meters, sensor at the origin, +z away from the sensor, +y up.
//   - +x points to the sensor's left, i.e. toward the USER'S RIGHT when the
//     user faces the sensor. User-right joints therefore have larger x.

constexpr size_t kMaxSkeletons = 6;

enum class JointId : uint8_t {
  kHipCenter = 0,
  kSpine,
  kShoulderCenter,
  kHead,
  kShoulderLeft,
  kElbowLeft,
  kWristLeft,
  kHandLeft,
  kShoulderRight,
  kElbowRight,
  kWristRight,
  kHandRight,
  kHipLeft,
  kKneeLeft,
  kAnkleLeft,
  kFootLeft,
  kHipRight,
  kKneeRight,
  kAnkleRight,
  kFootRight,
  kCount,
};
constexpr size_t kJointCount = static_cast<size_t>(JointId::kCount);

enum class TrackingState : uint8_t {
  kNotTracked = 0,
  kPositionOnly = 1,
  kTracked = 2,
};

struct Vec4 {
  float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;
};

struct Joint {
  Vec4 position;
  TrackingState state = TrackingState::kNotTracked;
};

struct Skeleton {
  TrackingState state = TrackingState::kNotTracked;
  uint32_t tracking_id = 0;
  Vec4 position;  // Overall position (hip center).
  std::array<Joint, kJointCount> joints;
};

struct SkeletonFrame {
  uint64_t frame_number = 0;  // Starts at 1.
  uint64_t timestamp_us = 0;
  Vec4 floor_clip_plane;  // Plane (a,b,c,d): ax + by + cz + d = 0.
  std::array<Skeleton, kMaxSkeletons> skeletons;
};

// Anything that can produce skeleton frames: a fake generator, the Windows
// Kinect SDK, libfreenect + a body tracker, a webcam pose estimator, a replay
// file. Implementations must be safe to call from the emulator's NUI thread.
class SkeletonSource {
 public:
  virtual ~SkeletonSource() = default;
  virtual bool Start() = 0;
  virtual void Stop() = 0;
  // Returns true and fills |out| only if a frame newer than the one returned
  // by the previous successful call is available.
  virtual bool TryGetLatestFrame(SkeletonFrame* out) = 0;
};

}  // namespace kinect
}  // namespace hid
}  // namespace xe

#endif  // XENIA_HID_KINECT_SKELETON_SOURCE_H_
