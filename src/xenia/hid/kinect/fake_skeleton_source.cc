/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/hid/kinect/fake_skeleton_source.h"

#include <chrono>
#include <cmath>
#include <utility>

namespace xe {
namespace hid {
namespace kinect {

namespace {

constexpr float kPi = 3.14159265358979f;
constexpr float kHipHeightM = 0.95f;  // Hip center height above the floor.
constexpr float kReachM = 0.40f;      // How far a raised hand extends forward.

struct Offset {
  float x, y, z;
};

// Neutral standing pose relative to the hip center, user-relative x
// (+x = user's right). Left/right are mirrored below.
constexpr Offset kCenterPose[] = {
    {0.00f, 0.00f, 0.0f},   // HipCenter
    {0.00f, 0.22f, 0.0f},   // Spine
    {0.00f, 0.52f, 0.0f},   // ShoulderCenter
    {0.00f, 0.70f, 0.0f},   // Head
};
constexpr Offset kSidePose[] = {
    {0.20f, 0.48f, 0.0f},    // Shoulder
    {0.26f, 0.22f, 0.0f},    // Elbow
    {0.28f, -0.02f, 0.0f},   // Wrist
    {0.29f, -0.10f, 0.0f},   // Hand
    {0.10f, -0.05f, 0.0f},   // Hip
    {0.11f, -0.48f, 0.0f},   // Knee
    {0.11f, -0.88f, 0.0f},   // Ankle
    {0.11f, -0.95f, 0.0f},   // Foot (floor level)
};

void Set(Skeleton* s, JointId id, float x, float y, float z) {
  Joint& j = s->joints[static_cast<size_t>(id)];
  j.position = {x, y, z, 1.0f};
  j.state = TrackingState::kTracked;
}

}  // namespace

FakeSkeletonSource::FakeSkeletonSource(FakeSkeletonConfig config,
                                       ClockFn clock_us)
    : config_(config), clock_us_(std::move(clock_us)) {
  if (!clock_us_) {
    clock_us_ = [] {
      return static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count());
    };
  }
}

bool FakeSkeletonSource::Start() {
  start_us_ = clock_us_();
  last_frame_number_ = 0;
  running_ = true;
  return true;
}

void FakeSkeletonSource::Stop() { running_ = false; }

bool FakeSkeletonSource::TryGetLatestFrame(SkeletonFrame* out) {
  if (!running_ || !out) {
    return false;
  }
  const uint64_t elapsed_us = clock_us_() - start_us_;
  const uint64_t frame_number = static_cast<uint64_t>(
      static_cast<double>(elapsed_us) * config_.frame_rate_hz / 1e6) + 1;
  if (frame_number <= last_frame_number_) {
    return false;
  }
  last_frame_number_ = frame_number;
  const uint64_t timestamp_us = static_cast<uint64_t>(
      static_cast<double>(frame_number - 1) * 1e6 / config_.frame_rate_hz);
  *out = GenerateFrame(frame_number, timestamp_us);
  return true;
}

void FakeSkeletonSource::SetRightHandOffset(float x, float y) {
  manual_x_ = x;
  manual_y_ = y;
  manual_raised_ = true;
}

void FakeSkeletonSource::ClearRightHand() { manual_raised_ = false; }

SkeletonFrame FakeSkeletonSource::GenerateFrame(uint64_t frame_number,
                                                uint64_t timestamp_us) const {
  SkeletonFrame frame;
  frame.frame_number = frame_number;
  frame.timestamp_us = timestamp_us;
  // Floor plane y + sensor_height = 0 in sensor space.
  frame.floor_clip_plane = {0.0f, 1.0f, 0.0f, config_.sensor_height_m};

  const float hip_x = config_.lateral_m;
  const float hip_y = config_.sensor_height_m * -1.0f + kHipHeightM;
  const float hip_z = config_.distance_m;

  Skeleton& s = frame.skeletons[0];
  s.state = TrackingState::kTracked;
  s.tracking_id = 1;
  s.position = {hip_x, hip_y, hip_z, 1.0f};

  // Centre joints.
  const JointId center_ids[] = {JointId::kHipCenter, JointId::kSpine,
                                JointId::kShoulderCenter, JointId::kHead};
  for (size_t i = 0; i < 4; ++i) {
    Set(&s, center_ids[i], hip_x + kCenterPose[i].x, hip_y + kCenterPose[i].y,
        hip_z + kCenterPose[i].z);
  }

  // Side joints. Sensor +x is the user's right, so user-left is mirrored.
  struct Side {
    float sign;
    JointId ids[8];
  };
  const Side sides[] = {
      {-1.0f,
       {JointId::kShoulderLeft, JointId::kElbowLeft, JointId::kWristLeft,
        JointId::kHandLeft, JointId::kHipLeft, JointId::kKneeLeft,
        JointId::kAnkleLeft, JointId::kFootLeft}},
      {+1.0f,
       {JointId::kShoulderRight, JointId::kElbowRight, JointId::kWristRight,
        JointId::kHandRight, JointId::kHipRight, JointId::kKneeRight,
        JointId::kAnkleRight, JointId::kFootRight}},
  };
  for (const Side& side : sides) {
    for (size_t i = 0; i < 8; ++i) {
      Set(&s, side.ids[i], hip_x + side.sign * kSidePose[i].x,
          hip_y + kSidePose[i].y, hip_z + kSidePose[i].z);
    }
  }

  // Right hand: raise toward the sensor, offset from the shoulder.
  float hx = manual_x_;
  float hy = manual_y_;
  bool raised = manual_raised_;
  if (config_.hand_motion != FakeHandMotion::kStill) {
    const float t = static_cast<float>(timestamp_us) / 1e6f;
    const float phase = 2.0f * kPi * t / config_.motion_period_s;
    if (config_.hand_motion == FakeHandMotion::kCircle) {
      hx += config_.motion_radius_m * std::cos(phase);
      hy += config_.motion_radius_m * std::sin(phase);
    } else {
      hx += config_.motion_radius_m * std::sin(phase);
    }
    raised = true;
  }
  if (raised) {
    const Vec4 shoulder =
        s.joints[static_cast<size_t>(JointId::kShoulderRight)].position;
    const float dx = hx, dy = hy, dz = -kReachM;  // Toward the sensor.
    Set(&s, JointId::kHandRight, shoulder.x + dx, shoulder.y + dy,
        shoulder.z + dz);
    Set(&s, JointId::kWristRight, shoulder.x + 0.92f * dx,
        shoulder.y + 0.92f * dy, shoulder.z + 0.92f * dz);
    Set(&s, JointId::kElbowRight, shoulder.x + 0.5f * dx,
        shoulder.y + 0.5f * dy - 0.08f, shoulder.z + 0.5f * dz);
  }
  return frame;
}

}  // namespace kinect
}  // namespace hid
}  // namespace xe
