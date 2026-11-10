/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XBOXKRNL_KINECT_TUNER_H_
#define XENIA_KERNEL_XBOXKRNL_KINECT_TUNER_H_

#include <cstdint>

namespace xe {
namespace kernel {
namespace xboxkrnl {

// What the Kinect Tuner (XamShowNuiTroubleshooterUI) shows, from
// xboxkrnl_usbcam.cc.
struct KinectTunerView {
  static constexpr uint32_t kWidth = 80;
  static constexpr uint32_t kHeight = 60;
  // Data from the Kinect SDK bridge within the last second.
  bool bridge = false;
  // Real-Kinect mode (allow_kinect_camera): the title tracks by itself.
  bool real_kinect = false;
  uint32_t people = 0;
  uint32_t full_skeletons = 0;
  // The depth image, 80x60: brightness by distance (0 = no reading) and the
  // player number of each pixel (0 = nobody).
  bool have_image = false;
  uint8_t grey[kWidth * kHeight] = {};
  uint8_t player[kWidth * kHeight] = {};
  // The tilt motor.
  bool have_tilt = false;
  int32_t tilt_degrees = 0;
  bool tilt_moving = false;
};

// False when there is nothing to show (no bridge, no real Kinect).
bool KinectGetTunerView(KinectTunerView* view);
// Tilts the Kinect (-27..27 degrees, relative to gravity): the real
// Kinect's motor in real-Kinect mode, else through the SDK bridge.
bool KinectBridgeRequestTilt(int32_t angle_deg);

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XBOXKRNL_KINECT_TUNER_H_
