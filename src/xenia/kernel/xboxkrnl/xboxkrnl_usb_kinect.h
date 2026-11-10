/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XBOXKRNL_XBOXKRNL_USB_KINECT_H_
#define XENIA_KERNEL_XBOXKRNL_XBOXKRNL_USB_KINECT_H_

#include <cstdint>

namespace xe {
namespace kernel {
namespace xboxkrnl {
namespace usb_kinect {

// Answers a PsCamDeviceRequest with a real Kinect v1 on the host, when
// allow_kinect_camera is set and a Kinect can be opened (tried once, on the
// first request). Returns false when there is no real camera, so the caller
// falls back to the emulated sensor.
bool TryHandleRequest(uint32_t request_ptr, uint32_t* status);

// True once a real Kinect is streaming to the title.
bool IsActive();

// The real Kinect's tilt motor while it is active (false otherwise): a
// move to an angle relative to gravity (-27..27 degrees, rate limited), and
// its angle, whether it's moving, and the accelerometer (about 819 counts
// per g; may be null).
bool MotorSetAngle(int32_t degrees);
bool MotorStatus(int32_t* degrees, bool* moving, int16_t accel[3]);
// The floor's distance below the real Kinect measured from its depth and
// accelerometer, as the console measures it once the sensor has settled
// (false if not measured).
bool MeasuredFloorHeight(float* millimetres);

// Frame timestamp in milliseconds for NUI frames built from the newest real
// depth frame (never 0), or 0 before the first depth frame.
uint64_t TimestampMs();

// How titles' NUI frames get timestamps while a real Kinect is active
// (kinect_usb_timestamps): fill in only zero timestamps, always, or never.
enum class TimestampMode { kAuto, kAlways, kNever };
TimestampMode GetTimestampMode();

void Shutdown();

}  // namespace usb_kinect
}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XBOXKRNL_XBOXKRNL_USB_KINECT_H_
