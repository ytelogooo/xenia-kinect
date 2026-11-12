/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_HID_KINECT_USB_CAMERA_H_
#define XENIA_HID_KINECT_USB_CAMERA_H_

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace xe {
namespace hid {
namespace kinect {

// Calibration stored in the camera.
struct UsbCameraCalibration {
  // Registration record: 29 little-endian 32-bit words.
  std::array<uint8_t, 116> registration = {};
  float dcmos_emitter_distance = 0.0f;
  float dcmos_rcmos_distance = 0.0f;
  float reference_distance = 0.0f;
  float reference_pixel_size = 0.0f;
  uint16_t const_shift = 0;
};

// A Kinect for Xbox 360 camera streaming 640x480 11-bit packed depth and
// 640x480 Bayer color at 30 fps.
class UsbCamera {
 public:
  static constexpr size_t kDepthFrameBytes = 640 * 480 * 11 / 8;
  static constexpr size_t kColorFrameBytes = 640 * 480;

  virtual ~UsbCamera() = default;

  // Returns nullptr when no camera is found or it cannot be opened.
  // `on_frame` is called from a camera thread whenever a new frame is ready.
  static std::unique_ptr<UsbCamera> Open(bool use_usbdk,
                                         std::function<void()> on_frame);

  // Plays a recording (kinect_record_path) instead of a live camera, looping
  // it at 30 fps, so a title's tracking can be tried without anyone in front
  // of a sensor. Returns nullptr when the directory holds no recording.
  // From Kixenia (Davi Carlos).
  static std::unique_ptr<UsbCamera> OpenRecording(
      const std::filesystem::path& directory, std::function<void()> on_frame);

  virtual const UsbCameraCalibration& calibration() const = 0;

  // Copies the newest complete frame if it is newer than `after`. Returns the
  // frame's sequence number, or 0 when there is no newer frame.
  virtual uint32_t CopyDepthFrame(uint32_t after,
                                  std::vector<uint8_t>& packed_depth) = 0;
  virtual uint32_t CopyColorFrame(uint32_t after,
                                  std::vector<uint8_t>& bayer) = 0;
};

// A recording directory holds calibration.bin and, for each frame N,
// depth_NNNNN.bin (packed 11-bit depth) and color_NNNNN.bin (Bayer color).
bool WriteRecordingCalibration(const std::filesystem::path& directory,
                               const UsbCameraCalibration& calibration);
bool WriteRecordingFrame(const std::filesystem::path& directory,
                         uint32_t index,
                         const std::vector<uint8_t>& packed_depth,
                         const std::vector<uint8_t>& bayer);

// The Kinect's motor device: the tilt motor (which tilts the sensor to an
// angle relative to gravity, measured with its accelerometer), the LED and
// the accelerometer. Protocol from libfreenect (src/tilt.c).
struct UsbMotorState {
  // Accelerometer, about 819 counts per g.
  int16_t accel[3] = {};
  // The tilt angle in degrees, if the motor knows it.
  bool angle_known = false;
  float angle = 0.0f;
  // 0 stopped, 1 at its limit, 4 moving.
  uint8_t status = 0;
};

class UsbMotor {
 public:
  enum class Led : uint16_t {
    kOff = 0,
    kGreen = 1,
    kRed = 2,
    kYellow = 3,
    // What the Kinect shows when nothing drives it.
    kBlinkGreen = 4,
    kBlinkRedYellow = 6,
  };
  static constexpr int kMaxTiltDegrees = 27;

  virtual ~UsbMotor() = default;

  // Returns nullptr when the motor can't be opened.
  static std::unique_ptr<UsbMotor> Open(bool use_usbdk);

  // Tilts to an angle relative to gravity, -27..27 degrees.
  virtual bool SetTilt(int degrees) = 0;
  virtual bool SetLed(Led led) = 0;
  virtual bool GetState(UsbMotorState* state) = 0;
};

}  // namespace kinect
}  // namespace hid
}  // namespace xe

#endif  // XENIA_HID_KINECT_USB_CAMERA_H_
