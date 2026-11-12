/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// PsCamDeviceRequest answered with a real Kinect v1 camera on the host (see
// hid/kinect/usb_camera.h). Ported from knuckleslee's camera device
// (xenia-canary PR #1274, as found in JacoMur05/kinect_DanceCentral), minus
// its title-specific diagnostics. Calibration math is from libfreenect
// (src/registration.c, Apache-2.0).
//
// Only used with allow_kinect_camera and a Kinect that opens; otherwise
// PsCamDeviceRequest keeps using the emulated sensor (xboxkrnl_usbcam.cc).

#include "xenia/kernel/xboxkrnl/xboxkrnl_usb_kinect.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/base/mutex.h"
#include "xenia/base/threading.h"
#include "xenia/cpu/processor.h"
#include "xenia/hid/kinect/usb_camera.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xthread.h"
#include "xenia/xbox.h"

DEFINE_bool(allow_kinect_camera, false,
            "Let titles use a connected Kinect for Xbox 360 (v1) camera. "
            "Without one (or if it can't be opened) the emulated sensor is "
            "used as before.",
            "HID");

DEFINE_path(kinect_record_path, "",
            "Real Kinect: record the depth and color frames the title gets "
            "into this folder (about 22 MB a second), to play back later "
            "with kinect_replay_path. From Kixenia (Davi Carlos).",
            "HID");
DEFINE_path(kinect_replay_path, "",
            "With allow_kinect_camera: play a recording made with "
            "kinect_record_path (looped, 30 frames a second) instead of a "
            "real Kinect, so a title can be tried without standing in front "
            "of one. No motor or microphones. From Kixenia (Davi Carlos).",
            "HID");
DEFINE_bool(kinect_use_usbdk, true,
            "Access a Kinect v1 camera through UsbDk, which borrows it from "
            "its Windows driver while the emulator runs. When disabled, the "
            "camera must use the WinUSB driver.",
            "HID");

DEFINE_bool(kinect_usb_motor, true,
            "Real Kinect: also take over its tilt motor, LED and accelerometer "
            "(titles' auto-tilt, the Kinect Tuner). Opened before the camera: "
            "borrowing it while the camera streams stops the camera.",
            "HID");
DEFINE_bool(kinect_colour_gains, true,
            "Real Kinect: apply the red, green and blue gains a title asks "
            "for to the color frames. The Kinect already sets its own "
            "exposure and white balance, so a title's gains come on top of "
            "them; turn off if faces look too bright or too saturated.",
            "HID");
DEFINE_double(kinect_colour_gain_max, 8.0,
              "Real Kinect: the most a title's color gains may brighten the "
              "picture (1 to 8). A console also lengthens the exposure, "
              "which Xenia can't, so high gains only brighten noise; lower "
              "this if faces look blown out, raise it if they look dark.",
              "HID");

DEFINE_bool(kinect_usb_keep_no_depth, true,
            "Real Kinect: hand titles pixels without depth as the sensor sends "
            "them (2047, all 11 bits set) rather than as 0.",
            "HID");

DEFINE_string(kinect_usb_timestamps, "auto",
              "Timestamps of NUI skeleton/image frames with a real Kinect: "
              "\"auto\" fills in only timestamps the runtime leaves 0, "
              "\"always\" replaces them, \"never\" leaves them alone.",
              "HID");

namespace xe {
namespace kernel {
namespace xboxkrnl {
namespace usb_kinect {

namespace {

constexpr X_STATUS kStatusCancelled = 0xC0000120;
constexpr X_STATUS kStatusDeviceNotConnected = 0xC000009D;

enum class CameraRequestCode : uint32_t {
  kGetState = 0,
  kStop = 2,
  kReadFrame = 5,
  kVersion = 13,
  kRegistration = 14,
  kTimeSync = 15,
  kDepthTable = 24,
  kTiming = 25,
  kConfigure = 27,
  kUnknown33 = 33,
  kSetControls = 35,
  kGetControls = 36,
};

enum class CameraStream : uint32_t {
  kColor = 2,
  kDepth = 4,
};

// Device states reported through code 0.
constexpr uint32_t kDeviceStateAbsent = 0;
constexpr uint32_t kDeviceStateStreaming = 2;
constexpr uint32_t kDeviceStateReady = 4;
constexpr uint32_t kNoCompletionRoutine = 0xFFFFFFFF;
constexpr uint32_t kFrameWidth = 640;
constexpr uint32_t kFrameHeight = 480;
constexpr uint32_t kDepthFrameBytes = kFrameWidth * kFrameHeight * 2;
constexpr uint32_t kColorFrameBytes = kFrameWidth * kFrameHeight;
constexpr uint32_t kDepthTableEntries = 8192;
constexpr uint16_t kNoDepthRaw = 2047;
// Frame period in microseconds, returned for both streams by code 25.
constexpr uint32_t kFramePeriodUs = 33333;
// Host-side choices, not console values.
constexpr uint32_t kWorkerStackSize = 128 * 1024;
constexpr auto kWorkerWaitTimeout = std::chrono::milliseconds(100);
// Reported through code 13; 5.1 or later selects the registration request.
constexpr uint8_t kDriverVersionMajor = 5;
constexpr uint8_t kDriverVersionMinor = 1;

// libfreenect registration constants (src/registration.c).
constexpr uint16_t kShiftToDepthPixelConst = 10;
constexpr double kShiftToDepthConstOffset = 0.375;
constexpr double kParameterCoefficient = 4.0;
constexpr double kShiftScale = 10.0;
constexpr double kMaxDepthMillimetres = 10000.0;

// Raw 11-bit depth to millimetres (libfreenect freenect_raw_to_mm), 0 for raw
// values without depth or out of range.
uint16_t RawDepthToMillimetres(uint32_t raw,
                               const hid::kinect::UsbCameraCalibration& c) {
  // Raw 0 is used for pixels without depth, see UnpackDepth.
  if (raw == 0 || raw >= kNoDepthRaw) {
    return 0;
  }
  const double fixed_ref_x =
      (raw - kParameterCoefficient * c.const_shift) / kParameterCoefficient -
      kShiftToDepthConstOffset;
  const double metric = fixed_ref_x * c.reference_pixel_size;
  const double value =
      kShiftScale * (metric * c.reference_distance /
                         (c.dcmos_emitter_distance - metric) +
                     c.reference_distance);
  return value > 0 && value < kMaxDepthMillimetres
             ? static_cast<uint16_t>(value)
             : 0;
}

// Request header. Codes 14 and 25 reuse +0x08 onwards for their replies.
struct X_PSCAM_REQUEST {
  xe::be<uint32_t> code;
  xe::be<uint32_t> unknown_04[3];
  xe::be<uint32_t> context;
  // Called with (request, status, bytes), 0 or kNoCompletionRoutine for none.
  xe::be<uint32_t> completion_routine;
};
static_assert_size(X_PSCAM_REQUEST, 0x18);

struct X_PSCAM_STATE_REQUEST {
  xe::be<uint32_t> code;
  xe::be<uint32_t> state;
};
static_assert_size(X_PSCAM_STATE_REQUEST, 0x8);

struct X_PSCAM_READ_FRAME_REQUEST {
  X_PSCAM_REQUEST header;
  xe::be<uint32_t> stream;
  xe::be<uint32_t> buffer_ptr;
  xe::be<uint32_t> buffer_size;
  xe::be<uint32_t> unknown_24;
  // Normally the request's own +0x40.
  xe::be<uint32_t> frame_counter_ptr;
  // Color frames only.
  uint8_t settings_generation;
  uint8_t unknown_2D[19];
  // The frame's sequence number and timestamp (guest uptime in ms), which
  // the runtime makes the skeleton frame's dwFrameNumber and liTimeStamp: a
  // timestamp left at 0 leaves everything it times from frame to frame (a
  // hand's speed) at nothing. Found by dcwestra/xenia-edge's Kinect work.
  xe::be<uint32_t> sequence;  // 0x40
  xe::be<uint32_t> unknown_44;
  xe::be<uint64_t> timestamp_ms;  // 0x48
};
static_assert_size(X_PSCAM_READ_FRAME_REQUEST, 0x50);

// Camera settings (code 35, read back by 36). From +0x1C: flags (bit 0 the
// exposure time valid, bits 1-3 the red, green and blue gains), the
// resolution, at +0x24 1 for a window of the sensor's full resolution
// rather than all of it, centred +0x28 / +0x2C full-resolution pixels from
// the sensor's centre (the identity runtime pans it onto a face), then the
// exposure: gains at +0x4C..+0x54 and the time at +0x58, 16.16 fixed point.
// (Layout from dcwestra/xenia-edge.)
struct X_PSCAM_CONTROLS_REQUEST {
  X_PSCAM_REQUEST header;
  xe::be<uint32_t> settings_generation;  // 0x18, set on completion
  xe::be<uint32_t> flags;                // 0x1C
  xe::be<uint32_t> resolution;           // 0x20
  xe::be<uint32_t> window;               // 0x24
  xe::be<int32_t> window_x;              // 0x28
  xe::be<int32_t> window_y;              // 0x2C
  xe::be<uint32_t> unknown_30[7];
  xe::be<uint32_t> gains[3];       // 0x4C
  xe::be<uint32_t> exposure_time;  // 0x58
};
static_assert_size(X_PSCAM_CONTROLS_REQUEST, 0x5C);

// What a title set of the color stream.
struct ColorSettings {
  bool window = false;
  int32_t window_x = 0;
  int32_t window_y = 0;
  // Applied to the frames (1 when kinect_colour_gains is off).
  float gains[3] = {1.0f, 1.0f, 1.0f};
  // As the title asked for them.
  float requested_gains[3] = {1.0f, 1.0f, 1.0f};
};

// The sensor's 8-bit Bayer (GRBG) color frame as the title asked for it:
// the quarter of it around the window's centre at twice the size, and with
// its gains, still Bayer. Adapted from dcwestra/xenia-edge's
// ViewColorFrame, without its color-to-depth registration (the title gets
// the camera's own registration through code 14 here).
void ViewColorFrame(const uint8_t* in, const ColorSettings& settings,
                    uint8_t* out) {
  constexpr int kWidth = 640;
  constexpr int kHeight = 480;
  struct Tap {
    int32_t i0, i1;
    float weight;
  };
  auto taps = [](int size, bool window, int32_t window_offset,
                 std::vector<Tap>& phase0, std::vector<Tap>& phase1) {
    phase0.resize(size);
    phase1.resize(size);
    for (int i = 0; i < size; ++i) {
      const float source = window ? size * 0.25f + window_offset * 0.5f +
                                        (i + 0.5f) * 0.5f - 0.5f
                                  : float(i);
      for (int phase = 0; phase < 2; ++phase) {
        // A color's pixels are every other one, in one of two phases.
        const int count = size / 2;
        const float f = (source - phase) * 0.5f;
        int k0 = int(std::floor(f));
        float weight = f - k0;
        if (k0 < 0) {
          k0 = 0;
          weight = 0.0f;
        } else if (k0 >= count - 1) {
          k0 = count - 1;
          weight = 0.0f;
        }
        const int k1 = std::min(k0 + 1, count - 1);
        (phase ? phase1 : phase0)[i] = {phase + 2 * k0, phase + 2 * k1, weight};
      }
    }
  };
  std::vector<Tap> columns[2], rows[2];
  taps(kWidth, settings.window, settings.window_x, columns[0], columns[1]);
  taps(kHeight, settings.window, settings.window_y, rows[0], rows[1]);
  auto sample = [&](int column_phase, int row_phase, int x, int y) {
    const Tap& column = columns[column_phase][x];
    const Tap& row = rows[row_phase][y];
    const uint8_t* row0 = in + row.i0 * kWidth;
    const uint8_t* row1 = in + row.i1 * kWidth;
    const float top =
        row0[column.i0] + (row0[column.i1] - row0[column.i0]) * column.weight;
    const float bottom =
        row1[column.i0] + (row1[column.i1] - row1[column.i0]) * column.weight;
    return top + (bottom - top) * row.weight;
  };
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      float value;
      if ((x ^ y) & 1) {
        // Red on even rows, blue on odd ones.
        value = y & 1 ? sample(0, 1, x, y) * settings.gains[2]
                      : sample(1, 0, x, y) * settings.gains[0];
      } else {
        value = (sample(0, 0, x, y) + sample(1, 1, x, y)) * 0.5f *
                settings.gains[1];
      }
      out[y * kWidth + x] = uint8_t(std::min(value + 0.5f, 255.0f));
    }
  }
}

struct X_PSCAM_VERSION_REQUEST {
  xe::be<uint32_t> code;
  uint8_t major;
  uint8_t minor;
  uint8_t unknown_06[2];
};
static_assert_size(X_PSCAM_VERSION_REQUEST, 0x8);

struct X_PSCAM_REGISTRATION_REQUEST {
  xe::be<uint32_t> code;
  xe::be<uint32_t> unknown_04;
  xe::be<uint32_t> registration[29];
  uint8_t unknown_7C[0x100C];
  // Depth-to-color shift parameters, used like libfreenect's
  // freenect_init_depth_to_rgb.
  xe::be<double> reference_distance;
  xe::be<double> reference_pixel_size;
  // The title divides by this; libfreenect's equivalent factor is 1.
  xe::be<uint16_t> frame_width;
  xe::be<uint16_t> shift_to_depth_pixel_const;
  uint8_t unknown_109C[4];
  xe::be<double> shift_to_depth_const_offset;
  xe::be<double> dcmos_rcmos_distance;
};
static_assert_size(X_PSCAM_REGISTRATION_REQUEST, 0x10B0);

struct X_PSCAM_DEPTH_TABLE_REQUEST {
  xe::be<uint32_t> code;
  xe::be<uint32_t> unknown_04;
  xe::be<uint32_t> table_ptr;
  xe::be<uint32_t> entry_count;
};
static_assert_size(X_PSCAM_DEPTH_TABLE_REQUEST, 0x10);

struct X_PSCAM_TIMING_REQUEST {
  xe::be<uint32_t> code;
  // Microseconds.
  xe::be<uint32_t> depth_frame_period;
  xe::be<uint32_t> unknown_08[4];
  // Microseconds.
  xe::be<uint32_t> color_frame_period;
  xe::be<uint32_t> unknown_1C[2];
  // The frame period in microseconds: the only value runtimes test. 0
  // skips the end of the camera's configuration and the runtime's
  // per-frame graphics and exposure work (dcwestra/xenia-edge).
  xe::be<uint32_t> frame_period;
  xe::be<uint32_t> unknown_28[5];
};
static_assert_size(X_PSCAM_TIMING_REQUEST, 0x3C);

struct PendingRequest {
  uint32_t request_ptr = 0;
  // Read at request time, the title may overwrite the field afterwards. 0 for
  // none.
  uint32_t completion = 0;
  X_STATUS status = X_STATUS_SUCCESS;
  uint32_t bytes = 0;
  // Camera settings request that sets this generation once completed.
  bool new_settings = false;
  uint8_t generation = 0;
  // Frame reads that were queued when streaming was stopped (code 2),
  // completed as cancelled before this request.
  std::vector<PendingRequest> cancelled_reads;
};

// Kinect Adventures (NUI 2.0.11775) initializes its NUI runtime for the
// console's "high quality" color: 640x480 at two bytes a pixel (its color
// frame buffers are 0x96000 bytes, against 0x4B000 for the 8-bit Bayer later
// titles take), and drops every color frame of another size as
// STATUS_INVALID_BUFFER_SIZE - so its Kinect ID never saw a face. The
// camera's own 16-bit format is YUV 4:2:2 as U Y0 V Y1 (UYVY); the 8-bit
// Bayer frame (GRBG, already windowed) is demosaiced and converted to it.
void BayerToUyvy(const uint8_t* bayer, uint8_t* out) {
  constexpr int kWidth = 640;
  constexpr int kHeight = 480;
  auto at = [bayer](int x, int y) {
    x = std::clamp(x, 0, kWidth - 1);
    y = std::clamp(y, 0, kHeight - 1);
    return int(bayer[y * kWidth + x]);
  };
  // GRBG: green where (x ^ y) is even, red on even rows, blue on odd ones.
  auto rgb = [&](int x, int y, int& r, int& g, int& b) {
    const int c = at(x, y);
    const int cross =
        (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) + 2) / 4;
    const int diagonal = (at(x - 1, y - 1) + at(x + 1, y - 1) +
                          at(x - 1, y + 1) + at(x + 1, y + 1) + 2) /
                         4;
    const int horizontal = (at(x - 1, y) + at(x + 1, y) + 1) / 2;
    const int vertical = (at(x, y - 1) + at(x, y + 1) + 1) / 2;
    if (!((x ^ y) & 1)) {
      g = c;
      if (y & 1) {  // green on a blue row: red above and below
        r = vertical;
        b = horizontal;
      } else {  // green on a red row
        r = horizontal;
        b = vertical;
      }
    } else if (!(y & 1)) {
      r = c;
      g = cross;
      b = diagonal;
    } else {
      b = c;
      g = cross;
      r = diagonal;
    }
  };
  auto clamp8 = [](float v) {
    return uint8_t(std::clamp(int(std::lround(v)), 0, 255));
  };
  for (int y = 0; y < kHeight; ++y) {
    uint8_t* row = out + y * kWidth * 2;
    for (int x = 0; x < kWidth; x += 2) {
      int r0, g0, b0, r1, g1, b1;
      rgb(x, y, r0, g0, b0);
      rgb(x + 1, y, r1, g1, b1);
      const float y0 = 0.299f * r0 + 0.587f * g0 + 0.114f * b0;
      const float y1 = 0.299f * r1 + 0.587f * g1 + 0.114f * b1;
      const float r = (r0 + r1) * 0.5f, g = (g0 + g1) * 0.5f,
                  b = (b0 + b1) * 0.5f;
      const float luma = 0.299f * r + 0.587f * g + 0.114f * b;
      row[x * 2 + 0] = clamp8(128.0f + (b - luma) * 0.564f);
      row[x * 2 + 1] = clamp8(y0);
      row[x * 2 + 2] = clamp8(128.0f + (r - luma) * 0.713f);
      row[x * 2 + 3] = clamp8(y1);
    }
  }
}

int64_t SteadyMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

class CameraDevice {
 public:
  explicit CameraDevice(KernelState* kernel_state)
      : kernel_state_(kernel_state), memory_(kernel_state->memory()) {
    if (!cvars::kinect_replay_path.empty()) {
      camera_ = hid::kinect::UsbCamera::OpenRecording(
          cvars::kinect_replay_path, [this]() { Notify(); });
      if (!camera_) {
        XELOGE("Kinect: no recording to play in {}",
               xe::path_to_utf8(cvars::kinect_replay_path));
        return;
      }
      StartWorker(kernel_state);
      return;
    }
    // The motor (tilt, LED, accelerometer) is a USB device of its own. It's
    // opened first: UsbDk taking it from the Kinect's Windows driver while the
    // camera streamed killed every camera transfer (and the IR projector).
    if (cvars::kinect_usb_motor) {
      motor_ = hid::kinect::UsbMotor::Open(cvars::kinect_use_usbdk);
      if (!motor_) {
        XELOGW("Kinect: the motor could not be opened (no tilt or LED)");
      } else {
        // Lets the Kinect's driver settle before the camera is taken too.
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
    }
    // The camera may be re-enumerating after that; retry for a few seconds.
    for (int attempt = 0; attempt < 6; ++attempt) {
      if (attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        XELOGI("Kinect: opening the camera again (attempt {})", attempt + 1);
      }
      camera_ = hid::kinect::UsbCamera::Open(cvars::kinect_use_usbdk,
                                             [this]() { Notify(); });
      if (camera_ || !motor_) {
        break;
      }
    }
    if (!camera_) {
      motor_.reset();
      XELOGI(
          "Kinect: no real Kinect camera could be opened, using the emulated "
          "sensor");
      return;
    }
    const auto& calibration = camera_->calibration();
    XELOGI(
        "Kinect: real camera calibration: const shift {}, reference distance "
        "{:.2f}, pixel size {:.4f}, emitter distance {:.2f}",
        calibration.const_shift, calibration.reference_distance,
        calibration.reference_pixel_size, calibration.dcmos_emitter_distance);
    // Whether the motor survived the camera being taken over too.
    if (motor_) {
      hid::kinect::UsbMotorState state;
      if (motor_->GetState(&state)) {
        motor_->SetLed(hid::kinect::UsbMotor::Led::kGreen);
        XELOGI("Kinect: motor still answers with the camera open (tilt {})",
               state.angle_known ? fmt::format("{} degrees", state.angle)
                                 : std::string("unknown"));
      } else {
        XELOGW(
            "Kinect: the motor stopped answering once the camera was opened; "
            "no tilt or LED (not reopened, that would stop the camera)");
        motor_.reset();
      }
    }
    if (!cvars::kinect_record_path.empty()) {
      if (hid::kinect::WriteRecordingCalibration(cvars::kinect_record_path,
                                                 camera_->calibration())) {
        recording_ = true;
        XELOGI("Kinect: recording the frames into {}",
               xe::path_to_utf8(cvars::kinect_record_path));
      } else {
        XELOGE("Kinect: can't record into {}",
               xe::path_to_utf8(cvars::kinect_record_path));
      }
    }
    StartWorker(kernel_state);
  }

  void StartWorker(KernelState* kernel_state) {
    worker_thread_ = object_ref<XHostThread>(new XHostThread(
        kernel_state, kWorkerStackSize, 0,
        [this]() {
          WorkerMain();
          return 0;
        },
        kernel_state->GetSystemProcess()));
    worker_thread_->set_can_debugger_suspend(true);
    worker_thread_->set_name("Kinect Camera Worker");
    worker_thread_->Create();
  }

  ~CameraDevice() {
    worker_running_ = false;
    Notify();
    if (worker_thread_) {
      worker_thread_->Wait(0, 0, 0, nullptr);
      worker_thread_.reset();
    }
    // Closing the camera stops its streams and returns it to its driver.
    camera_.reset();
    motor_.reset();
  }

  bool has_camera() const { return camera_ != nullptr; }

  // Tilt requests from the title, the Kinect Tuner or the auto-tilt: the
  // latest wins, at most one move every 1.2 s (the motor isn't made for
  // more; the Kinect SDK limits it the same way), equal angles ignored.
  bool MotorSetAngle(int32_t degrees) {
    std::lock_guard<xe_mutex> lock(motor_mutex_);
    if (!motor_) {
      return false;
    }
    degrees = std::clamp(degrees, -hid::kinect::UsbMotor::kMaxTiltDegrees,
                         hid::kinect::UsbMotor::kMaxTiltDegrees);
    motor_target_ = degrees;
    ApplyMotorTarget();
    return true;
  }

  bool MotorStatus(int32_t* degrees, bool* moving, int16_t accel[3]) {
    std::lock_guard<xe_mutex> lock(motor_mutex_);
    if (!motor_) {
      return false;
    }
    const int64_t now = SteadyMs();
    if (now - motor_polled_ms_ >= 100) {
      hid::kinect::UsbMotorState state;
      if (motor_->GetState(&state)) {
        motor_state_ = state;
        motor_have_state_ = true;
      }
      motor_polled_ms_ = now;
    }
    ApplyMotorTarget();
    if (!motor_have_state_) {
      return false;
    }
    const hid::kinect::UsbMotorState& state = motor_state_;
    float angle = state.angle;
    if (!state.angle_known) {
      // From the accelerometer: gravity along -y when level, tilting up
      // brings it towards +z.
      angle = float(std::atan2(double(state.accel[2]),
                               double(std::abs(state.accel[1]))) *
                    57.29577951308232);
    }
    *degrees = int32_t(std::lround(angle));
    *moving = state.status == 4 || motor_target_ != kNoMotorTarget ||
              now - motor_moved_ms_ < 600;
    if (accel) {
      std::memcpy(accel, state.accel, sizeof(state.accel));
    }
    return true;
  }

  // The floor's distance below the sensor in millimetres (0 if not measured),
  // measured like the console does (it measures the floor itself whenever the
  // sensor has settled after moving, without any title asking - seen with a
  // probe at the dashboard).
  float MeasuredFloorHeight() const { return measured_floor_mm_.load(); }

  uint64_t TimestampMs() const {
    const int64_t first = first_depth_ms_.load();
    if (!first) {
      return 0;
    }
    // ms since streaming started, +10 s like the emulated sensor. No
    // latency compensation: a real Kinect has the latency titles expect.
    return uint64_t(last_depth_ms_.load() - first + 10000);
  }

  X_STATUS Request(uint32_t request_ptr) {
    if (!IsGuestRange(request_ptr, sizeof(X_PSCAM_REQUEST))) {
      return X_STATUS_INVALID_PARAMETER;
    }
    auto request = Guest<X_PSCAM_REQUEST>(request_ptr);
    const auto code = static_cast<CameraRequestCode>(uint32_t(request->code));
    if (!IsGuestRange(request_ptr, GetRequestSize(code))) {
      return X_STATUS_INVALID_PARAMETER;
    }
    uint32_t completion = request->completion_routine;
    if (completion == kNoCompletionRoutine) {
      completion = 0;
    }
    LogRequest(code, request_ptr);

    if (!camera_) {
      if (code == CameraRequestCode::kGetState) {
        uint32_t bytes = 0;
        return Handle(code, request_ptr, bytes);
      }
      return kStatusDeviceNotConnected;
    }

    if (code == CameraRequestCode::kReadFrame) {
      auto read = Guest<X_PSCAM_READ_FRAME_REQUEST>(request_ptr);
      const auto stream = static_cast<CameraStream>(uint32_t(read->stream));
      const uint32_t counter_ptr = read->frame_counter_ptr;
      if (!completion ||
          (stream != CameraStream::kDepth && stream != CameraStream::kColor) ||
          (counter_ptr && !IsGuestRange(counter_ptr, sizeof(uint32_t)))) {
        if (diag_failures_++ < 20) {
          XELOGE(
              "Kinect: read-frame request rejected: completion={:08X} "
              "stream={} counter_ptr={:08X}",
              completion, static_cast<uint32_t>(stream), counter_ptr);
        }
        return X_STATUS_INVALID_PARAMETER;
      }
      PendingRequest pending;
      pending.request_ptr = request_ptr;
      pending.completion = completion;
      {
        std::lock_guard<xe_mutex> lock(mutex_);
        streaming_ = true;
        (stream == CameraStream::kDepth ? depth_reads_ : color_reads_)
            .push_back(pending);
      }
      Notify();
      return X_STATUS_PENDING;
    }

    uint32_t bytes = 0;
    const X_STATUS status = Handle(code, request_ptr, bytes);
    if (!XSUCCEEDED(status) && diag_failures_++ < 20) {
      XELOGE("Kinect: request code {} returned {:08X}",
             static_cast<uint32_t>(code), static_cast<uint32_t>(status));
    }
    PendingRequest pending;
    pending.request_ptr = request_ptr;
    pending.completion = completion;
    pending.status = status;
    pending.bytes = bytes;
    pending.new_settings =
        code == CameraRequestCode::kSetControls && XSUCCEEDED(status);
    pending.generation = requested_generation_;
    {
      std::lock_guard<xe_mutex> lock(mutex_);
      if (code == CameraRequestCode::kStop) {
        // Queued reads are completed with STATUS_CANCELLED on a stop.
        pending.cancelled_reads.assign(depth_reads_.begin(),
                                       depth_reads_.end());
        pending.cancelled_reads.insert(pending.cancelled_reads.end(),
                                       color_reads_.begin(),
                                       color_reads_.end());
        depth_reads_.clear();
        color_reads_.clear();
        streaming_ = false;
        ++stop_count_;
      }
      if (!completion && pending.cancelled_reads.empty()) {
        return status;
      }
      completions_.push_back(std::move(pending));
    }
    Notify();
    return completion ? X_STATUS_PENDING : status;
  }

 private:
  template <typename T>
  T* Guest(uint32_t address) const {
    return memory_->TranslateVirtual<T*>(address);
  }

  // Checks that the first and last byte of a guest range are mapped.
  bool IsGuestRange(uint32_t address, size_t size) const {
    if (!address || !size || size > UINT32_MAX) {
      return false;
    }
    const uint32_t last_address = address + static_cast<uint32_t>(size - 1);
    return last_address >= address && memory_->LookupHeap(address) &&
           memory_->LookupHeap(last_address);
  }

  static size_t GetRequestSize(CameraRequestCode code) {
    size_t size = 0;
    switch (code) {
      case CameraRequestCode::kGetState:
        size = sizeof(X_PSCAM_STATE_REQUEST);
        break;
      case CameraRequestCode::kReadFrame:
        size = sizeof(X_PSCAM_READ_FRAME_REQUEST);
        break;
      case CameraRequestCode::kVersion:
        size = sizeof(X_PSCAM_VERSION_REQUEST);
        break;
      case CameraRequestCode::kRegistration:
        size = sizeof(X_PSCAM_REGISTRATION_REQUEST);
        break;
      case CameraRequestCode::kDepthTable:
        size = sizeof(X_PSCAM_DEPTH_TABLE_REQUEST);
        break;
      case CameraRequestCode::kTiming:
        size = sizeof(X_PSCAM_TIMING_REQUEST);
        break;
      case CameraRequestCode::kSetControls:
      case CameraRequestCode::kGetControls:
        size = sizeof(X_PSCAM_CONTROLS_REQUEST);
        break;
      default:
        break;
    }
    return std::max(size, sizeof(X_PSCAM_REQUEST));
  }

  // Logs the first three requests of each code (and of each frame stream)
  // with their raw words.
  void LogRequest(CameraRequestCode code, uint32_t request_ptr) {
    uint32_t slot = static_cast<uint32_t>(code);
    if (code == CameraRequestCode::kReadFrame) {
      slot = 48 + (static_cast<uint32_t>(
                       Guest<X_PSCAM_READ_FRAME_REQUEST>(request_ptr)->stream) &
                   7);
    }
    if (slot >= xe::countof(request_counts_)) {
      return;
    }
    const uint32_t count = ++request_counts_[slot];
    if (count > 3) {
      return;
    }
    uint32_t size = 0x40;
    while (size > sizeof(X_PSCAM_REQUEST) && !IsGuestRange(request_ptr, size)) {
      size -= 4;
    }
    std::string words;
    for (uint32_t i = 0; i < size / 4; ++i) {
      char text[16];
      std::snprintf(text, sizeof(text), "%08X ",
                    static_cast<unsigned>(xe::load_and_swap<uint32_t>(
                        Guest<uint8_t>(request_ptr) + i * 4)));
      words += text;
    }
    XELOGI("Kinect: request code={} slot={} #{} words: {}",
           static_cast<uint32_t>(code), slot, count, words);
  }

  // Raw depth statistics of the frame just written to the title.
  void LogDepthStats(uint32_t buffer_ptr) {
    const auto* depth = Guest<xe::be<uint16_t>>(buffer_ptr);
    const uint32_t pixels = kFrameWidth * kFrameHeight;
    uint32_t valid = 0;
    uint16_t min_raw = 0xFFFF;
    uint16_t max_raw = 0;
    for (uint32_t i = 0; i < pixels; ++i) {
      const uint16_t raw = depth[i];
      if (raw) {
        ++valid;
        min_raw = std::min(min_raw, raw);
        max_raw = std::max(max_raw, raw);
      }
    }
    XELOGI(
        "Kinect: depth frame #{} valid pixels {}% raw {}..{}, color frames {}",
        depth_delivered_, valid * 100 / pixels, min_raw, max_raw,
        color_delivered_);
  }

  // Measures the floor in the depth frame just written to the title: the
  // depth pixels' distances below the sensor along gravity (from the
  // accelerometer), and the floor is the lowest surface covering a good part
  // of the picture. Taken when three measurements in a row agree.
  void MeasureFloor(uint32_t buffer_ptr) {
    int16_t accel[3];
    {
      std::lock_guard<xe_mutex> lock(motor_mutex_);
      if (!motor_ || !motor_have_state_ || motor_state_.status == 4 ||
          motor_target_ != kNoMotorTarget ||
          SteadyMs() - motor_moved_ms_ < 1500) {
        floor_candidate_count_ = 0;
        return;
      }
      std::memcpy(accel, motor_state_.accel, sizeof(accel));
    }
    if (accel[1] <= 409 || accel[1] >= 1229) {
      return;
    }
    const double up_length =
        std::sqrt(double(accel[1]) * accel[1] + double(accel[2]) * accel[2]);
    const double up_y = accel[1] / up_length, up_z = accel[2] / up_length;
    const hid::kinect::UsbCameraCalibration& calibration =
        camera_->calibration();
    if (depth_millimetres_.empty()) {
      depth_millimetres_.resize(kNoDepthRaw);
      for (uint32_t raw = 0; raw < kNoDepthRaw; ++raw) {
        depth_millimetres_[raw] = RawDepthToMillimetres(raw, calibration);
      }
    }
    if (calibration.reference_pixel_size <= 0.0) {
      return;
    }
    // libfreenect's reference pixel size is for 1280 pixels across.
    const double focal = calibration.reference_distance /
                         (2.0 * calibration.reference_pixel_size);
    // 1 cm bins of the distance below the sensor, 20 cm to 3 m.
    constexpr uint32_t kMinCm = 20, kBins = 280;
    uint32_t bins[kBins] = {};
    // The nearest and farthest depth in each bin: the floor covers a range of
    // depths, a wall or a person at one distance doesn't.
    uint16_t bin_min_z[kBins], bin_max_z[kBins] = {};
    std::fill(std::begin(bin_min_z), std::end(bin_min_z), uint16_t(0xFFFF));
    uint32_t samples = 0;
    // Samples less than 20 cm below the sensor or above it.
    uint32_t samples_higher = 0;
    const auto* depth = Guest<xe::be<uint16_t>>(buffer_ptr);
    for (uint32_t v = 0; v < kFrameHeight; v += 2) {
      const double row = (double(kFrameHeight) * 0.5 - v) / focal;
      for (uint32_t u = 0; u < kFrameWidth; u += 2) {
        const uint16_t raw = depth[v * kFrameWidth + u];
        if (raw >= kNoDepthRaw) {
          continue;
        }
        const uint16_t z = depth_millimetres_[raw];
        if (z < 500 || z > 5000) {
          continue;
        }
        // Camera space: y up, z forward; the distance below the sensor is
        // minus the point's component along up.
        const double below = -(row * z * up_y + z * up_z);
        const int32_t bin = int32_t(below * 0.1) - int32_t(kMinCm);
        if (bin < 0) {
          ++samples_higher;
        }
        if (bin >= 0 && bin < int32_t(kBins)) {
          ++bins[bin];
          bin_min_z[bin] = std::min(bin_min_z[bin], z);
          bin_max_z[bin] = std::max(bin_max_z[bin], z);
        }
        ++samples;
      }
    }
    // For tuning: the 8 fullest 5 cm bands (every 20th measurement).
    if (floor_logs_ < 6 && floor_measurements_++ % 20 == 0) {
      ++floor_logs_;
      constexpr uint32_t kBand = 5;
      uint32_t bands[kBins / kBand] = {};
      uint16_t band_min_z[kBins / kBand], band_max_z[kBins / kBand] = {};
      std::fill(std::begin(band_min_z), std::end(band_min_z),
                uint16_t(0xFFFF));
      for (uint32_t bin = 0; bin < kBins; ++bin) {
        bands[bin / kBand] += bins[bin];
        if (bins[bin]) {
          band_min_z[bin / kBand] =
              std::min(band_min_z[bin / kBand], bin_min_z[bin]);
          band_max_z[bin / kBand] =
              std::max(band_max_z[bin / kBand], bin_max_z[bin]);
        }
      }
      uint32_t order[kBins / kBand];
      for (uint32_t i = 0; i < kBins / kBand; ++i) {
        order[i] = i;
      }
      std::partial_sort(std::begin(order), std::begin(order) + 8,
                        std::end(order), [&](uint32_t a, uint32_t b) {
                          return bands[a] > bands[b];
                        });
      std::string list;
      for (uint32_t i = 0; i < 8 && bands[order[i]]; ++i) {
        const uint32_t band = order[i];
        list += fmt::format(" {}-{}cm:{}({}-{}mm)", kMinCm + band * kBand,
                            kMinCm + band * kBand + kBand, bands[band],
                            band_min_z[band], band_max_z[band]);
      }
      XELOGI(
          "Kinect: floor search, gravity {} {} {}, {} samples ({} less than "
          "20 cm below the sensor), fullest bands below the sensor (count, "
          "depths):{}",
          accel[0], accel[1], accel[2], samples, samples_higher, list);
    }
    // The lowest group of bins with at least 0.5% of the samples (and 300) -
    // from a sensor high up, the floor can be a small part of the picture
    // next to a bed or a table top (seen: 3% floor against 9% bed).
    const uint32_t threshold = std::max(samples / 200, uint32_t(300));
    int32_t top = -1;
    for (int32_t bin = kBins - 2; bin >= 1; --bin) {
      if (bins[bin - 1] + bins[bin] + bins[bin + 1] >= threshold) {
        top = bin;
        break;
      }
    }
    if (top < 0) {
      floor_candidate_count_ = 0;
      return;
    }
    int32_t peak = top;
    for (int32_t bin = std::max(top - 8, 0); bin <= top + 1; ++bin) {
      if (bins[bin] > bins[peak]) {
        peak = bin;
      }
    }
    const int32_t window_first = std::max(peak - 2, 0);
    const int32_t window_last = std::min(peak + 2, int32_t(kBins) - 1);
    double weight = 0.0, sum = 0.0;
    uint16_t window_min_z = 0xFFFF, window_max_z = 0;
    for (int32_t bin = window_first; bin <= window_last; ++bin) {
      weight += bins[bin];
      sum += bins[bin] * ((bin + kMinCm) * 10.0 + 5.0);
      if (bins[bin]) {
        window_min_z = std::min(window_min_z, bin_min_z[bin]);
        window_max_z = std::max(window_max_z, bin_max_z[bin]);
      }
    }
    // A floor is a narrow band of heights: much denser than the 10 cm just
    // above it (where walls and furniture spread evenly), and spanning depths.
    double above = 0.0;
    int32_t above_bins = 0;
    for (int32_t bin = std::max(window_first - 10, 0); bin < window_first;
         ++bin) {
      above += bins[bin];
      ++above_bins;
    }
    const double window_density = weight / (window_last - window_first + 1);
    if (weight < threshold ||
        (above_bins && window_density < 2.0 * above / above_bins) ||
        window_max_z - window_min_z < 250) {
      floor_candidate_count_ = 0;
      return;
    }
    const float candidate = float(sum / weight);
    floor_candidates_[floor_candidate_count_ % 3] = candidate;
    ++floor_candidate_count_;
    if (floor_candidate_count_ < 3) {
      return;
    }
    float sorted[3] = {floor_candidates_[0], floor_candidates_[1],
                       floor_candidates_[2]};
    std::sort(sorted, sorted + 3);
    if (sorted[2] - sorted[0] > 30.0f) {
      return;
    }
    const float measured = sorted[1];
    const float previous = measured_floor_mm_.load();
    // The floor doesn't rise: a measurement more than 5 cm above the floor
    // already found is most likely something standing on it covering the
    // floor for a moment (a bed, a table, the player), unless it stays so for
    // a while (the sensor was put elsewhere).
    if (previous > 0.0f && measured < previous - 50.0f) {
      if (++floor_higher_count_ < 20) {
        return;
      }
    }
    floor_higher_count_ = 0;
    if (std::abs(previous - measured) >= 30.0f) {
      XELOGI(
          "Kinect: measured the floor {:.0f} mm below the sensor (gravity "
          "{} {} {})",
          measured, accel[0], accel[1], accel[2]);
    }
    measured_floor_mm_.store(measured);
  }

  static constexpr int32_t kNoMotorTarget = INT32_MIN;

  // Called with motor_mutex_ held.
  void ApplyMotorTarget() {
    if (motor_target_ == kNoMotorTarget) {
      return;
    }
    const int64_t now = SteadyMs();
    if (motor_target_ == motor_last_angle_ ||
        (motor_have_state_ && motor_state_.angle_known &&
         std::abs(motor_state_.angle - motor_target_) < 1.0f &&
         motor_last_angle_ != kNoMotorTarget)) {
      motor_target_ = kNoMotorTarget;
      return;
    }
    if (now - motor_moved_ms_ < 1200 ||
        (motor_have_state_ && motor_state_.status == 4)) {
      return;  // later, from MotorStatus (polled every frame by the title)
    }
    const int32_t target = motor_target_;
    motor_target_ = kNoMotorTarget;
    if (motor_->SetTilt(target)) {
      motor_last_angle_ = target;
      motor_moved_ms_ = now;
      XELOGI("Kinect: motor tilting to {} degrees", target);
    }
  }

  // Wakes the worker for a new request or camera frame.
  void Notify() { work_event_->Set(); }

  X_STATUS Handle(CameraRequestCode code, uint32_t request_ptr,
                  uint32_t& bytes) {
    switch (code) {
      case CameraRequestCode::kGetState:
        Guest<X_PSCAM_STATE_REQUEST>(request_ptr)->state =
            !camera_     ? kDeviceStateAbsent
            : streaming_ ? kDeviceStateStreaming
                         : kDeviceStateReady;
        return X_STATUS_SUCCESS;
      case CameraRequestCode::kSetControls: {
        // Color frames carry the generation once this request has completed
        // (see Complete). The window and gains are applied to the frames
        // handed over (ViewColorFrame), not to the sensor.
        auto controls = Guest<X_PSCAM_CONTROLS_REQUEST>(request_ptr);
        controls->settings_generation = ++requested_generation_;
        ColorSettings settings;
        ColorSettings previous;
        {
          std::lock_guard<xe_mutex> lock(mutex_);
          previous = color_settings_;
        }
        settings.window = controls->window == 1;
        if (settings.window) {
          settings.window_x = controls->window_x;
          settings.window_y = controls->window_y;
        }
        const uint32_t flags = controls->flags;
        float requested_gains[3];
        for (uint32_t i = 0; i < 3; ++i) {
          requested_gains[i] =
              flags & (2u << i)
                  ? std::clamp(float(uint32_t(controls->gains[i])) / 65536.0f,
                               0.0f, 8.0f)
                  : previous.requested_gains[i];
          settings.requested_gains[i] = requested_gains[i];
          settings.gains[i] =
              cvars::kinect_colour_gains
                  ? std::min(requested_gains[i],
                             float(std::clamp(cvars::kinect_colour_gain_max,
                                              1.0, 8.0)))
                  : 1.0f;
        }
        {
          std::lock_guard<xe_mutex> lock(mutex_);
          const bool gains_changed =
              std::memcmp(settings.requested_gains,
                          color_settings_.requested_gains,
                          sizeof(settings.requested_gains)) != 0;
          if (settings.window != color_settings_.window ||
              ((settings.window || gains_changed) && controls_logged_ < 60)) {
            ++controls_logged_;
            XELOGI("Kinect: color window {} (at {}, {}), flags {:08X}, gains "
                   "{:.2f} {:.2f} {:.2f}{}, exposure {:.3f}, settings "
                   "generation {}",
                   settings.window ? "on" : "off", settings.window_x,
                   settings.window_y, flags, requested_gains[0],
                   requested_gains[1], requested_gains[2],
                   !cvars::kinect_colour_gains ? " (not applied)"
                   : requested_gains[0] > cvars::kinect_colour_gain_max ||
                           requested_gains[1] > cvars::kinect_colour_gain_max ||
                           requested_gains[2] > cvars::kinect_colour_gain_max
                       ? " (capped)"
                       : "",
                   float(uint32_t(controls->exposure_time)) / 65536.0f,
                   uint32_t(requested_generation_));
          }
          color_settings_ = settings;
        }
        return X_STATUS_SUCCESS;
      }
      case CameraRequestCode::kGetControls: {
        // The settings as code 35 sends them (dcwestra/xenia-edge): flags
        // whose low four bits mark the exposure values valid (or the
        // runtime asks again), the resolution, no window, then the
        // exposure's 16.16 limits and values. A value of 0 would read back
        // as no exposure at all.
        if (!IsGuestRange(request_ptr, 0x60)) {
          return X_STATUS_SUCCESS;
        }
        auto words = Guest<xe::be<uint32_t>>(request_ptr);
        words[7] = 0x8000000F;
        words[8] = 2;
        for (uint32_t i = 9; i < 15; ++i) {
          words[i] = 0;
        }
        words[15] = 0x1999;
        words[16] = 0x1999;
        words[17] = 0xCCCC;
        words[18] = 0xCCCC;
        for (uint32_t i = 19; i < 23; ++i) {
          words[i] = 0x10000;
        }
        words[23] = 0;
        return X_STATUS_SUCCESS;
      }
      case CameraRequestCode::kStop:
      case CameraRequestCode::kConfigure:
      case CameraRequestCode::kUnknown33:
      case CameraRequestCode::kTimeSync:
        return X_STATUS_SUCCESS;
      case CameraRequestCode::kVersion: {
        auto version = Guest<X_PSCAM_VERSION_REQUEST>(request_ptr);
        version->major = kDriverVersionMajor;
        version->minor = kDriverVersionMinor;
        return X_STATUS_SUCCESS;
      }
      case CameraRequestCode::kRegistration:
        WriteRegistration(request_ptr);
        return X_STATUS_SUCCESS;
      case CameraRequestCode::kDepthTable:
        return WriteDepthTable(request_ptr);
      case CameraRequestCode::kTiming: {
        auto timing = Guest<X_PSCAM_TIMING_REQUEST>(request_ptr);
        timing->depth_frame_period = kFramePeriodUs;
        std::memset(timing->unknown_08, 0, sizeof(timing->unknown_08));
        timing->color_frame_period = kFramePeriodUs;
        std::memset(timing->unknown_1C, 0, sizeof(timing->unknown_1C));
        timing->frame_period = kFramePeriodUs;
        std::memset(timing->unknown_28, 0, sizeof(timing->unknown_28));
        return X_STATUS_SUCCESS;
      }
      default:
        XELOGE("Kinect: unsupported request code {}",
               static_cast<uint32_t>(code));
        return X_STATUS_NOT_SUPPORTED;
    }
  }

  void WriteRegistration(uint32_t request_ptr) {
    const hid::kinect::UsbCameraCalibration& calibration =
        camera_->calibration();
    auto request = Guest<X_PSCAM_REGISTRATION_REQUEST>(request_ptr);
    static_assert(sizeof(request->registration) ==
                  sizeof(calibration.registration));
    for (size_t i = 0; i < xe::countof(request->registration); ++i) {
      request->registration[i] =
          xe::load<uint32_t>(calibration.registration.data() + i * 4);
    }
    request->reference_distance = calibration.reference_distance;
    request->reference_pixel_size = calibration.reference_pixel_size;
    request->frame_width = static_cast<uint16_t>(kFrameWidth);
    request->shift_to_depth_pixel_const = kShiftToDepthPixelConst;
    request->shift_to_depth_const_offset = kShiftToDepthConstOffset;
    request->dcmos_rcmos_distance = calibration.dcmos_rcmos_distance;
  }

  X_STATUS WriteDepthTable(uint32_t request_ptr) {
    auto request = Guest<X_PSCAM_DEPTH_TABLE_REQUEST>(request_ptr);
    const uint32_t table_ptr = request->table_ptr;
    const uint32_t entries = request->entry_count;
    if (entries != kDepthTableEntries ||
        !IsGuestRange(table_ptr, entries * sizeof(uint16_t))) {
      return X_STATUS_INVALID_PARAMETER;
    }
    const hid::kinect::UsbCameraCalibration& calibration =
        camera_->calibration();
    auto table = Guest<xe::be<uint16_t>>(table_ptr);
    for (uint32_t raw = 0; raw < entries; ++raw) {
      table[raw] = RawDepthToMillimetres(raw, calibration);
    }
    return X_STATUS_SUCCESS;
  }

  // Unpacks big-endian 11-bit samples into big-endian 16-bit words.
  static void UnpackDepth(const std::vector<uint8_t>& packed, uint8_t* out,
                          size_t pixels) {
    const bool keep_no_depth = cvars::kinect_usb_keep_no_depth;
    uint32_t bits = 0;
    int bit_count = 0;
    size_t in = 0;
    for (size_t i = 0; i < pixels; ++i) {
      while (bit_count < 11) {
        bits = (bits << 8) | packed[in++];
        bit_count += 8;
      }
      bit_count -= 11;
      uint16_t raw = static_cast<uint16_t>((bits >> bit_count) & 0x7FF);
      // Pixels without depth stay 2047, as the sensor sends them, or 0
      // (kinect_usb_keep_no_depth).
      if (raw == kNoDepthRaw && !keep_no_depth) {
        raw = 0;
      }
      xe::store_and_swap<uint16_t>(out + i * 2, raw);
    }
  }

  bool CompleteFrame(const PendingRequest& read, bool depth) {
    // Titles whose color buffers hold two bytes a pixel take YUV 4:2:2.
    const bool color16 =
        !depth && Guest<X_PSCAM_READ_FRAME_REQUEST>(read.request_ptr)
                          ->buffer_size >= kColorFrameBytes * 2;
    const uint32_t frame_bytes = depth     ? kDepthFrameBytes
                                 : color16 ? kColorFrameBytes * 2
                                           : kColorFrameBytes;
    uint32_t& last_sequence = depth ? depth_sequence_ : color_sequence_;
    const uint32_t sequence =
        depth ? camera_->CopyDepthFrame(last_sequence, frame_)
              : camera_->CopyColorFrame(last_sequence, frame_);
    if (!sequence) {
      return false;
    }
    last_sequence = sequence;

    PendingRequest done = read;
    auto request = Guest<X_PSCAM_READ_FRAME_REQUEST>(read.request_ptr);
    const uint32_t buffer = request->buffer_ptr;
    if (request->buffer_size < frame_bytes ||
        !IsGuestRange(buffer, frame_bytes)) {
      if (diag_invalid_++ < 10) {
        XELOGE(
            "Kinect: {} frame request rejected: buffer_size={} needed={} "
            "buffer={:08X}",
            depth ? "depth" : "color",
            static_cast<uint32_t>(request->buffer_size), frame_bytes, buffer);
      }
      done.status = X_STATUS_INVALID_PARAMETER;
      Complete(done);
      return true;
    }
    if (depth) {
      if (frame_.size() * 8 < size_t(frame_bytes / 2) * 11) {
        done.status = X_STATUS_INVALID_PARAMETER;
        Complete(done);
        return true;
      }
      UnpackDepth(frame_, Guest<uint8_t>(buffer), frame_bytes / 2);
      if (recording_) {
        // Each depth frame the title gets, with the newest color frame.
        camera_->CopyColorFrame(0, record_color_);
        if (!hid::kinect::WriteRecordingFrame(cvars::kinect_record_path,
                                              record_index_, frame_,
                                              record_color_)) {
          XELOGE("Kinect: recording stopped: can't write frame {}",
                 record_index_);
          recording_ = false;
        } else {
          ++record_index_;
        }
      }
      const int64_t now = SteadyMs();
      int64_t expected = 0;
      first_depth_ms_.compare_exchange_strong(expected, now);
      last_depth_ms_ = now;
    } else {
      ColorSettings settings;
      {
        std::lock_guard<xe_mutex> lock(mutex_);
        settings = color_settings_;
      }
      const bool plain = !settings.window && settings.gains[0] == 1.0f &&
                         settings.gains[1] == 1.0f && settings.gains[2] == 1.0f;
      if (color16) {
        if (!color16_logged_) {
          color16_logged_ = true;
          XELOGI("Kinect: the title takes 16-bit color, sent as YUV 4:2:2");
        }
        color_view_.resize(kColorFrameBytes);
        if (plain || frame_.size() < kColorFrameBytes) {
          std::memset(color_view_.data(), 0x80, kColorFrameBytes);
          std::memcpy(color_view_.data(), frame_.data(),
                      std::min<size_t>(kColorFrameBytes, frame_.size()));
        } else {
          ViewColorFrame(frame_.data(), settings, color_view_.data());
        }
        BayerToUyvy(color_view_.data(), Guest<uint8_t>(buffer));
      } else if (plain || frame_.size() < kColorFrameBytes) {
        std::memcpy(Guest<uint8_t>(buffer), frame_.data(),
                    std::min<size_t>(frame_bytes, frame_.size()));
      } else {
        ViewColorFrame(frame_.data(), settings, Guest<uint8_t>(buffer));
      }
      request->settings_generation = settings_generation_;
      if (settings_generation_ != last_delivered_generation_ &&
          generation_logged_ < 60) {
        ++generation_logged_;
        XELOGI("Kinect: first color frame of settings generation {} (window "
               "{}), frame {}",
               uint32_t(settings_generation_), settings.window ? "on" : "off",
               sequence);
      }
      last_delivered_generation_ = settings_generation_;
    }
    const uint32_t counter_ptr = request->frame_counter_ptr;
    if (counter_ptr) {
      *Guest<xe::be<uint32_t>>(counter_ptr) = sequence;
    }
    request->sequence = sequence;
    request->timestamp_ms =
        std::max<uint64_t>(Clock::QueryGuestUptimeMillis(), 1);
    if (depth) {
      ++depth_delivered_;
      if (depth_delivered_ <= 3 || depth_delivered_ % 300 == 0) {
        LogDepthStats(buffer);
      }
      if (depth_delivered_ % 15 == 0) {
        MeasureFloor(buffer);
      }
    } else {
      ++color_delivered_;
    }
    done.bytes = frame_bytes;
    Complete(done);
    return true;
  }

  // Does not read the request, which the title may have reused.
  void Complete(const PendingRequest& request) {
    if (request.completion) {
      uint64_t args[] = {request.request_ptr,
                         static_cast<uint32_t>(request.status), request.bytes};
      kernel_state_->processor()->Execute(worker_thread_->thread_state(),
                                          request.completion, args,
                                          xe::countof(args));
    }
    if (request.new_settings) {
      settings_generation_ = request.generation;
    }
  }

  void WorkerMain() {
    while (worker_running_) {
      std::deque<PendingRequest> completions;
      PendingRequest depth_read;
      PendingRequest color_read;
      uint32_t stop_count;
      xe::threading::Wait(work_event_.get(), false, kWorkerWaitTimeout);
      {
        std::lock_guard<xe_mutex> lock(mutex_);
        completions.swap(completions_);
        // Taken out of the queues so that a stop can't cancel them meanwhile.
        if (!depth_reads_.empty()) {
          depth_read = depth_reads_.front();
          depth_reads_.pop_front();
        }
        if (!color_reads_.empty()) {
          color_read = color_reads_.front();
          color_reads_.pop_front();
        }
        stop_count = stop_count_;
      }
      for (PendingRequest& request : completions) {
        for (PendingRequest& read : request.cancelled_reads) {
          read.status = kStatusCancelled;
          Complete(read);
        }
        if (request.completion) {
          Complete(request);
        }
      }
      for (bool depth : {true, false}) {
        PendingRequest& read = depth ? depth_read : color_read;
        if (!read.request_ptr || CompleteFrame(read, depth)) {
          continue;
        }
        // No new frame yet: requeue, unless streaming was stopped meanwhile.
        bool cancelled;
        {
          std::lock_guard<xe_mutex> lock(mutex_);
          cancelled = stop_count != stop_count_;
          if (!cancelled) {
            (depth ? depth_reads_ : color_reads_).push_front(read);
          }
        }
        if (cancelled) {
          read.status = kStatusCancelled;
          Complete(read);
        }
      }
    }
  }

  KernelState* kernel_state_;
  Memory* memory_;
  xe_mutex mutex_;
  std::unique_ptr<xe::threading::Event> work_event_ =
      xe::threading::Event::CreateAutoResetEvent(false);
  std::deque<PendingRequest> completions_;
  // Incremented by every stop (code 2).
  uint32_t stop_count_ = 0;
  std::deque<PendingRequest> depth_reads_;
  std::deque<PendingRequest> color_reads_;
  std::unique_ptr<hid::kinect::UsbCamera> camera_;
  std::unique_ptr<hid::kinect::UsbMotor> motor_;
  xe_mutex motor_mutex_;
  int32_t motor_target_ = kNoMotorTarget;
  int32_t motor_last_angle_ = kNoMotorTarget;
  int64_t motor_moved_ms_ = 0;
  int64_t motor_polled_ms_ = 0;
  bool motor_have_state_ = false;
  hid::kinect::UsbMotorState motor_state_;
  object_ref<XHostThread> worker_thread_;
  std::atomic<bool> worker_running_ = true;
  uint32_t depth_sequence_ = 0;
  uint32_t color_sequence_ = 0;
  std::atomic<bool> streaming_ = false;
  // Generation of the latest camera settings request, and of the latest one
  // whose completion has run, which color frames are tagged with.
  uint8_t requested_generation_ = 0;
  // Diagnostics for Kinect ID's face window (Kinect Adventures).
  uint32_t controls_logged_ = 0;
  uint32_t generation_logged_ = 0;
  uint8_t last_delivered_generation_ = 0;
  // The windowed Bayer frame, before 16-bit titles' YUV conversion.
  std::vector<uint8_t> color_view_;
  bool color16_logged_ = false;
  std::atomic<uint8_t> settings_generation_ = 0;
  // The window and gains of the latest camera settings request.
  ColorSettings color_settings_;
  std::vector<uint8_t> frame_;
  // kinect_record_path.
  bool recording_ = false;
  uint32_t record_index_ = 0;
  std::vector<uint8_t> record_color_;
  std::atomic<int64_t> first_depth_ms_{0};
  std::atomic<int64_t> last_depth_ms_{0};
  // Diagnostic counters.
  std::atomic<uint32_t> request_counts_[64] = {};
  std::atomic<uint32_t> diag_failures_{0};
  uint32_t depth_delivered_ = 0;
  uint32_t color_delivered_ = 0;
  uint32_t diag_invalid_ = 0;
  // MeasureFloor (worker thread).
  std::vector<uint16_t> depth_millimetres_;
  float floor_candidates_[3] = {};
  uint32_t floor_measurements_ = 0;
  uint32_t floor_higher_count_ = 0;
  uint32_t floor_logs_ = 0;
  uint32_t floor_candidate_count_ = 0;
  std::atomic<float> measured_floor_mm_{0.0f};
};

xe_mutex g_camera_device_mutex;
std::unique_ptr<CameraDevice> g_camera_device;
// Set once a real camera has opened; read from guest hooks without locking.
std::atomic<CameraDevice*> g_active_device{nullptr};

CameraDevice* GetCameraDevice() {
  std::lock_guard<xe_mutex> lock(g_camera_device_mutex);
  if (!g_camera_device) {
    // Tried once: without a camera the emulated sensor takes over for good.
    g_camera_device = std::make_unique<CameraDevice>(kernel_state());
    if (g_camera_device->has_camera()) {
      g_active_device = g_camera_device.get();
    }
  }
  return g_camera_device.get();
}

}  // namespace

bool TryHandleRequest(uint32_t request_ptr, uint32_t* status) {
  if (!cvars::allow_kinect_camera || !request_ptr) {
    return false;
  }
  CameraDevice* device = GetCameraDevice();
  if (!device->has_camera()) {
    return false;
  }
  *status = device->Request(request_ptr);
  return true;
}

bool IsActive() { return g_active_device.load() != nullptr; }

bool MotorSetAngle(int32_t degrees) {
  CameraDevice* device = g_active_device.load();
  return device && device->MotorSetAngle(degrees);
}

bool MotorStatus(int32_t* degrees, bool* moving, int16_t accel[3]) {
  CameraDevice* device = g_active_device.load();
  return device && device->MotorStatus(degrees, moving, accel);
}

bool MeasuredFloorHeight(float* millimetres) {
  CameraDevice* device = g_active_device.load();
  if (!device) {
    return false;
  }
  *millimetres = device->MeasuredFloorHeight();
  return *millimetres > 0.0f;
}

uint64_t TimestampMs() {
  CameraDevice* device = g_active_device.load();
  return device ? device->TimestampMs() : 0;
}

TimestampMode GetTimestampMode() {
  const std::string& mode = cvars::kinect_usb_timestamps;
  if (mode == "always") {
    return TimestampMode::kAlways;
  }
  if (mode == "never") {
    return TimestampMode::kNever;
  }
  return TimestampMode::kAuto;
}

void Shutdown() {
  std::unique_ptr<CameraDevice> camera_device;
  {
    std::lock_guard<xe_mutex> lock(g_camera_device_mutex);
    g_active_device = nullptr;
    camera_device = std::move(g_camera_device);
  }
  // Destroyed unlocked, as its worker may still be completing a request.
  camera_device.reset();
}

}  // namespace usb_kinect
}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe
