/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/emulator.h"
#include "xenia/kernel/kernel_flags.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/nui.h"
#include "xenia/kernel/xam/nui_data_store.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xam/xam_private.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_usb_kinect.h"
#include "xenia/xbox.h"

DECLARE_bool(kinect_id);

DEFINE_bool(kinect_remember_floor, true,
            "Kinect: keep the floor titles' NUI runtime remembers "
            "(XamNuiCameraRememberFloor) and hand it back in the tilt status, "
            "so its auto-tilt aims at fixed angles instead of searching. Only "
            "with a motor (the SDK bridge's, or a real Kinect's with "
            "kinect_usb_motor).",
            "HID");

DEFINE_double(
    kinect_floor_height, 0.0,
    "Kinect: the sensor's height above the floor in millimetres, reported as "
    "the floor the console remembers until a title remembers one itself "
    "(which is then kept in <content folder>/nui). Kinect Adventures needs a "
    "floor to aim the motor instead of searching for one forever. 0 = none; "
    "a console reports it in XamNuiCameraTiltGetStatus (one measured 593).",
    "HID");

DEFINE_bool(
    kinect_floor_auto, true,
    "Kinect: measure the floor from the real Kinect's depth and accelerometer "
    "whenever the sensor has settled, as a console does by itself, when "
    "kinect_floor_height is 0 and no title has remembered a floor this run. "
    "Measurements are kept for the next runs.",
    "HID");

namespace xe {
namespace kernel {
namespace xboxkrnl {
// Kinect SDK bridge tilt channel (xboxkrnl_usbcam.cc, kinect_auto_tilt).
bool KinectBridgeRequestTilt(int32_t angle_deg);
bool KinectBridgeTiltStatus(int32_t* angle_deg, bool* moving);
}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

namespace xe {
namespace kernel {
namespace xam {
// https://web.cs.ucdavis.edu/~okreylos/ResDev/Kinect/MainPage.html

struct X_NUI_DEVICE_STATUS {
  /* Notes:
     - for one side func of XamNuiGetDeviceStatus
       - if some data address is less than zero then tilt_status = it
       - else another func is called and its return can set
     tilt_status = c0051200 or some value involving DetroitDeviceRequest
       - next PsCamDeviceRequest is called and if its return is less than zero
     then camera_status = return of PsCamDeviceRequest
       - else it equals an unknown local_1c
       - finally McaDeviceRequest is called and if its return is less than zero
     then mic_status = return of McaDeviceRequest
       - else it equals an unknown local_14
     - status_bits can be set to status_bits | 0x44 or | 0x40
  */
  xe::be<uint32_t> camera_status;
  xe::be<uint32_t> tilt_status;
  xe::be<uint32_t> mic_status;
  xe::be<uint32_t> status_bits;
};
static_assert_size(X_NUI_DEVICE_STATUS, 0x10);

// Get
dword_result_t XamNuiGetDeviceStatus_entry(
    pointer_t<X_NUI_DEVICE_STATUS> status_ptr) {
  /* Notes:
     - it does return a value that is not always used
     - returns values are X_ERROR_SUCCESS, 0xC0050006, and others
     - 1) On func start *status_ptr = 0, status_ptr->tilt_status = 0,
     status_ptr->mic_status = 0, and status_ptr->status_bits = 0
     - 2) calls XamXStudioRequest(6,&var <- = 0);
     - if return is greater than -1 && var & 0x80000000 != 0 then set
     status_ptr->tilt_status = 0xC000009D, status_ptr->mic_status = 0xC000009D,
     and status_ptr->status_bits = 0x20
     - lots of branching functions after
  */

  status_ptr.Zero();

  const bool kinect_initialized = kernel_state()->nui()->NuiInitialized();

  // Bit 0x1 = sensor ready (see the XNuiTitleInitialize notes in
  // xam_app.cc: titles open the troubleshooter when 0x1 is clear but
  // 0x4/0x40 are set).
  status_ptr->status_bits = kinect_initialized ? 0x1 : 0x0;
  const uint32_t result = kinect_initialized ? X_ERROR_SUCCESS : 0xC0050006;

  XELOGD(
      "XamNuiGetDeviceStatus({:08X}) -> {:08X} [camera={:08X} tilt={:08X} "
      "mic={:08X} bits={:08X}]",
      status_ptr.guest_address(), result,
      static_cast<uint32_t>(status_ptr->camera_status),
      static_cast<uint32_t>(status_ptr->tilt_status),
      static_cast<uint32_t>(status_ptr->mic_status),
      static_cast<uint32_t>(status_ptr->status_bits));
  return result;
}
DECLARE_XAM_EXPORT1(XamNuiGetDeviceStatus, kNone, kStub);

dword_result_t XamUserNuiGetUserIndex_entry(unknown_t unk, lpdword_t index) {
  // Which signed-in user a Kinect ID enrollment is (Kinect Adventures asks
  // for each of 8). None are bound here.
  static std::atomic<uint32_t> logged{0};
  if (logged.fetch_add(1) < 16) {
    XELOGI("XamUserNuiGetUserIndex({:X}) -> no such user", uint32_t(unk));
  }
  return X_E_NO_SUCH_USER;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndex, kNone, kStub);

dword_result_t XamUserNuiGetEnrollmentIndex_entry(dword_t user_index,
                                                  lpdword_t enrollment_index) {
  // No biometric enrollment exists without real identity tracking.
  XELOGD("XamUserNuiGetEnrollmentIndex({:08X}, {:08X})", user_index.value(),
         enrollment_index.guest_address());
  if (enrollment_index) {
    *enrollment_index = 0xFFFFFFFF;
  }
  return X_E_NO_SUCH_USER;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetEnrollmentIndex, kNone, kStub);

dword_result_t XamUserNuiGetUserIndexForSignin_entry(lpdword_t user_index) {
  if (!user_index) {
    return X_E_INVALIDARG;
  }

  for (uint32_t i = 0; i < XUserMaxUserCount; i++) {
    auto profile = kernel_state()->xam_state()->GetUserProfile(i);
    if (profile) {
      *user_index = i;
      return X_E_SUCCESS;
    }
  }

  return X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndexForSignin, kNone, kImplemented);

dword_result_t XamUserNuiGetUserIndexForBind_entry(lpdword_t user_index) {
  if (!user_index) {
    return X_E_INVALIDARG;
  }

  return X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndexForBind, kNone, kStub);

struct X_NUI_DEPTH_CALIBRATION {
  uint8_t data[0x30];
};
static_assert_size(X_NUI_DEPTH_CALIBRATION, 0x30);

dword_result_t XamNuiGetDepthCalibration_entry(
    pointer_t<X_NUI_DEPTH_CALIBRATION> depth_data) {
  // always cleared before new data written
  std::memset(depth_data, 0, sizeof(X_NUI_DEPTH_CALIBRATION));
  return X_STATUS_NO_SUCH_FILE | X_FACILITY_NT_BIT;
}
DECLARE_XAM_EXPORT1(XamNuiGetDepthCalibration, kNone, kStub);

// Skeleton
qword_result_t XamNuiSkeletonGetBestSkeletonIndex_entry(int_t unk) {
  return 0xffffffffffffffff;
}
DECLARE_XAM_EXPORT1(XamNuiSkeletonGetBestSkeletonIndex, kNone, kStub);

/* XamNuiCamera Notes
   - most require message calls to xam in 0x0002Bxxx area
*/
dword_result_t XamNuiCameraTiltSetCallback_entry(dword_t callback) {
  kernel_state()->nui()->SetTiltCallback(callback);
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiCameraTiltSetCallback, kNone, kStub);

// The floor the title's NUI runtime last handed XamNuiCameraRememberFloor,
// with the accelerometer legs of that moment. The runtime's auto-tilt driver
// reads it back from the tilt status block's tail (+0x30..+0x4F): with a floor
// (d, the sensor's height in millimetres, non-zero) it aims at absolute
// targets; with none it searches blind - -25, then steps of +10, plus relative
// offsets after each move - which is the "Kinect is trying to adjust to a good
// angle" loop. Layout and use from the xenia-canary-kinect fork's notes
// (ryantimjohn, docs/nui.md, "The tilt status block").
//
// A console keeps the floor across titles and reboots, so a title normally
// starts with one (seen on a console: d 593 mm before Kinect Adventures had
// done anything), and after every motor move the plane's normal is the new
// gravity direction, the height staying - which is how it's reported here:
// the remembered height (a title's last XamNuiCameraRememberFloor, kept in the
// NUI data store, or kinect_floor_height), the plane from current gravity.
struct RememberedFloor {
  bool loaded = false;
  // Millimetres; 0 if none. Kept in the NUI data store.
  float height = 0.0f;
  // Whether a title remembered height during this run.
  bool title_remembered = false;
  // Consecutive queries with a measurement far from height.
  uint32_t moved_count = 0;
};
static std::mutex remembered_floor_mutex_;
static RememberedFloor remembered_floor_;
static const char kRememberedFloorKey[] = "KinectFloor";

// The floor height to report, 0 if none. Takes remembered_floor_mutex_.
static float RememberedFloorHeight() {
  std::lock_guard<std::mutex> lock(remembered_floor_mutex_);
  if (!remembered_floor_.loaded) {
    remembered_floor_.loaded = true;
    uint64_t version;
    std::vector<uint8_t> data;
    if (NuiDataStore::Get()->Read(NuiDataStore::Kind::kNamed,
                                  kRememberedFloorKey, version, data) &&
        version == 1 && data.size() >= sizeof(float)) {
      float height;
      std::memcpy(&height, data.data(), sizeof(height));
      if (std::isfinite(height) && height > 0.0f) {
        remembered_floor_.height = height;
        XELOGI("Kinect: the remembered floor is {:.0f} mm below the sensor",
               height);
      }
    }
  }
  // A title's own floor from this run, then the configured height, then the
  // kept floor (a title's - Kinect Adventures remembers the floor it finds
  // itself once it has one to start from - or an earlier measurement), and
  // Xenia's measurement only if there is none or the sensor was clearly moved
  // (another height for about 10 s), as the console's own measurement mostly
  // gets things started.
  if (remembered_floor_.title_remembered) {
    return remembered_floor_.height;
  }
  if (cvars::kinect_floor_height > 0.0) {
    return static_cast<float>(cvars::kinect_floor_height);
  }
  float measured;
  if (cvars::kinect_floor_auto &&
      xboxkrnl::usb_kinect::MeasuredFloorHeight(&measured)) {
    bool use_measured = remembered_floor_.height <= 0.0f;
    if (!use_measured &&
        std::abs(remembered_floor_.height - measured) > 150.0f) {
      use_measured = ++remembered_floor_.moved_count >= 300;
    } else {
      remembered_floor_.moved_count = 0;
    }
    if (use_measured) {
      remembered_floor_.moved_count = 0;
      XELOGI("Kinect: using the measured floor, {:.0f} mm below the sensor",
             measured);
      remembered_floor_.height = measured;
      NuiDataStore::Get()->Write(NuiDataStore::Kind::kNamed,
                                 kRememberedFloorKey, 1,
                                 reinterpret_cast<const uint8_t*>(&measured),
                                 sizeof(measured));
    }
  }
  return remembered_floor_.height;
}

// The accelerometer legs for an angle, raw Kinect counts (about 819 per g):
// x across, y the cosine leg, z the sine leg, so atan2(z, y) is the angle.
static void TiltLegsForAngle(int32_t angle_deg, int32_t legs[3]) {
  const double radians = angle_deg * 0.017453292519943295;
  legs[0] = 0;
  legs[1] = static_cast<int32_t>(std::lround(std::cos(radians) * 819.0));
  legs[2] = static_cast<int32_t>(std::lround(std::sin(radians) * 819.0));
}

// The legs for the block: a real Kinect's own accelerometer when its motor is
// open (raw counts, y about +819 when level, z negative looking down, so
// atan2(z, y) is where the head really points even if the motor didn't get
// there), else from the reported angle. x stays 0: which way the runtime
// counts a sideways roll isn't known (as in the xenia-canary-kinect fork).
static void TiltLegs(int32_t angle_deg, int32_t legs[3]) {
  int32_t real_angle;
  bool real_moving;
  int16_t accel[3] = {};
  if (xboxkrnl::usb_kinect::MotorStatus(&real_angle, &real_moving, accel) &&
      accel[1] > 409 && accel[1] < 1229) {
    // Scaled to 819 per g like a console reports (a console's legs are about
    // 819 long; this Kinect's raw ones were about 850).
    const double y = accel[1], z = accel[2];
    const double length = std::sqrt(y * y + z * z);
    legs[0] = 0;
    legs[1] = static_cast<int32_t>(std::lround(y * 819.0 / length));
    legs[2] = static_cast<int32_t>(std::lround(z * 819.0 / length));
    return;
  }
  TiltLegsForAngle(angle_deg, legs);
}

dword_result_t XamNuiCameraTiltGetStatus_entry(
    pointer_t<X_NUI_TILT_STATUS> tilt_status) {
  /* Notes:
     - Used by XamNuiCameraElevationGetAngle, and XamNuiCameraSetFlags
     - if it returns anything greater than -1 then both above functions continue
     - Both funcs request 0x50 bytes with XtSs header
  */
  if (!tilt_status || tilt_status->buffer_size < sizeof(X_NUI_TILT_STATUS) ||
      tilt_status->signature != make_fourcc("XtSs")) {
    return X_E_FAIL;
  }
  // With the Kinect SDK bridge running, report its motor. The NUI runtime's
  // auto-tilt at game start polls this every frame: it reads the angle as
  // atan2(gravity[2], gravity[1]) and waits while flags & 3 (moving); a
  // failure here makes it give up after its first move (the "look down for
  // the floor" one, -25 degrees), leaving the Kinect pointing at the floor.
  int32_t bridge_angle;
  bool bridge_moving;
  if (xboxkrnl::KinectBridgeTiltStatus(&bridge_angle, &bridge_moving)) {
    std::memset(reinterpret_cast<uint8_t*>(tilt_status.host_address()) + 8, 0,
                sizeof(X_NUI_TILT_STATUS) - 8);
    // Gravity as the accelerometer would see it at that angle (raw Kinect
    // accelerometer counts: about 819 per g).
    int32_t legs[3];
    TiltLegs(bridge_angle, legs);
    // A console fills +0x08..+0x1F with two more accelerometer triplets that
    // follow the sensor faster (latest and short-average readings, judging by
    // how they move), then the long average; one reading is all there is here.
    for (int i = 0; i < 3; ++i) {
      tilt_status->unknown_08[i] = uint32_t(legs[i]);
      tilt_status->unknown_08[3 + i] = uint32_t(legs[i]);
      tilt_status->gravity_long_avg[i] = legs[i];
    }
    tilt_status->flags = bridge_moving ? X_TILT_STATUS_FLAGS::TILT_STATUS_UNK1
                                       : X_TILT_STATUS_FLAGS(0);
    // The tail: the floor plane (normal = gravity direction, d = height), a
    // flags word (0; bit 0 would turn the runtime's auto-tilt off) and the
    // legs the plane goes with - as a console reports it.
    float floor_height =
        cvars::kinect_remember_floor ? RememberedFloorHeight() : 0.0f;
    if (floor_height > 0.0f) {
      const double gravity_length =
          std::sqrt(double(legs[0]) * legs[0] + double(legs[1]) * legs[1] +
                    double(legs[2]) * legs[2]);
      if (gravity_length > 0.0) {
        float plane[4] = {float(legs[0] / gravity_length),
                          float(legs[1] / gravity_length),
                          float(legs[2] / gravity_length), floor_height};
        auto* tail =
            reinterpret_cast<xe::be<uint32_t>*>(tilt_status->unknown_30);
        for (int i = 0; i < 4; ++i) {
          uint32_t bits;
          std::memcpy(&bits, &plane[i], sizeof(bits));
          tail[i] = bits;
        }
        tail[4] = 0;
        for (int i = 0; i < 3; ++i) {
          tail[5 + i] = uint32_t(legs[i]);
        }
      }
    }
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
      XELOGI(
          "XamNuiCameraTiltGetStatus: reporting the Kinect SDK bridge's motor "
          "({} degrees{})",
          bridge_angle, bridge_moving ? ", moving" : "");
    }
    return X_ERROR_SUCCESS;
  }
  // TODO(knuckleslee): Report the Kinect accelerometer and motor status.
  return X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiCameraTiltGetStatus, kNone, kStub);

dword_result_t XamNuiCameraElevationGetAngle_entry(lpdword_t elevation_angle,
                                                   lpdword_t camera_moving) {
  // With the Kinect SDK bridge running, report its motor's real angle.
  int32_t bridge_angle;
  bool bridge_moving;
  if (xboxkrnl::KinectBridgeTiltStatus(&bridge_angle, &bridge_moving)) {
    if (elevation_angle) {
      *elevation_angle = static_cast<uint32_t>(bridge_angle);
    }
    if (camera_moving) {
      *camera_moving = bridge_moving ? 1 : 0;
    }
    return X_ERROR_SUCCESS;
  }
  X_NUI_TILT_STATUS tilt_status = {};
  tilt_status.buffer_size = sizeof(tilt_status);
  tilt_status.signature = make_fourcc("XtSs");  // (XtSs)?
  X_STATUS result = XamNuiCameraTiltGetStatus_entry(&tilt_status);
  if (XSUCCEEDED(result)) {
    // TODO(knuckleslee): Calculate the angle in degrees from
    // gravity_long_avg[2] and gravity_long_avg[1].
    *elevation_angle = 0;
    *camera_moving =
        tilt_status.flags & (X_TILT_STATUS_FLAGS::TILT_STATUS_UNK1 |
                             X_TILT_STATUS_FLAGS::TILT_STATUS_UNK2);  // moving
    result = X_E_FAIL;  // until implementation improves
  }
  return result;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationGetAngle, kNone, kStub);

dword_result_t XamNuiCameraGetTiltControllerType_entry() {
  /* Notes:
     - Earlier decompile notes: DetroitDeviceRequest and
       (LZCOUNT(result) << 0x20) >> 0x25.
  */
  // TODO(boma): Get the controller type from connected device.
  // Currently only reports no controller or type 1.
  const bool kinect_initialized =
      kernel_state()->xconfig()->ReadSetting<uint32_t>(
          X_CONFIG_CATEGORY::XCONFIG_USER_CATEGORY, XCONFIG_USER_RETAIL_FLAGS) &
      X_RETAIL_FLAGS::KinectInitialized;
  return kinect_initialized;
}
DECLARE_XAM_EXPORT1(XamNuiCameraGetTiltControllerType, kNone, kStub);

dword_result_t XamNuiCameraSetFlags_entry(qword_t unk1, dword_t unk2) {
  X_STATUS result = X_E_DEVICE_NOT_CONNECTED;
  int controller_type = XamNuiCameraGetTiltControllerType_entry();

  if (controller_type == 1) {
    X_NUI_TILT_STATUS tilt_status = {};
    tilt_status.buffer_size = sizeof(tilt_status);
    tilt_status.signature = make_fourcc("XtSs");  // (XtSs)?
    result = XamNuiCameraTiltGetStatus_entry(&tilt_status);
    if (!XSUCCEEDED(result)) {
      return result;
    }
    // TODO(knuckleslee): Apply the flags once their meaning is known.
    return X_E_FAIL;  // until implementation improves
  }
  if (controller_type == 2) {
    return X_ERROR_SUCCESS;
  }
  if (controller_type >= 3) {
    return X_E_FAIL;  // until implementation improves
  }
  return result;
}
DECLARE_XAM_EXPORT1(XamNuiCameraSetFlags, kNone, kStub);

// Motor/tilt stubs. Kinect Adventures imports these; signatures are not yet
// reversed, so arguments are logged raw to help identify them.
dword_result_t XamNuiCameraElevationSetAngle_entry(dword_t angle,
                                                   dword_t unk2) {
  // The angle is in degrees relative to gravity (+ = up); the titles' NUI
  // runtime only calls this for -27..27 (the auto-tilt at game start).
  const int32_t degrees = static_cast<int32_t>(angle.value());
  if (degrees < -27 || degrees > 27) {
    return X_E_INVALIDARG;
  }
  const bool sent = xboxkrnl::KinectBridgeRequestTilt(degrees);
  static std::atomic<uint32_t> logged{0};
  if (logged.fetch_add(1) < 50) {
    XELOGI("XamNuiCameraElevationSetAngle({} degrees, {:08X}) -> {}", degrees,
           unk2.value(),
           sent ? "sent to the Kinect SDK bridge"
                : "not sent (no bridge heard from, or kinect_auto_tilt off)");
  }
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationSetAngle, kNone, kStub);

dword_result_t XamNuiCameraElevationStopMovement_entry() {
  XELOGD("XamNuiCameraElevationStopMovement()");
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationStopMovement, kNone, kStub);

dword_result_t XamNuiCameraTiltReportStatus_entry(dword_t unk1, dword_t unk2) {
  XELOGD("XamNuiCameraTiltReportStatus({:08X}, {:08X})", unk1.value(),
         unk2.value());
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiCameraTiltReportStatus, kNone, kStub);

// XamNuiCameraRememberFloor(plane a and b, plane c and d, flag): the floor
// plane the runtime's tracker fitted, as two pairs of floats packed in 64-bit
// arguments - the unit normal, then d, the sensor's height above the floor in
// millimetres. Kept with the accelerometer legs of the moment for
// XamNuiCameraTiltGetStatus to hand back.
dword_result_t XamNuiCameraRememberFloor_entry(qword_t plane_ab,
                                               qword_t plane_cd, dword_t flag) {
  float plane[4];
  const uint32_t words[4] = {
      uint32_t(uint64_t(plane_ab) >> 32), uint32_t(uint64_t(plane_ab)),
      uint32_t(uint64_t(plane_cd) >> 32), uint32_t(uint64_t(plane_cd))};
  std::memcpy(plane, words, sizeof(plane));
  static std::atomic<uint32_t> logged{0};
  if (logged.fetch_add(1) < 8) {
    XELOGI("XamNuiCameraRememberFloor({:.3f}, {:.3f}, {:.3f}, {:.1f} mm; "
           "flag {})",
           plane[0], plane[1], plane[2], plane[3], flag.value());
  }
  if (cvars::kinect_remember_floor && std::isfinite(plane[3]) &&
      plane[3] > 0.0f) {
    // Kept like a console does, for this run and the next ones.
    RememberedFloorHeight();  // Loaded first so it isn't loaded over this.
    std::lock_guard<std::mutex> lock(remembered_floor_mutex_);
    remembered_floor_.title_remembered = true;
    if (std::abs(remembered_floor_.height - plane[3]) >= 1.0f) {
      XELOGI("Kinect: the title remembered the floor {:.0f} mm below the "
             "sensor",
             plane[3]);
      remembered_floor_.height = plane[3];
      NuiDataStore::Get()->Write(
          NuiDataStore::Kind::kNamed, kRememberedFloorKey, 1,
          reinterpret_cast<const uint8_t*>(&plane[3]), sizeof(plane[3]));
    }
  }
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiCameraRememberFloor, kNone, kStub);

dword_result_t XamIsNuiUIActive_entry() {
  return kernel_state()->xam_state()->is_xam_dialog_present_.load();
}
DECLARE_XAM_EXPORT1(XamIsNuiUIActive, kNone, kImplemented);

dword_result_t XamNuiIsDeviceReady_entry() {
  /* device_state Notes:
   - used with XNotifyBroadcast(kXNotificationSystemNUIHardwareStatusChanged,
   device_state)
   - known values:
     - 0x0001
     - 0x0004
     - 0x0040
  */

  return kernel_state()->nui()->NuiInitialized();
}
DECLARE_XAM_EXPORT1(XamNuiIsDeviceReady, kNone, kImplemented);

dword_result_t XamIsNuiAutomationEnabled_entry(unknown_t unk1, unknown_t unk2) {
  /* Notes:
     - XamIsNuiAutomationEnabled = XamIsNatalPlaybackEnabled
     - Always returns X_E_SUCCESS? Maybe check later versions
     - Recieves param but never interacts with them
     - No Operations
  */
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT2(XamIsNuiAutomationEnabled, kNone, kStub, kHighFrequency);

dword_result_t XamIsNatalPlaybackEnabled_entry(unknown_t unk1, unknown_t unk2) {
  /* Notes:
     - XamIsNuiAutomationEnabled = XamIsNatalPlaybackEnabled
     - Always returns X_E_SUCCESS? Maybe check later versions
     - Recieves param but never interacts with them
     - No Operations
  */
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT2(XamIsNatalPlaybackEnabled, kNone, kStub, kHighFrequency);

dword_result_t XamNuiIsChatMicEnabled_entry() {
  /* Notes:
     - call ExGetXConfigSetting(7,9,&xconfig_nui,0x1c,local_40);
     - check xconfig_nui.flag & 0x8
     - return True or false based on result
  */
  return false;
}
DECLARE_XAM_EXPORT1(XamNuiIsChatMicEnabled, kNone, kImplemented);

/* HUD Notes:
   - XamNuiHudGetEngagedTrackingID, XamNuiHudIsEnabled,
   XamNuiHudSetEngagedTrackingID, XamNuiHudInterpretFrame, and
   XamNuiHudGetEngagedEnrollmentIndex all utilize the same data address
*/
uint32_t nui_unknown_1 = 0;
uint32_t engaged_tracking_id = 0;
char nui_unknown_2 = '\0';

dword_result_t XamNuiHudSetEngagedTrackingID_entry(dword_t id) {
  auto nui_ = kernel_state()->nui();
  if (id) {
    if (!nui_->GetNUIDataPtr()) {
      return X_E_FAIL;
    }
    nui_->SetEngagedTrackingId(id);
  }

  return X_STATUS_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiHudSetEngagedTrackingID, kNone, kImplemented);

qword_result_t XamNuiHudGetEngagedTrackingID_entry() {
  auto nui_ = kernel_state()->nui();
  if (nui_->GetNUIDataPtr()) {
    return nui_->GetEngagedTrackingId();
  }

  return X_STATUS_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiHudGetEngagedTrackingID, kNone, kImplemented);

dword_result_t XamNuiHudIsEnabled_entry() {
  auto nui_ = kernel_state()->nui();
  bool result = XamNuiIsDeviceReady_entry();
  if (nui_->GetNUIDataPtr() && nui_->GetUnknown2() && result) {
    return true;
  }
  return false;
}
DECLARE_XAM_EXPORT1(XamNuiHudIsEnabled, kNone, kImplemented);

dword_result_t XamNuiHudGetInitializeFlags_entry() {
  return kernel_state()->nui()->GetInitFlags();
}
DECLARE_XAM_EXPORT1(XamNuiHudGetInitializeFlags, kNone, kImplemented);

void XamNuiHudGetVersions_entry(lpqword_t xam_version_ptr,
                                lpqword_t xtl_version_ptr) {
  auto nui_ = kernel_state()->nui();
  if (xam_version_ptr) {
    *xam_version_ptr = nui_->GetNUIVersion(0);
  }
  if (xtl_version_ptr) {
    *xtl_version_ptr = nui_->GetNUIVersion(1);
  }
}
DECLARE_XAM_EXPORT1(XamNuiHudGetVersions, kNone, kImplemented);

/* XamNuiIdentity Notes:
   - most require message calls to xam in 0x0002Cxxx area
*/
qword_result_t XamNuiIdentityGetSessionId_entry() {
  auto nui_ = kernel_state()->nui();
  if (nui_->GetSessionId() == 0) {
    // xboxkrnl::XeCryptRandom_entry(NUI_Session_Id, 8);
    nui_->SetSessionId(0xDEADF00DDEADF00D);
  }
  return nui_->GetSessionId();
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetSessionId, kNone, kImplemented);

dword_result_t XamNuiIdentityEnrollForSignIn_entry(dword_t unk1, qword_t unk2,
                                                   qword_t unk3, dword_t unk4) {
  /* Notes:
     - Decompiler issues so double check
  */
  if (XamNuiHudIsEnabled_entry() == false) {
    return X_E_FAIL;
  }
  // buffer [2]
  // buffer[0] = unk
  // var = unk4
  // return func(0xfe,0x2c010,buffer,0xc,unk3);
  return X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityEnrollForSignIn, kNone, kStub);

dword_result_t XamNuiIdentityAbort_entry(dword_t unk) {
  if (XamNuiHudIsEnabled_entry() == false) {
    return X_E_FAIL;
  }
  // buffer [4]
  // buffer[0] = unk
  // return func(0xfe,0x2c00e,buffer,4,0)
  return X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityAbort, kNone, kStub);

// A profile's biometric (Kinect ID) data, which Kinect titles' identity
// runtime writes when it enrolls a player and reads to recognize them. Kept
// in <content folder>/nui (kinect_id). Based on the xenia-edge fork.
static X_RESULT CompleteBiometricRequest(uint32_t overlapped_ptr,
                                         X_RESULT result) {
  if (overlapped_ptr) {
    kernel_state()->CompleteOverlappedImmediate(overlapped_ptr, result);
    return X_ERROR_IO_PENDING;
  }
  return result;
}

dword_result_t XamReadBiometricData_entry(qword_t xuid, lpdword_t version_ptr,
                                          lpvoid_t buffer_ptr,
                                          lpdword_t size_ptr,
                                          dword_t overlapped_ptr) {
  if (!size_ptr) {
    return CompleteBiometricRequest(overlapped_ptr, X_ERROR_INVALID_PARAMETER);
  }
  uint64_t version = 0;
  std::vector<uint8_t> data;
  const bool found =
      cvars::kinect_id &&
      NuiDataStore::Get()->Read(NuiDataStore::Kind::kProfileBiometric,
                                NuiDataStore::ProfileKey(xuid), version, data);
  static std::atomic<uint32_t> logged{0};
  if (logged.fetch_add(1) < 8) {
    XELOGI("XamReadBiometricData({:016X}): {}", uint64_t(xuid),
           found ? fmt::format("version {:X}, {} bytes", version, data.size())
                 : std::string("none stored"));
  }
  if (!found) {
    return CompleteBiometricRequest(overlapped_ptr, X_ERROR_NOT_FOUND);
  }
  if (version_ptr) {
    *version_ptr = uint32_t(version);
  }
  const uint32_t capacity = *size_ptr;
  *size_ptr = uint32_t(data.size());
  if (!buffer_ptr || capacity < data.size()) {
    return CompleteBiometricRequest(overlapped_ptr,
                                    X_ERROR_INSUFFICIENT_BUFFER);
  }
  std::memcpy(buffer_ptr, data.data(), data.size());
  return CompleteBiometricRequest(overlapped_ptr, X_ERROR_SUCCESS);
}
DECLARE_XAM_EXPORT1(XamReadBiometricData, kUserProfiles, kImplemented);

dword_result_t XamWriteBiometricData_entry(qword_t xuid, lpdword_t version_ptr,
                                           lpvoid_t buffer_ptr,
                                           lpdword_t size_ptr,
                                           dword_t overlapped_ptr) {
  if (!version_ptr || !size_ptr || (*size_ptr && !buffer_ptr)) {
    return CompleteBiometricRequest(overlapped_ptr, X_ERROR_INVALID_PARAMETER);
  }
  XELOGI("XamWriteBiometricData({:016X}): version {:X}, {} bytes",
         uint64_t(xuid), uint32_t(*version_ptr), uint32_t(*size_ptr));
  const bool written =
      cvars::kinect_id &&
      NuiDataStore::Get()->Write(NuiDataStore::Kind::kProfileBiometric,
                                 NuiDataStore::ProfileKey(xuid), *version_ptr,
                                 buffer_ptr.as<const uint8_t*>(), *size_ptr);
  return CompleteBiometricRequest(
      overlapped_ptr, written ? X_ERROR_SUCCESS : X_ERROR_FUNCTION_FAILED);
}
DECLARE_XAM_EXPORT1(XamWriteBiometricData, kUserProfiles, kImplemented);

dword_result_t XamUserNuiEnableBiometric_entry(dword_t user_index,
                                               int_t enable) {
  // Whether the profile signs in with Kinect ID. Titles enrolling a player
  // turn it on once they've written the player's biometric data.
  if (user_index >= XUserMaxUserCount) {
    return X_E_INVALIDARG;
  }
  XELOGI("XamUserNuiEnableBiometric({}, {})", uint32_t(user_index),
         int32_t(enable));
  return cvars::kinect_id ? X_E_SUCCESS : X_E_INVALIDARG;
}
DECLARE_XAM_EXPORT1(XamUserNuiEnableBiometric, kNone, kSketchy);

struct X_NUI_PLAYER_ENGAGEMENT_UPDATE {
  uint8_t data[0x1C];
};
static_assert_size(X_NUI_PLAYER_ENGAGEMENT_UPDATE, 0x1C);

void XamNuiPlayerEngagementUpdate_entry(
    lpunknown_t unk1, unknown_t unk2,
    pointer_t<X_NUI_PLAYER_ENGAGEMENT_UPDATE> engagement_data) {
  // always cleared before new data written
  std::memset(engagement_data, 0, sizeof(X_NUI_PLAYER_ENGAGEMENT_UPDATE));
}
DECLARE_XAM_EXPORT1(XamNuiPlayerEngagementUpdate, kNone, kStub);

}  // namespace xam
}  // namespace kernel
}  // namespace xe

DECLARE_XAM_EMPTY_REGISTER_EXPORTS(NUI);
