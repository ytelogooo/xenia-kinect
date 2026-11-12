/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "xenia/base/byte_order.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/utf8.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/ppc/ppc_frontend.h"
#include "xenia/cpu/processor.h"
#include "xenia/emulator.h"
#include "xenia/gpu/texture_address.h"
#include "xenia/gpu/xenos.h"
#include "xenia/hid/input_system.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_private.h"
#include "xenia/kernel/xboxkrnl/kinect_tuner.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_usb_kinect.h"
#include "xenia/kernel/xthread.h"
#include "xenia/memory.h"
#include "xenia/xbox.h"

#ifdef XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
// NOTE: must be included last as it expects windows.h to already be included.
#include <winsock2.h>  // NOLINT(build/include_order)
#include <ws2tcpip.h>  // NOLINT(build/include_order)
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

DECLARE_bool(kinect_connected);
DECLARE_bool(allow_kinect_camera);
DECLARE_string(kinect_usb_timestamps);
DEFINE_uint32(kinect_emulated_wall_mm, 3500,
              "Distance in millimetres of the back wall the emulated Kinect "
              "depth camera reports when no real sensor is used.",
              "Kernel");
DEFINE_bool(kinect_emulated_person, false,
            "Draw a synthetic standing person into the emulated Kinect depth "
            "frames.",
            "Kernel");
DEFINE_bool(kinect_emulated_person_motion, true,
            "Animate the synthetic person: the room is empty for 3 s, then "
            "the person walks in from the left and keeps swaying gently.",
            "Kernel");
DEFINE_uint32(kinect_emulated_person_mm, 2800,
              "Distance in millimetres of the synthetic person from the "
              "emulated Kinect.",
              "Kernel");
DEFINE_uint32(kinect_calib_shift, 50,
              "Emulated Kinect calibration: constant shift of the "
              "depth-to-colour registration, in units of raw disparity / 4 "
              "(PrimeSense's ConstShift of 200 raw units is 50 here).",
              "Kernel");
DEFINE_double(kinect_calib_scale, 0.1042,
              "Emulated Kinect calibration: scale of the depth-to-colour "
              "registration shift (13th calibration value). Unconfirmed; "
              "0.1042 is the PrimeSense reference pixel size.",
              "Kernel");
DEFINE_bool(kinect_emulated_skeleton, false,
            "Write a tracked skeleton matching the synthetic person straight "
            "into the skeleton frames the title reads, bypassing the NUI "
            "runtime's own tracker. Diagnostic; needs a known title hook.",
            "Kernel");
DEFINE_bool(kinect_skeleton_controller, true,
            "Let controller 1 move the emulated skeleton's arms: left/right "
            "stick moves the left/right hand, LB/RB raises it.",
            "Kernel");
DEFINE_string(kinect_skeleton_hook, "",
              "Override for kinect_emulated_skeleton: \"ADDR:REG\" (hex guest "
              "address, then the register number holding the output "
              "NUI_SKELETON_FRAME pointer there). Empty selects the built-in "
              "hooks (Kinect Adventures, Just Dance 2014/4, Dance Central 3).",
              "Kernel");
DEFINE_uint32(kinect_skeleton_udp_port, 0,
              "Receive live skeletons on this UDP port (127.0.0.1) from an "
              "external tracker such as tools/kinect_bridge (camera + "
              "MediaPipe). They replace the fake player and also shape the "
              "emulated depth image. 0 = off; the bridge defaults to 27050.",
              "Kernel");
DEFINE_uint32(kinect_skeleton_interpolation_ms, 33,
              "Give the title a skeleton interpolated between the bridge's "
              "last two, this many milliseconds in the past, so every frame "
              "it reads is a new position. The bridge and the runtime both "
              "run at about 30 Hz without being in step, so otherwise some "
              "frames repeat the previous skeleton, which makes hand-speed "
              "based input (Just Dance 4 menus) slow. 0 = latest skeleton.",
              "Kernel");
DEFINE_bool(kinect_debug_jd4_hand, false,
            "Diagnostic for Just Dance 4: log the right-hand position the "
            "game uses next to the one in the skeleton frame it received.",
            "Kernel");
DEFINE_bool(kinect_auto_hooks, true,
            "When a title loads, look for the Kinect runtime code the "
            "skeleton and player-image hooks need, so titles without a "
            "built-in hook get the bridges' skeletons and silhouettes too.",
            "HID");
DEFINE_bool(jd4_debug_print, false,
            "Diagnostic for Just Dance 4: log the game's own debug messages "
            "(its print function, 82211F20, is empty in the retail game), "
            "e.g. the [DLC] package registration messages.",
            "Kernel");
DEFINE_bool(kinect_debug_jd4_cursor, false,
            "Diagnostic for Just Dance 4: log the menu cursor object "
            "(0x120 bytes at *(0x82DE9B44) + 0x5DC) about 20 times a second, "
            "in the same format as tools/title_key_dumper/watch_memory.py "
            "reads it from a console.",
            "Kernel");
DEFINE_bool(kinect_debug_jd4_filter, false,
            "Diagnostic for Just Dance 4: every 5 s, log how often its "
            "skeleton filter is told a hand is unreliable (which clamps the "
            "hand to 2 m/s and 0.1 m/s^2 for 0.3 s) and the time step it "
            "uses.",
            "Kernel");
DEFINE_int32(kinect_jd4_hand_flag, -1,
             "Test for Just Dance 4: overwrite its depth hand tracker's "
             "per-hand flag with this value before the skeleton filter reads "
             "it (0 = treat both hands as reliable). -1 = leave it.",
             "Kernel");
DEFINE_bool(kinect_jd4_skeleton_hands, true,
            "Just Dance 4 with a Kinect bridge: give its skeleton filter the "
            "hands from the skeleton frame, marked reliable, instead of its "
            "depth/colour hand tracker's estimate (which the bridges' images "
            "make jumpy and flag as unreliable, so the cursor crawled). Not "
            "used with a real Kinect (allow_kinect_camera), where the game's "
            "own hand tracker works.",
            "Kernel");
DEFINE_string(kinect_jd2014_colour_mapping, "nominal",
              "Just Dance 2014 crops the player's face for the player slot "
              "from where NuiImageGetColorPixelCoordinatesFromDepthPixel puts "
              "the head in the colour image. With Xenia's guessed camera "
              "calibration that lands far outside the image (the slot showed "
              "static). \"nominal\" = replace it with a mapping from the "
              "nominal Kinect v1 camera intrinsics; \"runtime\" = keep the "
              "runtime's answer. "
              "Both log what the runtime returned now and then.",
              "Kernel");
DEFINE_string(kinect_jd_colour_mapping, "auto",
              "Just Dance 2015 and 2017 crop each player's face for the "
              "player slots themselves, from where "
              "NuiImageGetColorPixelCoordinatesFromDepthPixel puts the head "
              "in the colour image; with Xenia's guessed camera calibration "
              "that lands far outside the image. \"nominal\" = replace it "
              "with a mapping from the nominal Kinect v1 camera intrinsics "
              "(needs the whole camera image: the SDK bridge's "
              "--no-face-crop); \"runtime\" = keep the runtime's answer; "
              "\"auto\" = nominal while the SDK bridge sends the whole "
              "camera image, runtime otherwise.",
              "Kernel");
DEFINE_double(kinect_jd_face_offset_m, 0.08,
              "Just Dance 2015 / 2017 face crop (kinect_jd_colour_mapping): "
              "move it down by this many metres, so the face sits in the "
              "middle of the player slot. The SDK bridge overrides it live "
              "(W/S in its window).",
              "Kernel");
DEFINE_double(kinect_jd2014_face_offset_m, 0.08,
              "Just Dance 2014 face crop: move it down by this many metres "
              "(the head joint sits near the top of the head, so the crop "
              "centred on it showed the area above the head).",
              "Kernel");
DEFINE_double(kinect_sensor_tilt_deg, -100.0,
              "Test: write the frame's normal to gravity (+0x20) as for a "
              "sensor tilted down by this many degrees, (0, cos, sin, 0); a "
              "console with the sensor level reports about 11 for a Kinect "
              "on a TV stand. Titles build their body coordinates from it. "
              "-100 = leave what the runtime wrote.",
              "Kernel");
DEFINE_bool(kinect_player_index, true,
            "Fill the depth-and-player-index images titles copy (the player "
            "silhouette and position guide) from the emulated depth scene, "
            "marking the injected player as player 1. Needs a known title "
            "hook.",
            "Kernel");
DEFINE_bool(kinect_title_workarounds, true,
            "Work around Kinect titles relying on behaviour Xenia does not "
            "emulate yet (Dance Central 3 starts its main loop from a guest "
            "exception handler).",
            "Kernel");
DEFINE_bool(kinect_live_depth, true,
            "Use depth images with the player's real outline sent by the "
            "Kinect SDK bridge (XKDP packets) for player images, instead of "
            "a body drawn around the skeleton.",
            "Kernel");
DEFINE_bool(kinect_live_depth_mirror, false,
            "Mirror the bridge's depth images left-right.", "Kernel");
DEFINE_bool(kinect_live_colour, true,
            "Use the camera image sent by the Kinect SDK bridge (XKCL "
            "packets) for the colour stream, so titles that show the "
            "player's face (the Just Dance player slots) get a real picture "
            "instead of flat grey.",
            "Kernel");
DEFINE_bool(kinect_colour_test_pattern, false,
            "Diagnostic: write a grid of coloured 80x80 squares (black "
            "borders) into the title's colour image instead of the camera "
            "picture, to see how it reads the image.",
            "Kernel");
DEFINE_bool(kinect_colour_test_gradient, false,
            "Diagnostic: write a gradient into the title's colour image "
            "instead of the camera picture: red grows left to right, green "
            "top to bottom, blue on every other 64-pixel square, so a "
            "screenshot shows which part of the image the title displays.",
            "Kernel");
DEFINE_string(kinect_live_colour_layout, "auto",
              "How the colour image is written into the title's copy: "
              "\"auto\" (as the hook says; all known titles are linear), "
              "\"linear\" or \"tiled\".",
              "Kernel");
DEFINE_bool(kinect_debug_colour_texture, false,
            "Diagnostic for Just Dance 2014/2015: log the GPU texture the "
            "title's colour copy belongs to (size, pitch, tiling, format).",
            "Kernel");
DEFINE_bool(kinect_live_colour_mirror, false,
            "Mirror the bridge's colour images left-right.", "Kernel");
DEFINE_bool(kinect_auto_tilt, true,
            "Pass the title's Kinect tilt requests "
            "(XamNuiCameraElevationSetAngle, the auto-tilt at game start) to "
            "the Kinect SDK bridge, which moves the motor, and report the "
            "motor's real angle back (XamNuiCameraElevationGetAngle).",
            "Kernel");
DEFINE_string(kinect_image_depth_encoding, "auto",
              "How player images written into depth-only NUI frames (type 4, "
              "Dance Central 3) encode depth: \"shifted\" = depth << 3 | "
              "player, \"mm\" = plain millimetres (no player), \"auto\" = "
              "like the runtime's own data, shifted if it has none.",
              "Kernel");
DEFINE_string(kinect_depth_format, "mm",
              "Encoding of 16-bit depth pixels handed to titles: \"mm\" "
              "(millimetres), \"mm3\" (millimetres << 3) or \"raw\" (Kinect v1 "
              "11-bit disparity).",
              "Kernel");

namespace xe {
namespace kernel {
namespace xboxkrnl {

dword_result_t XUsbcamCreate_entry(dword_t buffer,
                                   dword_t buffer_size,  // 0x4B000 640x480?
                                   lpdword_t handle_out) {
  // This function should return success.
  // It looks like it only allocates space for usbcam support.
  // returning error code might cause games to initialize incorrectly.
  // "Carcassonne" initalization function checks for result from this
  // function. If value is different than 0 instead of loading
  // rest of the game it returns from initalization function and tries
  // to run game normally which causes crash, due to uninitialized data.
  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XUsbcamCreate, kNone, kStub);

enum class XUsbCamState : uint32_t {
  CamNotConnected = 0x00000000,
  CamInitilizied = 0x00000001,
  CamConnected = 0x00000002,
};

dword_result_t XUsbcamGetState_entry() {
  return static_cast<uint32_t>(XUsbCamState::CamNotConnected);
}
DECLARE_XBOXKRNL_EXPORT1(XUsbcamGetState, kNone, kStub);

// ---------------------------------------------------------------------------
// Kinect (PrimeSense) camera: PsCamDeviceRequest
//
// The NUI runtime statically linked into Kinect titles sends every camera
// request through PsCamDeviceRequest(request) (or XamXStudioRequest(0x1003)
// when XStudio playback is active). Layout, recovered from the NUI 2.0.11775
// runtime in Kinect Adventures (big-endian):
//   +0x00 u32  request code
//   +0x04 ...  code-specific input/output
//   +0x10 u32  completion context (for async requests)
//   +0x14 u32  completion routine: void cb(request, NTSTATUS status,
//              u32 bytes). Async requests return STATUS_PENDING (0x103) and
//              are completed later through it. Only valid for async
//              requests; synchronous callers leave stack garbage here.
//
// Codes seen:
//   0x00 query device state -> +0x04 u32 (4 = present, needs setup;
//        2 = set up and streaming; 3/5 = present, not ready)
//   0x02 stop/reset (sent when the state is 2/3/6): outstanding frame
//        requests must be cancelled so the runtime's drain completes
//   0x05 read frame (async): +0x18 stream (4 = depth, 2 = colour),
//        +0x1C buffer, +0x20 buffer size, +0x28 -> u32 frame number (out).
//        Depth completes with bytes = 0x96000 (640x480 u16) or 0x67200
//        (640x480 11-bit packed; then expanded with 0x1E).
//   0x0B calibration: +0x08 13 doubles, +0x70 u16[0x800] raw->mm table,
//        +0x1074 u16 shift constant (used when firmware < 5.1)
//   0x0D firmware version: +0x04 u8 major, +0x05 u8 minor
//   0x0F timing parameters (async, completed immediately)
//   0x18 read u16[+0x0C] table into +0x08 (optional; identity on failure)
//   0x19 14 timing values at +0x04 (microseconds; +0x24 must be non-zero)
//   0x1B set mode (+0x1C flags)
//   0x1E expand packed 11-bit depth (+0x04 src, +0x08 0x67200, +0x0C dst,
//        +0x10 -> u32 bytes produced, expected 0x96000)
//   0x21 enable (+0x1C = 1)
//   0x23 periodic measurement (async; +0x18 result). Left pending.
//
// With kinect_connected set, an emulated sensor answers these and delivers
// synthetic frames at 30 Hz; otherwise the camera is reported absent.
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t kStatusSuccess = 0x00000000;
constexpr uint32_t kStatusPending = 0x00000103;
constexpr uint32_t kStatusUnsuccessful = 0xC0000001;
constexpr uint32_t kStatusInvalidDeviceRequest = 0xC0000010;
constexpr uint32_t kStatusDeviceNotConnected = 0xC000009D;

constexpr uint32_t kStatusCancelled = 0xC0000120;

// Device states reported by request 0x00. 4: present, needs setup (the
// runtime then configures it with 0x1B and friends); 2: set up and
// streaming. The runtime re-queries the state whenever a frame looks wrong
// and only treats the device as ready if it reads 2; it sends 0x02 to stop.
constexpr uint32_t kCamStateStreaming = 2;
constexpr uint32_t kCamStateNeedsSetup = 4;
constexpr uint32_t kStreamDepth = 4;
constexpr uint32_t kDepthWidth = 640;
constexpr uint32_t kDepthHeight = 480;
constexpr uint32_t kDepthBytes = kDepthWidth * kDepthHeight * 2;  // 0x96000
constexpr uint32_t kFrameIntervalUs = 33333;

std::string HexDump(const uint8_t* bytes, uint32_t length) {
  static const char kHex[] = "0123456789ABCDEF";
  std::string dump;
  for (uint32_t i = 0; i < length; ++i) {
    if (i % 16 == 0) {
      dump += "\n  ";
    }
    dump += kHex[bytes[i] >> 4];
    dump += kHex[bytes[i] & 0xF];
    dump += ' ';
  }
  return dump;
}

// Kinect v1 raw disparity (11-bit) -> millimetres, from the widely used fit
// for the PrimeSense sensor. 0 means "no reading". Placeholder until the real
// per-device calibration can be read from a sensor.
uint16_t RawToMillimetres(uint32_t raw) {
  if (raw >= 2047) {
    return 0;
  }
  const double denom = -0.0030711016 * double(raw) + 3.3309495161;
  if (denom <= 0.0) {
    return 0;
  }
  const double mm = 1000.0 / denom;
  return (mm >= 400.0 && mm < 8191.0) ? uint16_t(mm) : 0;
}

// --- Synthetic depth scene -------------------------------------------------
// Camera space in millimetres: sensor at the origin, +x towards image right,
// +y up, +z away from the sensor. Nominal Kinect v1 depth intrinsics.
constexpr float kFx = 571.26f;
constexpr float kFy = 571.26f;
constexpr float kCx = 319.5f;
constexpr float kCy = 239.5f;
constexpr float kSensorHeightMm = 800.0f;

struct Vec3 {
  float x, y, z;
};

Vec3 Lerp(const Vec3& a, const Vec3& b, float t) {
  return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
}

// Z-buffered sphere; depth is the near surface along each pixel's ray
// (small-angle approximation, fine at Kinect distances).
void DrawSphere(std::vector<uint16_t>& depth, const Vec3& c, float radius) {
  if (c.z <= radius + 1.0f) {
    return;
  }
  const float u0 = kCx + kFx * c.x / c.z;
  const float v0 = kCy - kFy * c.y / c.z;
  const float pr = kFx * radius / c.z;
  const float mm_per_px = c.z / kFx;
  const int u_min = std::max(0, int(u0 - pr));
  const int u_max = std::min(int(kDepthWidth) - 1, int(u0 + pr) + 1);
  const int v_min = std::max(0, int(v0 - pr));
  const int v_max = std::min(int(kDepthHeight) - 1, int(v0 + pr) + 1);
  for (int v = v_min; v <= v_max; ++v) {
    for (int u = u_min; u <= u_max; ++u) {
      const float du = float(u) - u0;
      const float dv = float(v) - v0;
      const float d2 = (du * du + dv * dv) * mm_per_px * mm_per_px;
      if (d2 > radius * radius) {
        continue;
      }
      const float z = c.z - std::sqrt(radius * radius - d2);
      const uint16_t zz = uint16_t(std::max(1.0f, z));
      uint16_t& dst = depth[size_t(v) * kDepthWidth + size_t(u)];
      if (dst == 0 || zz < dst) {
        dst = zz;
      }
    }
  }
}

void DrawCapsule(std::vector<uint16_t>& depth, const Vec3& a, const Vec3& b,
                 float radius_a, float radius_b) {
  const float dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  const float length = std::sqrt(dx * dx + dy * dy + dz * dz);
  const int steps = std::max(2, int(length / 12.0f));
  for (int i = 0; i <= steps; ++i) {
    const float t = float(i) / float(steps);
    DrawSphere(depth, Lerp(a, b, t), radius_a + (radius_b - radius_a) * t);
  }
}

// Pose of the synthetic person for one frame.
struct PersonPose {
  bool visible = true;
  float x_mm = 0.0f;     // lateral offset of the body
  float stride = 0.0f;   // -1..1 leg/arm swing while walking
  float lean_mm = 0.0f;  // sideways sway of the upper body
};

PersonPose PoseAt(double t) {
  PersonPose pose;
  if (!cvars::kinect_emulated_person_motion) {
    return pose;
  }
  constexpr double kEnter = 3.0, kWalk = 3.0;
  if (t < kEnter) {
    pose.visible = false;
    return pose;
  }
  if (t < kEnter + kWalk) {
    const double u = (t - kEnter) / kWalk;  // 0..1
    pose.x_mm = float(-1400.0 * (1.0 - u));
    pose.stride = float(std::sin((t - kEnter) * 2.0 * 3.14159265 * 1.2));
    return pose;
  }
  const double s = t - kEnter - kWalk;
  pose.lean_mm = float(40.0 * std::sin(s * 2.0 * 3.14159265 / 2.5));
  pose.x_mm = float(30.0 * std::sin(s * 2.0 * 3.14159265 / 7.0));
  return pose;
}

// Camera-space point on the synthetic person: x across the body, height
// above the floor, dz towards/away from the sensor. The upper body leans
// sideways in proportion to its height.
Vec3 BodyPoint(const PersonPose& pose, uint32_t person_mm, float x,
               float height, float dz = 0.0f) {
  const float lean = pose.lean_mm * std::max(0.0f, height - 900.0f) / 800.0f;
  return Vec3{pose.x_mm + x + lean, -kSensorHeightMm + height,
              float(person_mm) + dz};
}

// NUI_SKELETON_POSITION_INDEX order. Left is the person's left, which is
// image left in the (mirrored) depth image, i.e. -x.
constexpr uint32_t kJointCount = 20;
void SkeletonJoints(const PersonPose& pose, uint32_t person_mm,
                    Vec3 joints[kJointCount]) {
  // Joints sit inside the body, a few centimetres behind the surface.
  constexpr float kInside = 60.0f;
  const float swing = pose.stride * 90.0f;
  auto at = [&](float x, float height, float dz = 0.0f) {
    return BodyPoint(pose, person_mm, x, height, dz + kInside);
  };
  joints[0] = at(0, 950);   // hip centre
  joints[1] = at(0, 1150);  // spine
  joints[2] = at(0, 1420);  // shoulder centre
  joints[3] = at(0, 1650);  // head
  for (int i = 0; i < 2; ++i) {
    const float side = i == 0 ? -1.0f : 1.0f;  // left, right
    const float arm = -side * swing * 0.6f;
    const float leg = side * swing;
    Vec3* a = joints + 4 + i * 4;  // shoulder, elbow, wrist, hand
    a[0] = at(side * 190, 1400);
    a[1] = at(side * 260, 1130, arm * 0.5f);
    a[2] = at(side * 290, 880, arm);
    a[3] = at(side * 300, 810, arm - 10);
    Vec3* l = joints + 12 + i * 4;  // hip, knee, ankle, foot
    l[0] = at(side * 100, 900);
    l[1] = at(side * 110, 480, leg * 0.5f);
    l[2] = at(side * 110, 70, leg);
    l[3] = at(side * 110, 40, leg - 120);
  }
}

// Millimetres per pixel: back wall, floor 0.8 m below the sensor, and
// optionally a person facing the sensor with arms slightly out.
std::vector<uint16_t> BuildDepthScene(uint32_t wall_mm, bool person,
                                      uint32_t person_mm,
                                      const PersonPose& pose = {}) {
  std::vector<uint16_t> depth(size_t(kDepthWidth) * kDepthHeight, 0);
  for (uint32_t v = 0; v < kDepthHeight; ++v) {
    const float ray_y = -(float(v) - kCy) / kFy;
    float z = float(wall_mm);
    if (ray_y < 0.0f) {
      z = std::min(z, kSensorHeightMm / -ray_y);  // floor
    }
    const uint16_t zz = z < 8000.0f ? uint16_t(z) : 0;
    for (uint32_t u = 0; u < kDepthWidth; ++u) {
      depth[size_t(v) * kDepthWidth + u] = zz;
    }
  }
  if (!person || !pose.visible) {
    return depth;
  }
  auto at = [&](float x, float height, float dz = 0.0f) {
    return BodyPoint(pose, person_mm, x, height, dz);
  };
  const float swing = pose.stride * 90.0f;  // mm, forward/back
  // Torso, head.
  DrawCapsule(depth, at(0, 950), at(0, 1420), 150, 165);
  DrawSphere(depth, at(0, 1650), 105);
  DrawCapsule(depth, at(0, 1440), at(0, 1540), 55, 55);  // neck
  for (float side : {-1.0f, 1.0f}) {
    // Arms swing opposite to the legs while walking.
    const float arm = -side * swing * 0.6f;
    const float leg = side * swing;
    // Arm: shoulder, elbow, wrist, hand.
    DrawCapsule(depth, at(side * 190, 1400), at(side * 260, 1130, arm * 0.5f),
                55, 45);
    DrawCapsule(depth, at(side * 260, 1130, arm * 0.5f),
                at(side * 290, 880, arm), 42, 35);
    DrawSphere(depth, at(side * 300, 810, arm - 10), 48);
    // Leg: hip, knee, ankle, foot.
    DrawCapsule(depth, at(side * 100, 900), at(side * 110, 480, leg * 0.5f),
                80, 60);
    DrawCapsule(depth, at(side * 110, 480, leg * 0.5f),
                at(side * 110, 70, leg), 55, 42);
    DrawCapsule(depth, at(side * 110, 40, leg),
                at(side * 110, 40, leg - 120), 40, 40);
  }
  return depth;
}

uint16_t EncodeDepth(uint16_t mm, const std::string& format) {
  if (format == "mm3") {
    return uint16_t(std::min<uint32_t>(mm, 0x1FFF) << 3);
  }
  if (format == "raw") {
    // Invert RawToMillimetres (monotonic over its valid range).
    static std::vector<uint16_t> mm_to_raw = [] {
      std::vector<uint16_t> table(8192, 2047);
      uint16_t last_mm = 0;
      for (uint32_t raw = 0; raw < 2047; ++raw) {
        const uint16_t m = RawToMillimetres(raw);
        if (m) {
          for (uint32_t i = last_mm; i <= m && i < table.size(); ++i) {
            table[i] = uint16_t(raw);
          }
          last_mm = m;
        }
      }
      return table;
    }();
    return mm ? mm_to_raw[std::min<uint32_t>(mm, 8191)] : 2047;
  }
  return mm;
}

// --- Live skeletons over UDP -----------------------------------------------
// Packets (little-endian):
// v1, 276 bytes, one player (tools/kinect_bridge, older SDK bridges):
//   char magic[4] = "XKSK"; u32 version = 1; u32 sequence; u32 tracked;
//   f32 joints[20][3]  (metres, Kinect skeleton space: +x = sensor's left,
//                       +y up, +z away from the sensor)
//   u8 joint_states[20] (0 = not tracked, 1 = inferred, 2 = tracked)
// v2, everyone the Kinect sees (tools/kinect_sdk_bridge):
//   char magic[4] = "XKSK"; u32 version = 2; u32 sequence; u32 count (0-6);
//   then count bodies of 280 bytes:
//     u32 tracking ID (never 0); u8 number (1-6: the skeleton slot + 1 and
//     the value marking this body in the XKDP outline); u8 state (1 =
//     position only, 2 = tracked: full skeleton); u8 primary (1 = the
//     player the single-player features follow); u8 0;
//     f32 position[3]; f32 joints[20][3]; u8 joint_states[20]
//   A Kinect v1 gives full skeletons for two people at a time; the others
//   are position only. Titles that pick who gets the full skeletons
//   (NuiSkeletonSetTrackedSkeletons) have their choice passed back to the
//   bridge as XKCH packets.

struct ExternalSkeleton {
  bool tracked = false;
  Vec3 joints[kJointCount] = {};  // millimetres
  uint8_t states[kJointCount] = {};
};

constexpr uint32_t kMaxBodies = 6;

struct ExternalBody {
  uint32_t tracking_id = 0;
  uint8_t number = 0;  // 1-6: skeleton slot + 1, its value in the outline
  uint8_t state = 0;   // 1 = position only, 2 = tracked
  Vec3 position = {};  // millimetres
  Vec3 joints[kJointCount] = {};  // millimetres (tracked only)
  uint8_t states[kJointCount] = {};
};

struct ExternalBodies {
  uint32_t count = 0;
  ExternalBody body[kMaxBodies];
};

class SkeletonReceiver {
 public:
  static SkeletonReceiver& Get() {
    // Leaked on purpose, like the camera: its thread runs until exit.
    static SkeletonReceiver* instance = new SkeletonReceiver();
    return *instance;
  }

  void Start(uint16_t port) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) {
      return;
    }
    started_ = true;
    std::thread([this, port]() { Run(port); }).detach();
  }

  bool started() {
    std::lock_guard<std::mutex> lock(mutex_);
    return started_;
  }

  // Number of skeleton packets received so far.
  uint32_t packet_count() const { return packet_count_.load(); }

  // Latest skeleton if it is newer than max_age_s.
  bool Latest(ExternalSkeleton* out, double max_age_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!have_ || std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - received_)
                          .count() > max_age_s) {
      return false;
    }
    *out = latest_;
    return true;
  }

  // The skeleton as it was delay_s ago, interpolated between the last two
  // received (joint states from the newer one); the latest one if that time
  // isn't between them. False like Latest.
  bool Interpolated(ExternalSkeleton* out, double max_age_s, double delay_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = std::chrono::steady_clock::now();
    if (!have_ ||
        std::chrono::duration<double>(now - received_).count() > max_age_s) {
      return false;
    }
    *out = latest_;
    if (!have_previous_ || !previous_.tracked || !latest_.tracked) {
      return true;
    }
    const double span =
        std::chrono::duration<double>(received_ - previous_received_).count();
    if (span <= 0.001 || span > 0.2) {
      return true;
    }
    const double at =
        std::chrono::duration<double>(now - previous_received_).count() -
        delay_s;
    const float t = float(std::clamp(at / span, 0.0, 1.0));
    for (uint32_t i = 0; i < kJointCount; ++i) {
      out->joints[i] = Lerp(previous_.joints[i], latest_.joints[i], t);
    }
    return true;
  }

  // Everyone the bridge sees, as it was delay_s ago (joints interpolated
  // like Interpolated, matched by tracking ID). False like Latest.
  bool InterpolatedBodies(ExternalBodies* out, double max_age_s,
                          double delay_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = std::chrono::steady_clock::now();
    if (!have_ ||
        std::chrono::duration<double>(now - received_).count() > max_age_s) {
      return false;
    }
    *out = bodies_;
    if (!have_previous_) {
      return true;
    }
    const double span =
        std::chrono::duration<double>(received_ - previous_received_).count();
    if (span <= 0.001 || span > 0.2) {
      return true;
    }
    const double at =
        std::chrono::duration<double>(now - previous_received_).count() -
        delay_s;
    const float t = float(std::clamp(at / span, 0.0, 1.0));
    for (uint32_t b = 0; b < out->count; ++b) {
      ExternalBody& body = out->body[b];
      for (uint32_t p = 0; p < previous_bodies_.count; ++p) {
        const ExternalBody& before = previous_bodies_.body[p];
        if (before.tracking_id != body.tracking_id) {
          continue;
        }
        body.position = Lerp(before.position, body.position, t);
        if (before.state == 2 && body.state == 2) {
          for (uint32_t i = 0; i < kJointCount; ++i) {
            body.joints[i] = Lerp(before.joints[i], body.joints[i], t);
          }
        }
        break;
      }
    }
    return true;
  }

  // Passes the title's choice of who gets the two full skeletons
  // (NuiSkeletonSetTrackedSkeletons; tracking IDs from our frames, 0 = none)
  // to the bridge. XKCH v1: "XKCH", u32 version 1, u32 id[2].
  bool ChooseSkeletons(uint32_t first, uint32_t second) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!have_bridge_address_) {
      return false;
    }
    uint8_t packet[16];
    std::memcpy(packet, "XKCH", 4);
    const uint32_t version = 1;
    std::memcpy(packet + 4, &version, 4);
    std::memcpy(packet + 8, &first, 4);
    std::memcpy(packet + 12, &second, 4);
    return sendto(socket_, reinterpret_cast<const char*>(packet),
                  int(sizeof(packet)), 0,
                  reinterpret_cast<const sockaddr*>(&bridge_address_),
                  int(sizeof(bridge_address_))) == int(sizeof(packet));
  }

  // For the player image log: depth frames received so far, the age of the
  // latest (seconds, -1 = none), and its outline (pixels with a player
  // number in the bridge's 320x240 image, bit n set = number n present).
  void ImageStats(uint32_t* frames, double* age_s, uint32_t* outline_pixels,
                  uint32_t* numbers) {
    std::lock_guard<std::mutex> lock(mutex_);
    *frames = image_frames_;
    *age_s = image_depth_.empty()
                 ? -1.0
                 : std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - image_received_)
                       .count();
    *outline_pixels = last_outline_pixels_;
    *numbers = last_outline_numbers_;
  }

  // The bridge's camera image settings (XKFO): whether it sends the whole
  // camera image (no face crop of its own) and the face offset in metres.
  // False if none arrived within max_age_s.
  bool FaceSettings(bool* whole_image, float* offset_m, double max_age_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!have_face_settings_ ||
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      face_settings_received_)
                .count() > max_age_s) {
      return false;
    }
    *whole_image = face_whole_image_;
    *offset_m = face_offset_m_;
    return true;
  }

  // Latest live depth image (640x480 mm + player mask) if it is newer than
  // max_age_s.
  bool LatestImage(std::vector<uint16_t>* depth_mm, std::vector<uint8_t>* mask,
                   double max_age_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (image_depth_.empty() ||
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      image_received_)
                .count() > max_age_s) {
      return false;
    }
    *depth_mm = image_depth_;
    *mask = image_mask_;
    return true;
  }

  // Latest colour image (B, G, R per pixel, top row first) if it is newer than
  // max_age_s. The image is shared, not copied.
  bool LatestColour(std::shared_ptr<const std::vector<uint8_t>>* bgr,
                    uint32_t* width, uint32_t* height, double max_age_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!colour_ ||
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      colour_received_)
                .count() > max_age_s) {
      return false;
    }
    *bgr = colour_;
    *width = colour_width_;
    *height = colour_height_;
    return true;
  }

  // Asks the Kinect SDK bridge to tilt the sensor to angle_deg (relative to
  // gravity, + = up). Sent from the listening socket to the address the
  // bridge's packets come from (its socket only accepts packets from there).
  // False if no bridge has been heard from yet.
  // XKTR v1: "XKTR", u32 version 1, i32 angle in degrees. Little-endian.
  bool RequestTilt(int32_t angle_deg) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!have_bridge_address_) {
      return false;
    }
    uint8_t packet[12];
    std::memcpy(packet, "XKTR", 4);
    const uint32_t version = 1;
    std::memcpy(packet + 4, &version, 4);
    std::memcpy(packet + 8, &angle_deg, 4);
    if (sendto(socket_, reinterpret_cast<const char*>(packet),
               int(sizeof(packet)), 0,
               reinterpret_cast<const sockaddr*>(&bridge_address_),
               int(sizeof(bridge_address_))) != int(sizeof(packet))) {
      return false;
    }
    // Report "moving" from now on, until the bridge has had time to answer:
    // the title checks the motor right after asking for a move.
    tilt_requested_ = std::chrono::steady_clock::now();
    if (have_tilt_) {
      tilt_moving_ = true;
    }
    return true;
  }

  // The motor's angle (degrees, + = up) and whether it is moving, as the
  // bridge last reported it (XKTS packets), if newer than max_age_s.
  // While the motor moves the bridge is busy with it and sends nothing, so a
  // "moving" status stays valid for longer (a full sweep takes a few seconds).
  bool TiltStatus(int32_t* angle_deg, bool* moving, double max_age_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    const double age = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - tilt_received_)
                           .count();
    if (!have_tilt_ || age > (tilt_moving_ ? std::max(max_age_s, 10.0)
                                           : max_age_s)) {
      return false;
    }
    *angle_deg = tilt_angle_;
    *moving = tilt_moving_;
    return true;
  }

 private:
  // XKTS v1: "XKTS", u32 version 1, i32 motor angle in degrees, u32 flags
  // (1 = moving or a move is queued), 3 floats accelerometer (gravity, in g).
  void ReceiveTiltStatus(const uint8_t* packet, int n) {
    if (n < 16) {
      return;
    }
    uint32_t version, flags;
    int32_t angle;
    std::memcpy(&version, packet + 4, 4);
    std::memcpy(&angle, packet + 8, 4);
    std::memcpy(&flags, packet + 12, 4);
    if (version != 1 || angle < -40 || angle > 40) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!have_tilt_ || tilt_angle_ != angle) {
      XELOGI("NUI tilt: the Kinect is at {} degrees{}", angle,
             (flags & 1) ? " (moving)" : "");
    }
    tilt_angle_ = angle;
    // A "not moving" sent before the bridge saw the latest request is stale.
    const bool request_in_flight =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      tilt_requested_)
            .count() < 0.7;
    tilt_moving_ = (flags & 1) != 0 || (request_in_flight && tilt_moving_);
    tilt_received_ = std::chrono::steady_clock::now();
    have_tilt_ = true;
  }

  // XKCL v1: "XKCL", u32 version 1, u32 frame, u16 width, u16 height, u16
  // first row, u16 rows, then rows * width * 3 bytes (B, G, R), rows top to
  // bottom, as the camera sees them (not mirrored). Little-endian header.
  void ReceiveColourRows(const uint8_t* packet, int n) {
    if (n < 20) {
      return;
    }
    uint32_t version;
    uint16_t width, height, first, rows;
    std::memcpy(&version, packet + 4, 4);
    std::memcpy(&width, packet + 12, 2);
    std::memcpy(&height, packet + 14, 2);
    std::memcpy(&first, packet + 16, 2);
    std::memcpy(&rows, packet + 18, 2);
    if (version != 1 || !width || !height || width > 640 || height > 480 ||
        uint32_t(first) + rows > height ||
        n < 20 + int(rows) * int(width) * 3) {
      return;
    }
    const size_t row_bytes = size_t(width) * 3;
    if (pending_colour_.size() != row_bytes * height) {
      pending_colour_.assign(row_bytes * height, 0);
    }
    std::memcpy(pending_colour_.data() + size_t(first) * row_bytes,
                packet + 20, size_t(rows) * row_bytes);
    if (uint32_t(first) + rows != height) {
      return;  // published with the last rows of the frame
    }
    std::shared_ptr<const std::vector<uint8_t>> colour =
        std::make_shared<std::vector<uint8_t>>(pending_colour_);
    std::lock_guard<std::mutex> lock(mutex_);
    colour_ = std::move(colour);
    colour_width_ = width;
    colour_height_ = height;
    colour_received_ = std::chrono::steady_clock::now();
    if (!logged_first_colour_) {
      logged_first_colour_ = true;
      XELOGI("NUI skeleton: first live colour image received ({}x{})", width,
             height);
    }
  }

  // XKDP v1: "XKDP", u32 version 1, u32 frame, u16 width, u16 height,
  // u16 first row, u16 rows, then rows * width u16 (depth mm << 3 | player
  // number: the body's number in the XKSK v2 packets, 1 with v1),
  // little-endian.
  void ReceiveDepthRows(const uint8_t* packet, int n) {
    if (n < 20) {
      return;
    }
    uint32_t version;
    uint16_t width, height, first, rows;
    std::memcpy(&version, packet + 4, 4);
    std::memcpy(&width, packet + 12, 2);
    std::memcpy(&height, packet + 14, 2);
    std::memcpy(&first, packet + 16, 2);
    std::memcpy(&rows, packet + 18, 2);
    if (version != 1 || !width || !height || width > 640 || height > 480 ||
        uint32_t(first) + rows > height ||
        n < 20 + int(rows) * width * 2) {
      return;
    }
    if (pending_raw_.size() != size_t(width) * height) {
      pending_raw_.assign(size_t(width) * height, 0);
    }
    std::memcpy(pending_raw_.data() + size_t(first) * width, packet + 20,
                size_t(rows) * width * 2);
    if (uint32_t(first) + rows != height) {
      return;  // published with the last rows of the frame
    }
    std::vector<uint16_t> depth(size_t(kDepthWidth) * kDepthHeight);
    std::vector<uint8_t> mask(depth.size());
    const bool mirror = cvars::kinect_live_depth_mirror;
    for (uint32_t y = 0; y < kDepthHeight; ++y) {
      const uint32_t sy = y * height / kDepthHeight;
      for (uint32_t x = 0; x < kDepthWidth; ++x) {
        uint32_t sx = x * width / kDepthWidth;
        if (mirror) {
          sx = width - 1 - sx;
        }
        const uint16_t raw = pending_raw_[size_t(sy) * width + sx];
        depth[size_t(y) * kDepthWidth + x] = raw >> 3;
        mask[size_t(y) * kDepthWidth + x] = uint8_t(raw & 7);
      }
    }
    uint32_t outline_pixels = 0;
    uint32_t numbers = 0;
    for (uint16_t raw : pending_raw_) {
      if (raw & 7) {
        ++outline_pixels;
        numbers |= 1u << (raw & 7);
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    image_depth_ = std::move(depth);
    image_mask_ = std::move(mask);
    image_received_ = std::chrono::steady_clock::now();
    ++image_frames_;
    last_outline_pixels_ = outline_pixels;
    last_outline_numbers_ = numbers;
    if (!logged_first_image_) {
      logged_first_image_ = true;
      XELOGI("NUI skeleton: first live depth image received ({}x{})", width,
             height);
    }
  }

  static void ReadJoints(const uint8_t* p, ExternalBody* body) {
    for (uint32_t i = 0; i < kJointCount; ++i) {
      float xyz[3];
      std::memcpy(xyz, p + i * 12, 12);
      body->joints[i] = {xyz[0] * 1000.0f, xyz[1] * 1000.0f,
                         xyz[2] * 1000.0f};
      body->states[i] = p[kJointCount * 12 + i];
    }
  }

  // XKSK v1 or v2 into the primary player's skeleton and the body list.
  static bool ParseSkeletonPacket(const uint8_t* packet, int n,
                                  ExternalSkeleton* skeleton,
                                  ExternalBodies* bodies) {
    uint32_t version, value;
    std::memcpy(&version, packet + 4, 4);
    std::memcpy(&value, packet + 12, 4);
    if (version == 1) {
      if (n < 276) {
        return false;
      }
      ExternalBody body;
      ReadJoints(packet + 16, &body);
      skeleton->tracked = value != 0;
      std::memcpy(skeleton->joints, body.joints, sizeof(body.joints));
      std::memcpy(skeleton->states, body.states, sizeof(body.states));
      if (skeleton->tracked) {
        body.tracking_id = 1;
        body.number = 1;
        body.state = 2;
        body.position = body.joints[0];
        bodies->body[0] = body;
        bodies->count = 1;
      }
      return true;
    }
    constexpr int kBodySize = 4 + 4 + 12 + kJointCount * 13;  // 280
    if (version != 2 || value > kMaxBodies || n < 16 + int(value) * kBodySize) {
      return false;
    }
    for (uint32_t b = 0; b < value; ++b) {
      const uint8_t* p = packet + 16 + b * kBodySize;
      ExternalBody body;
      std::memcpy(&body.tracking_id, p, 4);
      body.number = p[4];
      body.state = p[5];
      const bool primary = p[6] != 0;
      if (!body.tracking_id || body.number < 1 || body.number > kMaxBodies ||
          (body.state != 1 && body.state != 2)) {
        continue;
      }
      float xyz[3];
      std::memcpy(xyz, p + 8, 12);
      body.position = {xyz[0] * 1000.0f, xyz[1] * 1000.0f, xyz[2] * 1000.0f};
      if (body.state == 2) {
        ReadJoints(p + 20, &body);
      }
      if (primary && body.state == 2) {
        skeleton->tracked = true;
        std::memcpy(skeleton->joints, body.joints, sizeof(body.joints));
        std::memcpy(skeleton->states, body.states, sizeof(body.states));
      }
      bodies->body[bodies->count++] = body;
    }
    return true;
  }

  void Run(uint16_t port) {
#ifdef XE_PLATFORM_WIN32
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
    using socket_t = SOCKET;
    const socket_t kBad = INVALID_SOCKET;
#else
    using socket_t = int;
    const socket_t kBad = -1;
#endif
    socket_t sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == kBad) {
      XELOGE("NUI skeleton: could not create UDP socket");
      return;
    }
    // Depth and colour images arrive as several large datagrams per frame.
    int receive_buffer = 8 * 1024 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF,
               reinterpret_cast<const char*>(&receive_buffer),
               sizeof(receive_buffer));
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(sock, reinterpret_cast<sockaddr*>(&address), sizeof(address)) !=
        0) {
      XELOGE("NUI skeleton: could not listen on UDP port {} (in use?)", port);
      return;
    }
    XELOGI("NUI skeleton: listening for live skeletons on 127.0.0.1:{}",
           port);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      socket_ = sock;
    }

    std::vector<uint8_t> buffer(65536);
    uint8_t* packet = buffer.data();
    uint32_t received = 0;
    while (true) {
      sockaddr_in from = {};
#ifdef XE_PLATFORM_WIN32
      int from_length = int(sizeof(from));
#else
      socklen_t from_length = sizeof(from);
#endif
      const int n = int(recvfrom(sock, reinterpret_cast<char*>(packet),
                                 int(buffer.size()), 0,
                                 reinterpret_cast<sockaddr*>(&from),
                                 &from_length));
      if (n >= 4 && from.sin_family == AF_INET &&
          (std::memcmp(packet, "XKSK", 4) == 0 ||
           std::memcmp(packet, "XKTS", 4) == 0)) {
        // Where tilt requests go back to.
        std::lock_guard<std::mutex> lock(mutex_);
        bridge_address_ = from;
        have_bridge_address_ = true;
      }
      if (n >= 4 && std::memcmp(packet, "XKTS", 4) == 0) {
        ReceiveTiltStatus(packet, n);
        continue;
      }
      if (n >= 4 && std::memcmp(packet, "XKCL", 4) == 0) {
        ReceiveColourRows(packet, n);
        continue;
      }
      if (n >= 16 && std::memcmp(packet, "XKFO", 4) == 0) {
        // XKFO v1: "XKFO", u32 version 1, u32 flags (1 = the whole camera
        // image, no face crop), f32 face offset in metres.
        uint32_t version, flags;
        float offset;
        std::memcpy(&version, packet + 4, 4);
        std::memcpy(&flags, packet + 8, 4);
        std::memcpy(&offset, packet + 12, 4);
        if (version == 1 && offset > -0.5f && offset < 0.5f) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (!have_face_settings_ || face_offset_m_ != offset ||
              face_whole_image_ != ((flags & 1) != 0)) {
            XELOGI("NUI skeleton: the bridge sends {}, face offset {:.2f} m",
                   (flags & 1) ? "the whole camera image"
                               : "a face crop of the main player",
                   offset);
          }
          face_whole_image_ = (flags & 1) != 0;
          face_offset_m_ = offset;
          face_settings_received_ = std::chrono::steady_clock::now();
          have_face_settings_ = true;
        }
        continue;
      }
      if (n >= 4 && std::memcmp(packet, "XKDP", 4) == 0) {
        ReceiveDepthRows(packet, n);
        continue;
      }
      if (n < 16 || std::memcmp(packet, "XKSK", 4) != 0) {
        continue;
      }
      ExternalSkeleton skeleton;
      ExternalBodies bodies;
      if (!ParseSkeletonPacket(packet, n, &skeleton, &bodies)) {
        continue;
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (have_) {
          previous_ = latest_;
          previous_bodies_ = bodies_;
          previous_received_ = received_;
          have_previous_ = true;
        }
        latest_ = skeleton;
        bodies_ = bodies;
        received_ = std::chrono::steady_clock::now();
        have_ = true;
        if (bodies.count > max_bodies_seen_) {
          max_bodies_seen_ = bodies.count;
          XELOGI("NUI skeleton: the bridge sees {} people", bodies.count);
        }
      }
      packet_count_.fetch_add(1);
      if (++received == 1) {
        XELOGI("NUI skeleton: first live skeleton received");
      }
    }
  }

  std::mutex mutex_;
  bool started_ = false;
  bool have_ = false;
  std::atomic<uint32_t> packet_count_{0};
  ExternalSkeleton latest_;
  ExternalSkeleton previous_;
  ExternalBodies bodies_;
  ExternalBodies previous_bodies_;
  uint32_t image_frames_ = 0;
  uint32_t last_outline_pixels_ = 0;
  uint32_t last_outline_numbers_ = 0;
  bool have_face_settings_ = false;
  bool face_whole_image_ = false;
  float face_offset_m_ = 0.0f;
  std::chrono::steady_clock::time_point face_settings_received_;
  uint32_t max_bodies_seen_ = 1;
  std::chrono::steady_clock::time_point previous_received_;
  bool have_previous_ = false;
  std::chrono::steady_clock::time_point received_;
  std::vector<uint16_t> pending_raw_;  // receive thread only
  std::vector<uint16_t> image_depth_;
  std::vector<uint8_t> image_mask_;
  std::chrono::steady_clock::time_point image_received_;
  bool logged_first_image_ = false;
  std::vector<uint8_t> pending_colour_;  // receive thread only
  std::shared_ptr<const std::vector<uint8_t>> colour_;
  uint32_t colour_width_ = 0;
  uint32_t colour_height_ = 0;
  std::chrono::steady_clock::time_point colour_received_;
  bool logged_first_colour_ = false;
#ifdef XE_PLATFORM_WIN32
  SOCKET socket_ = INVALID_SOCKET;
#else
  int socket_ = -1;
#endif
  sockaddr_in bridge_address_ = {};
  bool have_bridge_address_ = false;
  int32_t tilt_angle_ = 0;
  bool tilt_moving_ = false;
  bool have_tilt_ = false;
  std::chrono::steady_clock::time_point tilt_received_;
  std::chrono::steady_clock::time_point tilt_requested_;
};

constexpr double kLiveSkeletonMaxAge = 0.25;  // seconds

// Draws a body around a skeleton (millimetres) into a depth image.
void DrawBodyFromJoints(std::vector<uint16_t>& depth,
                        const Vec3 joints[kJointCount]) {
  struct Bone {
    int a, b;
    float ra, rb;
  };
  static const Bone kBones[] = {
      {0, 1, 150, 150},  {1, 2, 150, 160},                     // torso
      {2, 3, 55, 55},                                          // neck
      {2, 4, 60, 55},    {4, 5, 55, 45},   {5, 6, 42, 35},     // left arm
      {6, 7, 40, 45},    {2, 8, 60, 55},   {8, 9, 55, 45},     // right arm
      {9, 10, 42, 35},   {10, 11, 40, 45}, {0, 12, 90, 80},    // left leg
      {12, 13, 80, 60},  {13, 14, 55, 42}, {14, 15, 40, 40},   //
      {0, 16, 90, 80},   {16, 17, 80, 60}, {17, 18, 55, 42},   // right leg
      {18, 19, 40, 40},
  };
  for (const Bone& bone : kBones) {
    DrawCapsule(depth, joints[bone.a], joints[bone.b], bone.ra, bone.rb);
  }
  DrawSphere(depth, joints[3], 105);  // head
}

class EmulatedNuiCamera {
 public:
  static EmulatedNuiCamera& Get() {
    // Leaked on purpose: the frame thread may outlive static destruction.
    static EmulatedNuiCamera* instance = new EmulatedNuiCamera();
    return *instance;
  }

  uint32_t HandleRequest(cpu::ppc::PPCContext* ctx, uint32_t req) {
    uint8_t* r = ctx->TranslateVirtual(req);
    const uint32_t code = xe::load_and_swap<uint32_t>(r + 0x00);

    switch (code) {
      case 0x00:  // device state
        xe::store_and_swap<uint32_t>(r + 0x04, state_.load());
        return kStatusSuccess;

      case 0x1B:  // set mode: the device is now configured and streaming
        SetState(kCamStateStreaming);
        return kStatusSuccess;

      case 0x02: {  // stop/reset: cancel outstanding frames from the thread
        SetState(kCamStateNeedsSetup);
        std::lock_guard<std::mutex> lock(mutex_);
        cancel_pending_ = true;
        return kStatusSuccess;
      }

      case 0x21:
        return kStatusSuccess;

      case 0x05: {  // read frame (async)
        const uint32_t stream = xe::load_and_swap<uint32_t>(r + 0x18);
        {
          std::lock_guard<std::mutex> lock(mutex_);
          pending_frames_.push_back(req);
        }
        if (!logged_first_frame_request_) {
          logged_first_frame_request_ = true;
          XELOGI("NUI camera: first frame request {:08X} (stream {}, {:08X} "
                 "bytes)",
                 req, stream, xe::load_and_swap<uint32_t>(r + 0x20));
        }
        EnsureFrameThread();
        return kStatusPending;
      }

      case 0x0B:
        FillCalibration(r);
        return kStatusSuccess;

      case 0x0D:  // firmware version: report < 5.1 to use calibration 0x0B
        r[0x04] = 1;
        r[0x05] = 0;
        return kStatusSuccess;

      case 0x0F: {  // timing parameters (async, complete immediately)
        const uint32_t callback = xe::load_and_swap<uint32_t>(r + 0x14);
        if (callback) {
          uint64_t args[] = {req, kStatusSuccess, 0};
          ctx->processor->Execute(ctx->thread_state, callback, args, 3);
        }
        return kStatusSuccess;
      }

      case 0x18:  // optional table: let the runtime use its identity default
        return kStatusUnsuccessful;

      case 0x19:  // timing values, microseconds
        for (uint32_t i = 0; i < 14; ++i) {
          xe::store_and_swap<uint32_t>(r + 0x04 + i * 4, 0);
        }
        xe::store_and_swap<uint32_t>(r + 0x04, kFrameIntervalUs);
        xe::store_and_swap<uint32_t>(r + 0x18, kFrameIntervalUs);
        xe::store_and_swap<uint32_t>(r + 0x24, kFrameIntervalUs);
        return kStatusSuccess;

      case 0x1E:  // expand packed depth; never needed (frames are unpacked)
        return kStatusInvalidDeviceRequest;

      case 0x23:  // periodic measurement: leave pending
        return kStatusPending;

      case 0x24:  // per-component state query: mode nibble at +0x1C and
                  // four values at +0x4C; report all zero.
        std::memset(r + 0x18, 0, 0x60 - 0x18);
        return kStatusSuccess;

      default:
        XELOGW("NUI camera: unhandled request code {:02X} at {:08X}:{}",
               code, req, HexDump(r, 0x40));
        return kStatusInvalidDeviceRequest;
    }
  }

  // Latest emulated depth (mm, 640x480) and player mask (0 = nobody, else
  // the player's number: skeleton slot + 1).
  bool GetPlayerImage(std::vector<uint16_t>* depth_mm,
                      std::vector<uint8_t>* mask) {
    // The bridge's real depth and player outline, when it sends them.
    if (cvars::kinect_live_depth && cvars::kinect_skeleton_udp_port &&
        SkeletonReceiver::Get().LatestImage(depth_mm, mask,
                                            kLiveSkeletonMaxAge)) {
      CountPlayerImage(true, *mask);
      return true;
    }
    std::lock_guard<std::mutex> lock(image_mutex_);
    if (image_depth_.empty()) {
      return false;
    }
    *depth_mm = image_depth_;
    *mask = image_mask_;
    CountPlayerImage(false, *mask);
    return true;
  }

  // Every 5 s: where the player images came from (the bridge's live image
  // or the emulated scene, when the live one was older than 0.25 s) and how
  // much player outline they had.
  void CountPlayerImage(bool live, const std::vector<uint8_t>& mask) {
    if (!cvars::kinect_skeleton_udp_port) {
      return;
    }
    uint32_t pixels = 0;
    for (size_t i = 0; i < mask.size(); i += 4) {
      pixels += mask[i] != 0;
    }
    static std::mutex stats_mutex;
    static auto start = std::chrono::steady_clock::now();
    static uint32_t live_calls = 0, emulated_calls = 0, empty_calls = 0;
    static uint64_t pixel_sum = 0;
    std::lock_guard<std::mutex> lock(stats_mutex);
    ++(live ? live_calls : emulated_calls);
    empty_calls += pixels == 0;
    pixel_sum += pixels;
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - start).count();
    if (seconds < 5.0) {
      return;
    }
    uint32_t frames, outline_pixels, numbers;
    double age;
    SkeletonReceiver::Get().ImageStats(&frames, &age, &outline_pixels,
                                       &numbers);
    const uint32_t calls = live_calls + emulated_calls;
    XELOGI("NUI player image stats: {} requests ({} live, {} emulated), {} "
           "with an empty outline, {} outline pixels on average (of 76800); "
           "bridge depth frames {}, latest {:.2f} s old with {} outline "
           "pixels, numbers {:02X}",
           calls, live_calls, emulated_calls, empty_calls,
           calls ? pixel_sum / calls : 0, frames, age, outline_pixels,
           numbers);
    start = now;
    live_calls = emulated_calls = empty_calls = 0;
    pixel_sum = 0;
  }

  // Frame timestamp in milliseconds (since the camera started streaming;
  // never 0, which titles treat as "no timestamp").
  uint64_t TimestampMs() const {
    const double t = SceneTime();
    return t < 0.0 ? 1 : uint64_t(t * 1000.0) + 1;
  }

  // Seconds since the emulated depth stream started, or < 0 before that.
  double SceneTime() const {
    const int64_t start = scene_start_ns_.load();
    if (!start) {
      return -1.0;
    }
    const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    return double(now - start) * 1e-9;
  }

 private:
  void SetState(uint32_t state) {
    const uint32_t old = state_.exchange(state);
    if (old != state) {
      XELOGI("NUI camera: device state {} -> {}", old, state);
    }
  }

  void FillCalibration(uint8_t* r) {
    // TODO: real values. The 13 doubles feed the tracker's depth/colour
    // camera models; these are the nominal Kinect v1 intrinsics.
    const double values[13] = {
        571.26, 571.26, 319.5, 239.5, 0.0, 0.0,  // depth (guess)
        531.15, 531.15, 319.5, 239.5, 0.0, 0.0,  // colour (guess)
        cvars::kinect_calib_scale,               // disparity table scale
    };
    for (int i = 0; i < 13; ++i) {
      xe::store_and_swap<double>(r + 0x08 + i * 8, values[i]);
    }
    for (uint32_t i = 0; i < 0x800; ++i) {
      xe::store_and_swap<uint16_t>(r + 0x70 + i * 2, RawToMillimetres(i));
    }
    // The runtime builds its mm -> disparity table as
    //   16 * scale * (raw / 4 - shift)
    // i.e. the PrimeSense "fixed reference X" disparity relative to the
    // reference plane, so the shift must be the sensor's constant shift.
    xe::store_and_swap<uint16_t>(r + 0x1074,
                                 uint16_t(cvars::kinect_calib_shift));
    XELOGI("NUI camera: calibration sent (shift {}, scale {})",
           cvars::kinect_calib_shift, cvars::kinect_calib_scale);
  }

  void EnsureFrameThread() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frame_thread_) {
      return;
    }
    auto* state = kernel_state();
    frame_thread_ = object_ref<XHostThread>(new XHostThread(
        state, 128 * 1024, 0,
        [this]() {
          FrameThreadMain();
          return 0;
        },
        state->GetTitleProcess()));
    // Runs guest code (completion routines), so the debugger may suspend it.
    frame_thread_->set_can_debugger_suspend(true);
    frame_thread_->set_name("NUI Camera");
    frame_thread_->Create();
  }

  void FrameThreadMain() {
    auto* state = kernel_state();
    auto* memory = state->memory();
    auto* processor = state->processor();
    const std::string format = cvars::kinect_depth_format;
    const bool live_skeleton = cvars::kinect_skeleton_udp_port != 0;
    const bool animate =
        live_skeleton ||
        (cvars::kinect_emulated_person && cvars::kinect_emulated_person_motion);
    std::vector<xe::be<uint16_t>> encoded(size_t(kDepthWidth) * kDepthHeight);
    auto render = [&](double t) {
      std::vector<uint16_t> scene = BuildDepthScene(
          cvars::kinect_emulated_wall_mm,
          cvars::kinect_emulated_person && !live_skeleton,
          cvars::kinect_emulated_person_mm, PoseAt(t));
      ExternalSkeleton live;
      if (live_skeleton &&
          SkeletonReceiver::Get().Latest(&live, kLiveSkeletonMaxAge) &&
          live.tracked) {
        DrawBodyFromJoints(scene, live.joints);
      }
      for (size_t i = 0; i < scene.size(); ++i) {
        encoded[i] = EncodeDepth(scene[i], format);
      }
      // Player mask: wherever the person/body changed the empty room.
      if (background_.empty()) {
        background_ = BuildDepthScene(cvars::kinect_emulated_wall_mm, false,
                                      0);
      }
      std::vector<uint8_t> mask(scene.size());
      for (size_t i = 0; i < scene.size(); ++i) {
        mask[i] = scene[i] != background_[i];
      }
      std::lock_guard<std::mutex> lock(image_mutex_);
      image_depth_ = std::move(scene);
      image_mask_ = std::move(mask);
    };
    render(animate ? 0.0 : 1e9);
    XELOGI("NUI camera: emulated depth scene ready (wall {} mm, person {}, "
           "format {})",
           cvars::kinect_emulated_wall_mm,
           cvars::kinect_emulated_person
               ? std::to_string(cvars::kinect_emulated_person_mm) + " mm" +
                     (animate ? ", walking in after 3 s" : ", static")
               : std::string("none"),
           format);
    const auto start = std::chrono::steady_clock::now();
    scene_start_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          start.time_since_epoch())
                          .count();

    auto next = std::chrono::steady_clock::now();
    uint32_t delivered = 0;
    while (true) {
      next += std::chrono::microseconds(kFrameIntervalUs);
      std::this_thread::sleep_until(next);
      ++frame_number_;

      // Complete one outstanding request per stream per tick, or all of
      // them with STATUS_CANCELLED after a stop (0x02).
      std::vector<uint32_t> batch;
      bool cancel = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cancel_pending_) {
          cancel_pending_ = false;
          cancel = true;
          batch.swap(pending_frames_);
        }
        std::vector<uint32_t> streams_done;
        for (auto it = pending_frames_.begin();
             !cancel && it != pending_frames_.end();) {
          const uint32_t stream = xe::load_and_swap<uint32_t>(
              memory->TranslateVirtual(*it + 0x18));
          if (std::find(streams_done.begin(), streams_done.end(), stream) ==
              streams_done.end()) {
            streams_done.push_back(stream);
            batch.push_back(*it);
            it = pending_frames_.erase(it);
          } else {
            ++it;
          }
        }
      }

      if (cancel && !batch.empty()) {
        XELOGI("NUI camera: cancelling {} outstanding frame requests",
               batch.size());
      }
      for (uint32_t req : batch) {
        uint8_t* r = memory->TranslateVirtual(req);
        if (cancel) {
          const uint32_t cancel_callback =
              xe::load_and_swap<uint32_t>(r + 0x14);
          if (cancel_callback) {
            uint64_t args[] = {req, kStatusCancelled, 0};
            processor->Execute(frame_thread_->thread_state(), cancel_callback,
                               args, 3);
          }
          continue;
        }
        const uint32_t stream = xe::load_and_swap<uint32_t>(r + 0x18);
        const uint32_t buffer = xe::load_and_swap<uint32_t>(r + 0x1C);
        const uint32_t size = xe::load_and_swap<uint32_t>(r + 0x20);
        const uint32_t info = xe::load_and_swap<uint32_t>(r + 0x28);
        const uint32_t callback = xe::load_and_swap<uint32_t>(r + 0x14);
        uint32_t bytes = 0;

        if (stream == kStreamDepth && buffer && size >= kDepthBytes) {
          if (animate) {
            render(std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count());
          }
          std::memcpy(memory->TranslateVirtual(buffer), encoded.data(),
                      kDepthBytes);
          bytes = kDepthBytes;
          if (++delivered % 300 == 1) {
            XELOGI("NUI camera: {} depth frames delivered", delivered);
          }
        } else if (buffer && size) {
          std::memset(memory->TranslateVirtual(buffer), 0x80, size);
          bytes = size;
        }
        if (info) {
          xe::store_and_swap<uint32_t>(memory->TranslateVirtual(info),
                                       frame_number_);
        }
        if (callback) {
          uint64_t args[] = {req, kStatusSuccess, bytes};
          processor->Execute(frame_thread_->thread_state(), callback, args,
                             3);
        }
      }
    }
  }

  std::mutex mutex_;
  std::vector<uint32_t> pending_frames_;
  std::atomic<uint32_t> state_{kCamStateNeedsSetup};
  bool cancel_pending_ = false;
  object_ref<XHostThread> frame_thread_;
  uint32_t frame_number_ = 0;
  bool logged_first_frame_request_ = false;
  std::atomic<int64_t> scene_start_ns_{0};
  std::vector<uint16_t> background_;
  std::mutex image_mutex_;
  std::vector<uint16_t> image_depth_;
  std::vector<uint8_t> image_mask_;
};

// --- Real Kinect ------------------------------------------------------------
// With a real Kinect (allow_kinect_camera, xboxkrnl_usb_kinect.cc) the
// title's own tracker fills skeleton frames and player indices, so the hooks
// below only give those frames timestamps (kinect_usb_timestamps) and leave
// their contents alone.

// True when the hooks should inject the emulated/live skeleton and player
// image (kinect_connected with kinect_skeleton_udp_port or
// kinect_emulated_skeleton), as before real camera support.
bool skeleton_injection_enabled = false;

void StampRealCameraFrame(cpu::ppc::PPCContext_s* ctx, uint32_t frame,
                          const char* kind) {
  const auto mode = usb_kinect::GetTimestampMode();
  uint8_t* p = ctx->TranslateVirtual(frame);
  const uint64_t runtime_time = xe::load_and_swap<uint64_t>(p);
  static std::atomic<uint32_t> logged{0};
  if (logged.fetch_add(1) < 4) {
    XELOGI("Kinect: {} frame {:08X} from the title's tracker, timestamp {} "
           "(kinect_usb_timestamps mode {})",
           kind, frame, runtime_time, cvars::kinect_usb_timestamps);
  }
  if (mode == usb_kinect::TimestampMode::kNever ||
      (mode == usb_kinect::TimestampMode::kAuto && runtime_time != 0)) {
    return;
  }
  const uint64_t ms = usb_kinect::TimestampMs();
  if (ms) {
    xe::store_and_swap<uint64_t>(p, ms);
  }
}

// Logs what the title's own tracker found, so a log shows whether the real
// camera path works.
void LogRealCameraSkeletons(cpu::ppc::PPCContext_s* ctx, uint32_t frame) {
  static std::atomic<uint32_t> frames{0};
  static std::atomic<int> last_tracked{-1};
  const uint32_t n = ++frames;
  const uint8_t* f = ctx->TranslateVirtual(frame);
  int tracked = 0;
  int position_only = 0;
  int first = -1;
  for (int i = 0; i < 6; ++i) {
    const uint32_t state = xe::load_and_swap<uint32_t>(f + 0x30 + i * 0x1C0);
    if (state == 2) {
      ++tracked;
      if (first < 0) {
        first = i;
      }
    } else if (state == 1) {
      ++position_only;
    }
  }
  // Logged when either count changes, and every 300 frames.
  const int counts = tracked | position_only << 8;
  if (counts == last_tracked.exchange(counts) && n != 1 && n % 300 != 0) {
    return;
  }
  if (first < 0) {
    XELOGI("Kinect: tracker frame #{} (frame number {}): nobody tracked, {} "
           "seen (position only)",
           n, xe::load_and_swap<uint32_t>(f + 0x08), position_only);
    return;
  }
  const uint8_t* s = f + 0x30 + first * 0x1C0;
  auto joint = [&](int j, int axis) {
    return xe::load_and_swap<float>(s + 0x20 + j * 0x10 + axis * 4);
  };
  XELOGI("Kinect: tracker frame #{} (frame number {}): {} tracked ({} more "
         "seen), slot {} id {} hip ({:.2f}, {:.2f}, {:.2f}) m, right hand ({:.2f}, {:.2f}, "
         "{:.2f}) m",
         n, xe::load_and_swap<uint32_t>(f + 0x08), tracked, position_only,
         first, xe::load_and_swap<uint32_t>(s + 0x04), joint(0, 0), joint(0, 1),
         joint(0, 2), joint(11, 0), joint(11, 1), joint(11, 2));
}

// --- Player-index images ------------------------------------------------------
// Titles show the player's silhouette (and judge whether they stand in the
// play space) from the depth-and-player-index image stream: 16-bit pixels,
// (depth_mm << 3) | player_index. The NUI runtime fills the player index from
// its own tracker, which doesn't find anyone under emulation, so the image
// is empty. Right after the title copies a frame out of the stream's
// texture, rewrite its copy from the emulated depth scene and mark the
// injected player as player 1.

// A value found at the hook: register + offset, optionally read from memory.
struct GuestValue {
  uint8_t reg;
  int32_t offset;
  bool load;
  uint32_t Get(cpu::ppc::PPCContext_s* ctx) const {
    const uint32_t address = uint32_t(ctx->r[reg]) + uint32_t(offset);
    return load ? xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(address))
                : address;
  }
};

struct PlayerImageHook {
  const char* title;
  uint32_t address;            // instruction after the title's memcpy
  uint32_t expected_previous;  // that memcpy call
  GuestValue buffer;           // destination of the copy
  GuestValue frame;            // NUI_IMAGE_FRAME*
  GuestValue pitch;            // D3DLOCKED_RECT.Pitch
  // Register pointing at {width, rows} (at -4 / +0); 0 = derive the size from
  // the frame's resolution.
  uint32_t table_register;
  // Size of the destination in bytes (reg 0 = unknown, not limited).
  GuestValue size = {0, 0, false};
  // The title's colour copy is uploaded as a tiled texture, so the colour
  // image has to be written in the GPU's tiled layout (Just Dance 2014).
  bool colour_tiled = false;
};

constexpr PlayerImageHook kKnownPlayerImageHooks[] = {
    // Per-stream loop in 82C50598; streams opened with types 0 (depth +
    // player index, 320x240), 7 (80x60 player map, drives the play-space
    // guide), 1 (colour), 4 (depth).
    {"Just Dance 2014", 0x82C506E8, 0x4BEC22CD, {29, 0, false},
     {1, 0x50, true}, {1, 0x58, true}, 28},
    // Same consumer loop as Just Dance 2014 (NUI 2.0.21256): the memcpy at
    // 82CD5CDC copies rows * pitch bytes from the locked frame texture into
    // r29; streams are 0 (320x240), 7 (80x60), 1 (640x480), 4 (320x240).
    {"Just Dance 2015", 0x82CD5CE0, 0x4BEA5A0D, {29, 0, false},
     {1, 0x50, true}, {1, 0x58, true}, 28},
    // Same consumer loop again (NUI 2.0.21256), 82DA2AE8..: stream table at
    // 82FD3580, memcpy at 82DA2B9C into r29, LockRect pitch at sp+0x58.
    {"Just Dance 2017", 0x82DA2BA0, 0x4BDC967D, {29, 0, false},
     {1, 0x50, true}, {1, 0x58, true}, 28},
    // 82885380(obj, stream): LockRect, then memcpy into *(stream_obj+0x70C);
    // stream_obj = r30, frame at stream_obj+0x704, pitch at sp+0x50.
    {"Just Dance 4", 0x828853E0, 0x4BB232F5, {30, 0x70C, true},
     {30, 0x704, true}, {1, 0x50, true}, 0},
    // The same code with title update 1 (0.0.1.2).
    {"Just Dance 4 (title update 1)", 0x82885AD0, 0x4BB23085,
     {30, 0x70C, true}, {30, 0x704, true}, {1, 0x50, true}, 0},
    // 825AF638 (stream object method): frame copied to r30 (0x28 bytes),
    // LockRect into sp+0x50 {Pitch, pBits}, memcpy into *(r25) of 4(r25)
    // bytes.
    {"Just Dance 3", 0x825AF718, 0x4BECC43D, {25, 0, true}, {30, 0, false},
     {1, 0x50, true}, 0, {25, 4, true}},
};

std::atomic<int> player_image_hook_status[std::size(kKnownPlayerImageHooks)];

// kinect_debug_colour_texture: in the Just Dance 2014/2015 stream loop
// (r31 = Kinect object, r30 = stream offset), the stream's engine texture
// object is at *(*(r31 + 8) + r30 + 0x48). Look through it for a texture
// fetch constant, embedded or behind a pointer, that describes the buffer the
// colour frame was copied to (or any 640x480 texture), and log it.
void LogColourTexture(cpu::ppc::PPCContext_s* ctx, uint32_t buffer) {
  auto* memory = kernel_state()->memory();
  // Readable, committed guest memory only (a heap can contain reserved but
  // unmapped pages, and touching those crashes the emulator).
  auto valid = [&](uint32_t address) {
    if (address < 0x10000) {
      return false;
    }
    BaseHeap* heap = memory->LookupHeap(address);
    uint32_t protect = 0;
    return heap && heap->QueryProtect(address, &protect) &&
           (protect & kMemoryProtectRead);
  };
  const uint32_t kinect = uint32_t(ctx->r[31]);
  if (!valid(kinect + 8)) {
    return;
  }
  const uint32_t streams =
      xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(kinect + 8));
  const uint32_t slot = streams + uint32_t(ctx->r[30]) + 0x48;
  if (!valid(streams) || !valid(slot)) {
    return;
  }
  const uint32_t object = xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(slot));
  XELOGI("Colour texture: buffer {:08X} (physical {:08X}), engine object {:08X}",
         buffer, buffer & 0x1FFFFFFF, object);
  if (!valid(object) || !valid(object + 0x200)) {
    return;
  }
  auto check = [&](uint32_t fetch_address, const char* how, uint32_t where) {
    if (!valid(fetch_address) || !valid(fetch_address + 0x18)) {
      return;
    }
    gpu::xenos::xe_gpu_texture_fetch_t fetch;
    xe::copy_and_swap_32_unaligned(
        &fetch,
        reinterpret_cast<const uint32_t*>(ctx->TranslateVirtual(fetch_address)),
        6);
    const uint32_t base = fetch.base_address << 12;
    const uint32_t width = fetch.size_2d.width + 1;
    const uint32_t height = fetch.size_2d.height + 1;
    const int64_t distance =
        int64_t(base & 0x1FFFFFFF) - int64_t(buffer & 0x1FFFFFFF);
    if (!(base && distance >= -0x2000 && distance <= 0x2000) &&
        !(width == 640 && height == 480)) {
      return;
    }
    XELOGI("Colour texture: fetch constant {} {:X} (at {:08X}): {}x{}, pitch "
           "{} texels, tiled {}, format {}, endian {}, dimension {}, base "
           "{:08X}",
           how, where, fetch_address, width, height,
           uint32_t(fetch.pitch) << 5, uint32_t(fetch.tiled),
           uint32_t(fetch.format), uint32_t(fetch.endianness),
           uint32_t(fetch.dimension), base);
  };
  for (uint32_t offset = 0; offset < 0x200; offset += 4) {
    check(object + offset, "embedded at object +", offset);
    const uint32_t pointer =
        xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(object + offset));
    if (valid(pointer)) {
      check(pointer + 0x1C, "behind pointer at object +", offset);
    }
  }
}

// Colour (type 1, 640x480): the title copies the frame's texture into its own
// 32-bit buffer and crops the player's face for the player slots out of it.
// The camera emulation delivers flat grey, so overwrite that copy with the
// camera image the Kinect SDK bridge sends (XKCL packets).
void WriteColourImage(cpu::ppc::PPCContext_s* ctx, const PlayerImageHook& hook,
                      uint32_t frame) {
  if (usb_kinect::IsActive() || !skeleton_injection_enabled) {
    return;
  }
  uint32_t width, rows;
  if (hook.table_register) {
    const uint32_t table = uint32_t(ctx->r[hook.table_register]);
    width = xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(table - 4));
    rows = xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(table));
  } else {
    // NUI_IMAGE_RESOLUTION at +0x10: 0 = 80x60, 1 = 320x240, 2 = 640x480.
    const uint32_t resolution =
        xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(frame + 0x10));
    if (resolution > 2) {
      return;
    }
    width = resolution == 0 ? 80 : resolution == 1 ? 320 : 640;
    rows = resolution == 0 ? 60 : resolution == 1 ? 240 : 480;
  }
  const uint32_t pitch = hook.pitch.Get(ctx);
  const uint32_t buffer = hook.buffer.Get(ctx);
  if (!buffer || !width || !rows || width > 640 || rows > 480) {
    return;
  }
  if (cvars::kinect_debug_colour_texture &&
      (hook.address == 0x82C506E8 || hook.address == 0x82CD5CE0)) {
    static std::atomic<uint32_t> calls{0};
    if (calls.fetch_add(1) % 300 == 0) {
      LogColourTexture(ctx, buffer);
    }
  }
  const std::string& layout = cvars::kinect_live_colour_layout;
  const bool tiled = layout == "tiled" || (layout != "linear" && hook.colour_tiled);
  const uint32_t pitch_pixels = xe::align(pitch / 4, 32u);
  // Byte offset of pixel (x, y) in the title's buffer.
  auto pixel_offset = [&](uint32_t x, uint32_t y) -> size_t {
    return tiled ? size_t(gpu::texture_address::Tiled2D(
                       int32_t(x), int32_t(y), pitch_pixels, 2))
                 : size_t(y) * pitch + x * 4;
  };

  static std::atomic<bool> logged_frame{false};
  if (!logged_frame.exchange(true)) {
    const uint32_t texture =
        xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(frame + 0x14));
    if (texture) {
      gpu::xenos::xe_gpu_texture_fetch_t fetch;
      xe::copy_and_swap_32_unaligned(
          &fetch,
          reinterpret_cast<const uint32_t*>(
              ctx->TranslateVirtual(texture + 0x1C)),
          6);
      XELOGI(
          "NUI player image: colour frame {}x{}, pitch {} bytes, texture "
          "format {} tiled {} endian {}",
          width, rows, pitch, uint32_t(fetch.format), uint32_t(fetch.tiled),
          uint32_t(fetch.endianness));
    } else {
      XELOGI("NUI player image: colour frame {}x{}, pitch {} bytes", width,
             rows, pitch);
    }
  }
  if (cvars::kinect_colour_test_gradient && pitch >= width * 4) {
    uint8_t* out = ctx->TranslateVirtual(buffer);
    for (uint32_t y = 0; y < rows; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const uint32_t r = x * 255 / (width - 1);
        const uint32_t g = y * 255 / (rows - 1);
        const uint32_t b = ((x / 64 + y / 64) & 1) ? 255 : 0;
        xe::store_and_swap<uint32_t>(out + pixel_offset(x, y),
                                     0xFF000000u | (r << 16) | (g << 8) | b);
      }
    }
    return;
  }
  if (cvars::kinect_colour_test_pattern && pitch >= width * 4) {
    uint8_t* out = ctx->TranslateVirtual(buffer);
    for (uint32_t y = 0; y < rows; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const uint32_t cell = (y / 80) * 8 + x / 80;
        uint32_t value = 0xFF000000u;
        if (x % 80 >= 2 && y % 80 >= 2) {
          // Distinct colours: red, green and blue levels from the cell index.
          const uint32_t r = (cell & 3) * 85, g = ((cell >> 2) & 3) * 85,
                         b = ((cell >> 4) & 3) * 85;
          value |= (r << 16) | (g << 8) | b;
        }
        xe::store_and_swap<uint32_t>(out + pixel_offset(x, y), value);
      }
    }
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
      XELOGI("NUI player image: writing the colour test pattern");
    }
    return;
  }
  if (!cvars::kinect_live_colour) {
    return;
  }
  std::shared_ptr<const std::vector<uint8_t>> colour;
  uint32_t colour_width, colour_height;
  if (!SkeletonReceiver::Get().LatestColour(&colour, &colour_width,
                                            &colour_height, 0.5)) {
    return;
  }
  if (pitch < width * 4) {
    static std::atomic<bool> logged_pitch{false};
    if (!logged_pitch.exchange(true)) {
      XELOGW(
          "NUI player image: colour pitch {} is too small for 32-bit pixels; "
          "leaving the colour image alone",
          pitch);
    }
    return;
  }
  xe::store_and_swap<uint64_t>(ctx->TranslateVirtual(frame),
                               EmulatedNuiCamera::Get().TimestampMs());
  const bool mirror = cvars::kinect_live_colour_mirror;
  uint8_t* out = ctx->TranslateVirtual(buffer);
  const uint8_t* in = colour->data();
  for (uint32_t y = 0; y < rows; ++y) {
    const uint32_t sy = y * colour_height / rows;
    const uint8_t* source_row = in + size_t(sy) * colour_width * 3;
    for (uint32_t x = 0; x < width; ++x) {
      uint32_t sx = x * colour_width / width;
      if (mirror) {
        sx = colour_width - 1 - sx;
      }
      const uint8_t* bgr = source_row + size_t(sx) * 3;
      // X8R8G8B8 as the big-endian guest stores it.
      const uint32_t value = 0xFF000000u | (uint32_t(bgr[2]) << 16) |
                             (uint32_t(bgr[1]) << 8) | uint32_t(bgr[0]);
      xe::store_and_swap<uint32_t>(out + pixel_offset(x, y), value);
    }
  }
  static std::atomic<uint32_t> written{0};
  const uint32_t n = ++written;
  if (n == 1 || n % 600 == 0) {
    XELOGI("NUI player image: wrote colour {}x{} frame #{} ({})", width, rows,
           n, tiled ? "tiled" : "linear");
  }
}

void PlayerImageHookHandler(cpu::ppc::PPCContext_s* ctx, uint32_t address) {
  size_t index = 0;
  while (index < std::size(kKnownPlayerImageHooks) &&
         kKnownPlayerImageHooks[index].address != address) {
    ++index;
  }
  if (index == std::size(kKnownPlayerImageHooks)) {
    return;
  }
  const PlayerImageHook& hook = kKnownPlayerImageHooks[index];
  std::atomic<int>& status = player_image_hook_status[index];
  if (status.load(std::memory_order_relaxed) == 2) {
    return;
  }
  if (status.load(std::memory_order_relaxed) == 0) {
    if (xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(address - 4)) !=
        hook.expected_previous) {
      status = 2;
      return;
    }
    XELOGI("NUI player image: using hook for {}", hook.title);
    status = 1;
  }

  const uint32_t frame = hook.frame.Get(ctx);
  if (!frame) {
    return;
  }
  const uint32_t type =
      xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(frame + 0x0C));
  if (type == 1) {  // colour
    WriteColourImage(ctx, hook, frame);
    return;
  }
  if (type != 0 && type != 7) {  // depth + player index (and the 80x60 map)
    return;
  }
  if (usb_kinect::IsActive()) {
    StampRealCameraFrame(ctx, frame, "player image");
    return;
  }
  if (!skeleton_injection_enabled) {
    return;
  }
  uint32_t width, rows;
  if (hook.table_register) {
    const uint32_t table = uint32_t(ctx->r[hook.table_register]);
    width = xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(table - 4));
    rows = xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(table));
  } else {
    // NUI_IMAGE_RESOLUTION at +0x10: 0 = 80x60, 1 = 320x240, 2 = 640x480.
    const uint32_t resolution =
        xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(frame + 0x10));
    if (resolution > 2) {
      return;
    }
    width = resolution == 0 ? 80 : resolution == 1 ? 320 : 640;
    rows = resolution == 0 ? 60 : resolution == 1 ? 240 : 480;
  }
  const uint32_t pitch = hook.pitch.Get(ctx);
  const uint32_t buffer = hook.buffer.Get(ctx);
  if (!buffer || !width || !rows || width > kDepthWidth ||
      rows > kDepthHeight || pitch < width * 2) {
    return;
  }
  uint32_t write_rows = rows;
  if (hook.size.reg) {
    // Only rows that fit in the destination.
    const uint32_t size = hook.size.Get(ctx);
    write_rows = size < width * 2
                     ? 0
                     : std::min<uint32_t>(rows, (size - width * 2) / pitch + 1);
    if (!write_rows) {
      return;
    }
  }

  std::vector<uint16_t> depth;
  std::vector<uint8_t> mask;
  if (!EmulatedNuiCamera::Get().GetPlayerImage(&depth, &mask)) {
    return;
  }
  // Same timestamp source as the skeleton frames (the runtime leaves 0).
  xe::store_and_swap<uint64_t>(ctx->TranslateVirtual(frame),
                               EmulatedNuiCamera::Get().TimestampMs());
  uint8_t* out = ctx->TranslateVirtual(buffer);
  uint32_t player_pixels = 0;
  for (uint32_t y = 0; y < write_rows; ++y) {
    const uint32_t sy = y * kDepthHeight / rows;
    for (uint32_t x = 0; x < width; ++x) {
      const size_t s = size_t(sy) * kDepthWidth + x * kDepthWidth / width;
      // The player's number (skeleton slot + 1); with the bridge, everyone
      // it sees has their own.
      const uint16_t player = mask[s];
      player_pixels += player != 0;
      // Type 0 is (depth << 3) | player. Type 7 (80x60) looks like a plain
      // player map: with depth in it the whole frame counted as "player",
      // and the position guide reported the player clipped at the edge.
      const uint16_t value =
          type == 7 ? player
                    : uint16_t((std::min<uint32_t>(depth[s], 0x1FFF) << 3) |
                               player);
      xe::store_and_swap<uint16_t>(out + size_t(y) * pitch + x * 2, value);
    }
  }

  static std::atomic<uint32_t> written{0};
  const uint32_t n = ++written;
  if (n == 1 || n % 600 == 0) {
    XELOGI("NUI player image: wrote type {} {}x{} frame #{} ({} player "
           "pixels)",
           type, width, rows, n, player_pixels);
  }
}

// Titles that draw the player straight from the NUI frame's texture (no
// copy to hook, e.g. Dance Central 3's Milo engine): hook the runtime's
// NuiImageStreamGetNextFrame right after it hands out the frame, and write
// the player image into the frame texture itself.
struct ImageFrameHook {
  const char* title;
  uint32_t address;            // instruction after the store to *ppFrame
  uint32_t expected_previous;  // that store
  uint8_t frame_register;      // NUI_IMAGE_FRAME*
  uint32_t types;              // bit per NUI image type to write
};

constexpr ImageFrameHook kKnownImageFrameHooks[] = {
    // NuiImageStreamGetNextFrame 8280C6E8 (NUI 2.0.21173): stw r9, 0(r28).
    {"Dance Central 3", 0x8280C94C, 0x913C0000, 9,
     (1u << 0) | (1u << 7) | (1u << 4)},
    // NuiImageStreamGetNextFrame 82D6CFE8 (NUI 2.0.20764): same code. JD3
    // opens (824C7948..) colour 640x480 (handle at +0x34), depth 320x240
    // (type 4, +0x2C), the 80x60 player map (type 7, +0x30) and depth +
    // player index 320x240 (type 0, +0x190C). Type 0 feeds its hand glow
    // (824D3B48 -> 824D3620 with the chosen skeleton's player index); the
    // others are read through a script-bound stream reader (825AF638).
    // Type 4 is written too, as for Dance Central 3: like there, the
    // runtime may leave it empty in memory, which would leave the body out
    // of the player square while the hand glow (from type 0) shows.
    {"Just Dance 3", 0x82D6D24C, 0x913C0000, 9,
     (1u << 0) | (1u << 7) | (1u << 4)},
};

// The known hooks plus any found by ScanTitleForHooks (reserved so that
// adding one never moves the others).
constexpr size_t kMaxImageFrameHooks = 16;
std::vector<ImageFrameHook> active_image_frame_hooks = [] {
  std::vector<ImageFrameHook> hooks;
  hooks.reserve(kMaxImageFrameHooks);
  hooks.assign(std::begin(kKnownImageFrameHooks),
               std::end(kKnownImageFrameHooks));
  return hooks;
}();
std::atomic<int> image_frame_hook_status[kMaxImageFrameHooks];

void ImageFrameHookHandler(cpu::ppc::PPCContext_s* ctx, uint32_t address) {
  size_t index = 0;
  while (index < active_image_frame_hooks.size() &&
         active_image_frame_hooks[index].address != address) {
    ++index;
  }
  if (index == active_image_frame_hooks.size()) {
    return;
  }
  const ImageFrameHook& hook = active_image_frame_hooks[index];
  std::atomic<int>& status = image_frame_hook_status[index];
  if (status.load(std::memory_order_relaxed) == 2) {
    return;
  }
  if (status.load(std::memory_order_relaxed) == 0) {
    if (xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(address - 4)) !=
        hook.expected_previous) {
      status = 2;
      return;
    }
    XELOGI("NUI image frame: using hook for {}", hook.title);
    status = 1;
  }
  const uint32_t frame = uint32_t(ctx->r[hook.frame_register]);
  if (!frame) {
    return;
  }
  const uint32_t type =
      xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(frame + 0x0C));
  const uint32_t texture =
      xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(frame + 0x14));
  if (!texture) {
    return;
  }
  // D3DBaseTexture: D3DResource (0x18), MipFlush, then the fetch constant
  // (holding a virtual address of the physical memory, see VdSwap).
  gpu::xenos::xe_gpu_texture_fetch_t fetch;
  xe::copy_and_swap_32_unaligned(
      &fetch,
      reinterpret_cast<const uint32_t*>(ctx->TranslateVirtual(texture + 0x1C)),
      6);
  const uint32_t width = fetch.size_2d.width + 1;
  const uint32_t height = fetch.size_2d.height + 1;
  const uint32_t base = fetch.base_address << 12;
  const uint32_t pitch = uint32_t(fetch.pitch) << 5;  // texels

  static std::atomic<uint32_t> logged_types{0};
  const uint32_t bit = 1u << std::min<uint32_t>(type, 31);
  if (!(logged_types.fetch_or(bit) & bit)) {
    XELOGI("NUI image frame: type {} resolution {} texture {:08X}: {}x{} "
           "format {} tiled {} pitch {} base {:08X} endian {}",
           type,
           xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(frame + 0x10)),
           texture, width, height, uint32_t(fetch.format),
           uint32_t(fetch.tiled), pitch, base, uint32_t(fetch.endianness));
  }

  if (usb_kinect::IsActive()) {
    StampRealCameraFrame(ctx, frame, "image");
    return;
  }
  // Types 0 (depth << 3 | player) and 7 (player map); Dance Central 3 only
  // opens type 4 (depth, 320x240) and type 2 (colour), so type 4 is written
  // too, in the encoding the runtime itself uses for it.
  if (!skeleton_injection_enabled || type > 31 ||
      !(hook.types & (1u << type))) {
    return;
  }
  if (fetch.format != gpu::xenos::TextureFormat::k_16 || !base ||
      width > kDepthWidth || height > kDepthHeight || pitch < width ||
      !kernel_state()->memory()->LookupHeap(base)) {
    return;
  }
  const uint32_t pitch_aligned = xe::align(pitch, 32u);
  auto texel_offset = [&](uint32_t x, uint32_t y) {
    return fetch.tiled ? uint32_t(gpu::texture_address::Tiled2D(
                             int32_t(x), int32_t(y), pitch_aligned, 1))
                       : (y * pitch + x) * 2;
  };

  // Type 4: look at what the runtime wrote (shifted by 3 like type 0, or
  // plain millimetres?) before replacing it. Logged now and then.
  static std::atomic<int> depth_shifted{-1};  // -1 unknown, 0 mm, 1 << 3
  static std::atomic<uint32_t> depth_inspections{0};
  if (type == 4) {
    const uint32_t inspection = depth_inspections.fetch_add(1);
    const std::string encoding = cvars::kinect_image_depth_encoding;
    if (depth_shifted.load() < 0 && encoding != "auto") {
      depth_shifted = encoding == "mm" ? 0 : 1;
      XELOGI("NUI image frame: type 4 written as {} (kinect_image_depth_"
             "encoding)", depth_shifted.load() ? "depth << 3 | player"
                                                : "plain millimetres");
    }
    if (depth_shifted.load() < 0 || inspection % 900 == 0) {
      const uint8_t* in = ctx->TranslateVirtual(base);
      uint32_t nonzero = 0, low_bits = 0;
      uint32_t min_value = 0xFFFF, max_value = 0;
      std::vector<uint16_t> values;
      for (uint32_t y = 0; y < height; y += 4) {
        for (uint32_t x = 0; x < width; x += 4) {
          const uint16_t v =
              xe::load_and_swap<uint16_t>(in + texel_offset(x, y));
          if (!v) {
            continue;
          }
          ++nonzero;
          low_bits += (v & 7) != 0;
          min_value = std::min<uint32_t>(min_value, v);
          max_value = std::max<uint32_t>(max_value, v);
          values.push_back(v);
        }
      }
      uint32_t median = 0;
      if (!values.empty()) {
        std::nth_element(values.begin(), values.begin() + values.size() / 2,
                         values.end());
        median = values[values.size() / 2];
      }
      if (inspection < 3 || inspection % 900 == 0)
      XELOGI("NUI image frame: runtime's type 4 data: {} nonzero samples, "
             "raw {}..{}, median {}, {} with low 3 bits set",
             nonzero, nonzero ? min_value : 0, max_value, median, low_bits);
      if (depth_shifted.load() < 0 && nonzero <= 100 && inspection >= 30) {
        // The runtime leaves the CPU copy empty (its depth processing runs
        // on the GPU): use the depth-and-player-index encoding.
        depth_shifted = 1;
        XELOGI("NUI image frame: type 4 is empty in memory; writing depth "
               "<< 3 | player");
      }
      if (depth_shifted.load() < 0 && nonzero > 100) {
        // The emulated room is 0.8..3.5 m: shifted values are ~8x larger.
        depth_shifted = median >= 8000 ? 1 : 0;
        XELOGI("NUI image frame: type 4 holds {}", depth_shifted.load()
                                                        ? "depth << 3 | player"
                                                        : "plain millimetres");
      }
    }
    if (depth_shifted.load() < 0) {
      return;
    }
  }

  std::vector<uint16_t> depth;
  std::vector<uint8_t> mask;
  if (!EmulatedNuiCamera::Get().GetPlayerImage(&depth, &mask)) {
    return;
  }
  xe::store_and_swap<uint64_t>(ctx->TranslateVirtual(frame),
                               EmulatedNuiCamera::Get().TimestampMs());
  uint8_t* out = ctx->TranslateVirtual(base);
  uint32_t player_pixels = 0;
  for (uint32_t y = 0; y < height; ++y) {
    const uint32_t sy = y * kDepthHeight / height;
    for (uint32_t x = 0; x < width; ++x) {
      const size_t s = size_t(sy) * kDepthWidth + x * kDepthWidth / width;
      // The player's number (skeleton slot + 1); with the bridge, everyone
      // it sees has their own.
      const uint16_t player = mask[s];
      player_pixels += player != 0;
      uint16_t value;
      if (type == 7) {
        value = player;
      } else if (type == 4 && depth_shifted.load() == 0) {
        value = depth[s];
      } else {
        value = uint16_t((std::min<uint32_t>(depth[s], 0x1FFF) << 3) | player);
      }
      const uint32_t offset = texel_offset(x, y);
      // Written like the runtime's own CPU writes; the GPU's write watch on
      // the texture memory notices them.
      xe::store_and_swap<uint16_t>(out + offset, value);
    }
  }
  static std::atomic<uint32_t> written{0};
  const uint32_t n = ++written;
  if (n == 1 || n % 600 == 0) {
    XELOGI("NUI image frame: wrote type {} {}x{} frame #{} ({} player "
           "pixels)",
           type, width, height, n, player_pixels);
  }
}

void InstallPlayerImageHooks() {
  if (!cvars::kinect_player_index) {
    return;
  }
  for (const PlayerImageHook& hook : kKnownPlayerImageHooks) {
    cpu::ppc::RegisterGuestHook(hook.address, PlayerImageHookHandler);
  }
  for (const ImageFrameHook& hook : active_image_frame_hooks) {
    cpu::ppc::RegisterGuestHook(hook.address, ImageFrameHookHandler);
  }
}

// --- Emulated skeleton ------------------------------------------------------
// With kinect_emulated_skeleton, a tracked skeleton for the synthetic person
// is written into slot 0 of every NUI_SKELETON_FRAME the title receives from
// NuiSkeletonGetNextFrame, so title code can be exercised before the
// runtime's own tracker works under emulation. Its arms follow controller 1
// (kinect_skeleton_controller): the sticks move the hands, LB/RB raise them.
//
// NUI_SKELETON_FRAME (0xAB0): +0x08 frame number, +0x10 floor clip plane,
// +0x20 normal to gravity, +0x30 NUI_SKELETON_DATA[6] (stride 0x1C0):
//   +0x000 tracking state (2 = tracked), +0x004 tracking ID,
//   +0x008 enrollment index (-1 = none), +0x00C user index (0xFE = none),
//   +0x010 position (XMVECTOR), +0x020 joints[20] (XMVECTOR, metres),
//   +0x160 joint tracking states[20] (2 = tracked), +0x1B0 quality flags.

struct SkeletonHook {
  const char* title;
  uint32_t address;
  uint32_t frame_register;
  // Instruction word expected just before address (0 = don't check): the
  // memcpy call that fills the caller's frame inside NuiSkeletonGetNextFrame.
  uint32_t expected_previous;
};

// Each hook sits right after NuiSkeletonGetNextFrame copies the new frame
// into the caller's buffer. All are installed; the one whose signature
// matches the running title is used.
constexpr SkeletonHook kKnownSkeletonHooks[] = {
    {"Kinect Adventures (NUI 2.0.11775)", 0x82D2686C, 27, 0x484861F9},
    {"Just Dance 2014 (NUI 2.0.21250)", 0x82A7A69C, 26, 0x480983A1},
    {"Just Dance 2015 (NUI 2.0.21256)", 0x82AE4ACC, 26, 0x48096CA9},
    // NuiSkeletonGetNextFrame 82B77128 (same runtime code as 2015): the memcpy
    // of the 0xAB0-byte frame at 82B77230, the frame in r26.
    {"Just Dance 2017 (NUI 2.0.21256)", 0x82B77234, 26, 0x4BFF5071},
    {"Dance Central 3", 0x82806A9C, 26, 0x4BFF8CA1},
    {"Just Dance 4 (NUI 2.0.21076)", 0x8267C13C, 26, 0x4BF9A6C1},
    // The same code with title update 1 (0.0.1.2).
    {"Just Dance 4 title update 1 (NUI 2.0.21076)", 0x8267C64C, 26,
     0x4BF9A0E9},
    {"Just Dance 3 (NUI 2.0.20764)", 0x82D6EA24, 26, 0x4B70D131},
};

std::vector<SkeletonHook> active_skeleton_hooks;
// Per hook: 0 = not checked yet, 1 = matches, 2 = does not match.
constexpr size_t kMaxSkeletonHooks = std::size(kKnownSkeletonHooks) + 8;
std::atomic<int> skeleton_hook_status[kMaxSkeletonHooks];
// Whether InstallSkeletonHook set the hooks up (so a scan may add to them).
bool skeleton_hooks_installed = false;

void StoreVector(uint8_t* p, const Vec3& mm) {
  xe::store_and_swap<float>(p + 0x0, mm.x * 0.001f);
  xe::store_and_swap<float>(p + 0x4, mm.y * 0.001f);
  xe::store_and_swap<float>(p + 0x8, mm.z * 0.001f);
  xe::store_and_swap<float>(p + 0xC, 1.0f);
}

float StickAxis(int16_t value) {
  constexpr float kDeadZone = 0.2f;
  const float v = std::max(-1.0f, float(value) / 32767.0f);
  if (std::abs(v) < kDeadZone) {
    return 0.0f;
  }
  return (v - std::copysign(kDeadZone, v)) / (1.0f - kDeadZone);
}

// Reposition one arm (shoulder, elbow, wrist, hand at a[0..3]) from a stick
// and a "raise" button. side is -1 for the left arm, +1 for the right.
void ApplyArmControl(Vec3* a, float side, float stick_x, float stick_y,
                     bool raise) {
  const Vec3 shoulder = a[0];
  Vec3 hand;
  if (raise) {
    // Straight up, well above the head.
    hand = {shoulder.x + side * 120.0f, shoulder.y + 620.0f, shoulder.z};
  } else if (stick_x != 0.0f || stick_y != 0.0f) {
    // Arm reaching towards the sensor; the stick moves the hand across a
    // 1 m x 1 m area centred a little outside the shoulder.
    hand = {shoulder.x + side * 150.0f + stick_x * 500.0f,
            shoulder.y + stick_y * 500.0f, shoulder.z - 380.0f};
  } else {
    return;
  }
  a[1] = Lerp(shoulder, hand, 0.5f);  // elbow
  a[1].x += side * 60.0f;
  a[1].y -= 60.0f;
  a[2] = Lerp(shoulder, hand, 0.92f);  // wrist
  a[3] = hand;
}

void SkeletonHookHandler(cpu::ppc::PPCContext_s* ctx, uint32_t address) {
  size_t index = 0;
  while (index < active_skeleton_hooks.size() &&
         active_skeleton_hooks[index].address != address) {
    ++index;
  }
  if (index == active_skeleton_hooks.size()) {
    return;
  }
  const SkeletonHook& hook = active_skeleton_hooks[index];
  std::atomic<int>& status = skeleton_hook_status[index];
  if (status.load(std::memory_order_relaxed) == 2) {
    return;
  }
  if (status.load(std::memory_order_relaxed) == 0) {
    const uint32_t previous =
        hook.expected_previous
            ? xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(address - 4))
            : 0;
    if (previous != hook.expected_previous) {
      XELOGI("NUI skeleton: hook for {} does not match this title, ignored",
             hook.title);
      status = 2;
      return;
    }
    XELOGI("NUI skeleton: using hook for {}", hook.title);
    status = 1;
  }

  const uint32_t frame = uint32_t(ctx->r[hook.frame_register]);
  if (!frame) {
    return;
  }
  if (usb_kinect::IsActive()) {
    StampRealCameraFrame(ctx, frame, "skeleton");
    LogRealCameraSkeletons(ctx, frame);
    return;
  }
  if (!skeleton_injection_enabled) {
    return;
  }
  Vec3 joints[kJointCount];
  uint8_t states[kJointCount];
  std::memset(states, 2, sizeof(states));
  const bool live_skeleton = cvars::kinect_skeleton_udp_port != 0;
  ExternalBodies bodies;
  if (live_skeleton) {
    // Live tracker: nobody fresh from the bridge means nobody is there.
    if (!SkeletonReceiver::Get().InterpolatedBodies(
            &bodies, kLiveSkeletonMaxAge,
            cvars::kinect_skeleton_interpolation_ms * 0.001) ||
        !bodies.count) {
      return;
    }
    // For the log lines below: the first full skeleton (zeros if everyone
    // is position only).
    std::memset(joints, 0, sizeof(joints));
    for (uint32_t b = 0; b < bodies.count; ++b) {
      if (bodies.body[b].state == 2) {
        std::memcpy(joints, bodies.body[b].joints, sizeof(joints));
        break;
      }
    }
  } else {
    const double t = EmulatedNuiCamera::Get().SceneTime();
    if (t < 0.0) {
      return;
    }
    const PersonPose pose = PoseAt(t);
    if (!pose.visible) {
      return;
    }
    SkeletonJoints(pose, cvars::kinect_emulated_person_mm, joints);
  }

  if (!live_skeleton && cvars::kinect_skeleton_controller) {
    hid::X_INPUT_STATE input = {};
    X_RESULT result;
    {
      auto input_system = kernel_state()->emulator()->input_system();
      auto lock = input_system->lock();
      result = input_system->GetState(
          0, hid::X_INPUT_FLAG::X_INPUT_FLAG_GAMEPAD, &input);
    }
    if (result == X_ERROR_SUCCESS) {
      const uint16_t buttons = input.gamepad.buttons;
      ApplyArmControl(joints + 4, -1.0f, StickAxis(input.gamepad.thumb_lx),
                      StickAxis(input.gamepad.thumb_ly),
                      buttons & hid::X_INPUT_GAMEPAD_LEFT_SHOULDER);
      ApplyArmControl(joints + 8, 1.0f, StickAxis(input.gamepad.thumb_rx),
                      StickAxis(input.gamepad.thumb_ry),
                      buttons & hid::X_INPUT_GAMEPAD_RIGHT_SHOULDER);
    }
  }

  // liTimeStamp (+0x00, milliseconds). The runtime leaves it 0 under
  // emulation, and Just Dance places skeleton samples on its timeline by it:
  // with every frame at time 0 no move ever matched (all "X").
  xe::store_and_swap<uint64_t>(ctx->TranslateVirtual(frame),
                               EmulatedNuiCamera::Get().TimestampMs());

  {
    uint8_t* header = ctx->TranslateVirtual(frame);
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
      XELOGI("NUI skeleton frame header from the runtime: floor plane ({}, "
             "{}, {}, {}), normal to gravity ({}, {}, {}, {})",
             xe::load_and_swap<float>(header + 0x10),
             xe::load_and_swap<float>(header + 0x14),
             xe::load_and_swap<float>(header + 0x18),
             xe::load_and_swap<float>(header + 0x1C),
             xe::load_and_swap<float>(header + 0x20),
             xe::load_and_swap<float>(header + 0x24),
             xe::load_and_swap<float>(header + 0x28),
             xe::load_and_swap<float>(header + 0x2C));
    }
    if (cvars::kinect_sensor_tilt_deg > -90.0) {
      const float tilt = float(cvars::kinect_sensor_tilt_deg) * 0.017453293f;
      xe::store_and_swap<float>(header + 0x20, 0.0f);
      xe::store_and_swap<float>(header + 0x24, std::cos(tilt));
      xe::store_and_swap<float>(header + 0x28, std::sin(tilt));
      xe::store_and_swap<float>(header + 0x2C, 0.0f);
    }
  }

  if (!live_skeleton) {
    // The synthetic person, in slot 0.
    ExternalBody& body = bodies.body[0];
    body.tracking_id = 1;
    body.number = 1;
    body.state = 2;
    body.position = joints[0];
    std::memcpy(body.joints, joints, sizeof(joints));
    std::memcpy(body.states, states, sizeof(states));
    bodies.count = 1;
  }
  // Everyone in their own slot (number - 1, matching their value in the
  // player image), the rest cleared: tracked bodies with all 20 joints,
  // position-only ones with just the position, like the runtime does.
  uint8_t* slots = ctx->TranslateVirtual(frame + 0x30);
  std::memset(slots, 0, 0x1C0 * kMaxBodies);
  for (uint32_t b = 0; b < bodies.count; ++b) {
    const ExternalBody& body = bodies.body[b];
    uint8_t* s = slots + (body.number - 1) * 0x1C0;
    xe::store_and_swap<uint32_t>(s + 0x000, body.state);  // 2 = tracked
    xe::store_and_swap<uint32_t>(s + 0x004, body.tracking_id);
    xe::store_and_swap<uint32_t>(s + 0x008, 0xFFFFFFFF);  // not enrolled
    xe::store_and_swap<uint32_t>(s + 0x00C, 0xFE);        // no user
    StoreVector(s + 0x010, body.position);
    if (body.state != 2) {
      continue;
    }
    for (uint32_t i = 0; i < kJointCount; ++i) {
      StoreVector(s + 0x020 + i * 0x10, body.joints[i]);
      xe::store_and_swap<uint32_t>(s + 0x160 + i * 4, body.states[i]);
    }
  }

  // Rates, every 5 s: how often the title gets a new skeleton frame, how far
  // the runtime's frame numbers advance, how many skeletons the bridge sent
  // and how many frames reused the bridge skeleton of the previous frame.
  if (live_skeleton) {
    static std::mutex rate_mutex;
    static auto window_start = std::chrono::steady_clock::now();
    static uint32_t calls = 0, reused = 0, last_packets = 0,
                    window_packets = 0, first_number = 0, last_number = 0,
                    same_hand = 0;
    static Vec3 last_hand = {};
    static bool have_number = false;
    std::lock_guard<std::mutex> lock(rate_mutex);
    const uint32_t packets = SkeletonReceiver::Get().packet_count();
    const uint32_t number =
        xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(frame + 0x08));
    if (!have_number) {
      have_number = true;
      first_number = number;
      window_packets = packets;
    }
    ++calls;
    reused += packets == last_packets;
    same_hand += joints[11].x == last_hand.x && joints[11].y == last_hand.y &&
                 joints[11].z == last_hand.z;
    last_hand = joints[11];
    last_packets = packets;
    last_number = number;
    const auto now = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(now - window_start).count();
    if (seconds >= 5.0) {
      XELOGI("NUI skeleton rate: title got {:.1f} frames/s (runtime frame "
             "numbers +{:.1f}/s), bridge sent {:.1f} skeletons/s, {}% of "
             "frames reused the previous bridge skeleton, {}% had the right "
             "hand exactly where it was",
             calls / seconds, (last_number - first_number) / seconds,
             (packets - window_packets) / seconds,
             calls ? reused * 100 / calls : 0,
             calls ? same_hand * 100 / calls : 0);
      window_start = now;
      calls = reused = same_hand = 0;
      first_number = number;
      window_packets = packets;
    }
  }

  static std::atomic<uint32_t> injected{0};
  const uint32_t n = ++injected;
  if (n == 1 || n % 300 == 0) {
    XELOGI(
        "NUI skeleton: injected skeleton frame #{} into frame {:08X} ({} "
        "people; first full skeleton: hip {:.2f}, {:.2f}, {:.2f} m; right "
        "hand {:.2f}, {:.2f}, {:.2f} m)",
        n, frame, bodies.count, joints[0].x * 0.001f, joints[0].y * 0.001f,
        joints[0].z * 0.001f, joints[11].x * 0.001f, joints[11].y * 0.001f,
        joints[11].z * 0.001f);
  }
}

// --- The title's choice of full skeletons ------------------------------------
// A Kinect v1 gives full skeletons for two people at a time. Titles with more
// players (Just Dance: NuiSkeletonTrackingEnable with flag 2, "title sets
// tracked skeletons") pick the two with NuiSkeletonSetTrackedSkeletons and
// rotate through the others. The hook sits in that runtime function, after
// its prologue, with the array of two tracking IDs in r3; the IDs are ours
// (written into the frames above), so they go straight to the bridge, which
// asks the Kinect SDK to track those two people.

struct TrackedSkeletonsHook {
  const char* title;
  uint32_t address;
  uint32_t expected_previous;  // the bl to the register-save helper
};

constexpr TrackedSkeletonsHook kKnownTrackedSkeletonsHooks[] = {
    // NuiSkeletonSetTrackedSkeletons 82B76A80; called from 826E29D4,
    // 826E29F0, 826E2C00 and 826E2C30.
    {"Just Dance 2017 (NUI 2.0.21256)", 0x82B76A88, 0x4BDE1C79},
    // Same runtime function at 82AE41C0.
    {"Just Dance 2015 (NUI 2.0.21256)", 0x82AE41C8, 0x481A29C9},
};

void TrackedSkeletonsHookHandler(cpu::ppc::PPCContext_s* ctx,
                                 uint32_t address) {
  const TrackedSkeletonsHook* hook = nullptr;
  for (const TrackedSkeletonsHook& known : kKnownTrackedSkeletonsHooks) {
    if (known.address == address) {
      hook = &known;
    }
  }
  if (!hook || xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(
                   address - 4)) != hook->expected_previous) {
    return;
  }
  const uint32_t ids = uint32_t(ctx->r[3]);
  if (!ids) {
    return;
  }
  const uint32_t first = xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(ids));
  const uint32_t second =
      xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(ids + 4));
  // Just Dance 2017 alternates every frame between its pick and (0, 0);
  // passing the (0, 0) on would have the Kinect drop and re-acquire the
  // skeletons all the time, so it only counts as "no change".
  if (!first && !second) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
      XELOGI("NUI skeleton: {} also asks for no full skeletons (0, 0); "
             "ignored, the last pick stays",
             hook->title);
    }
    return;
  }
  // The title calls this every frame; the bridge hears about changes, and
  // once a second in case a packet was lost.
  static std::mutex mutex;
  static uint32_t last_first = 0, last_second = 0;
  static auto last_sent = std::chrono::steady_clock::time_point();
  static uint32_t changes = 0;
  std::lock_guard<std::mutex> lock(mutex);
  const auto now = std::chrono::steady_clock::now();
  const bool changed = first != last_first || second != last_second;
  if (!changed &&
      std::chrono::duration<double>(now - last_sent).count() < 1.0) {
    return;
  }
  const bool sent = SkeletonReceiver::Get().ChooseSkeletons(first, second);
  if (sent) {
    last_first = first;
    last_second = second;
    last_sent = now;
  }
  if (changed && changes++ < 200) {
    XELOGI("NUI skeleton: {} asks for full skeletons for tracking IDs {} and "
           "{} -> {}",
           hook->title, first, second,
           sent ? "sent to the Kinect SDK bridge" : "no bridge heard from");
  }
}

void InstallSkeletonHook() {
  skeleton_injection_enabled =
      cvars::kinect_connected &&
      (cvars::kinect_skeleton_udp_port || cvars::kinect_emulated_skeleton);
  // A real Kinect may turn up on the first camera request; the hooks are
  // installed now (before translation) so they can timestamp its frames.
  if (!skeleton_injection_enabled && !cvars::allow_kinect_camera) {
    return;
  }
  if (cvars::kinect_connected && cvars::kinect_skeleton_udp_port) {
    SkeletonReceiver::Get().Start(
        uint16_t(cvars::kinect_skeleton_udp_port));
  }
  const std::string spec = cvars::kinect_skeleton_hook;
  if (!spec.empty()) {
    const auto parts = xe::utf8::split(spec, ":");
    SkeletonHook hook = {"kinect_skeleton_hook", 0, 0, 0};
    if (parts.size() == 2) {
      hook.address =
          uint32_t(std::strtoul(std::string(parts[0]).c_str(), nullptr, 16));
      hook.frame_register =
          uint32_t(std::strtoul(std::string(parts[1]).c_str(), nullptr, 10));
    }
    if (!hook.address || hook.frame_register > 31) {
      XELOGE("NUI skeleton: kinect_skeleton_hook must be \"ADDR:REG\", got "
             "\"{}\"",
             spec);
      return;
    }
    active_skeleton_hooks.push_back(hook);
  } else {
    active_skeleton_hooks.reserve(kMaxSkeletonHooks);
    active_skeleton_hooks.assign(std::begin(kKnownSkeletonHooks),
                                 std::end(kKnownSkeletonHooks));
    skeleton_hooks_installed = true;
  }
  InstallPlayerImageHooks();
  if (cvars::kinect_connected && cvars::kinect_skeleton_udp_port) {
    for (const TrackedSkeletonsHook& hook : kKnownTrackedSkeletonsHooks) {
      cpu::ppc::RegisterGuestHook(hook.address, TrackedSkeletonsHookHandler);
    }
  }
  for (const SkeletonHook& hook : active_skeleton_hooks) {
    cpu::ppc::RegisterGuestHook(hook.address, SkeletonHookHandler);
    XELOGI("NUI skeleton: emulated skeleton hook at {:08X} (frame in r{}) "
           "for {}",
           hook.address, hook.frame_register, hook.title);
  }
}

}  // namespace

// Used by XamNuiCameraElevationSetAngle / GetAngle (xam_nui.cc).
// The tilt motor, for the title (XamNuiCameraElevation*, its auto-tilt) and
// the Kinect Tuner: the real Kinect's own motor in real-Kinect mode
// (allow_kinect_camera), else the SDK bridge's.
bool KinectBridgeRequestTilt(int32_t angle_deg) {
  if (!cvars::kinect_auto_tilt) {
    return false;
  }
  if (usb_kinect::IsActive()) {
    return usb_kinect::MotorSetAngle(angle_deg);
  }
  SkeletonReceiver& receiver = SkeletonReceiver::Get();
  return receiver.started() && receiver.RequestTilt(angle_deg);
}

bool KinectBridgeTiltStatus(int32_t* angle_deg, bool* moving) {
  if (!cvars::kinect_auto_tilt) {
    return false;
  }
  if (usb_kinect::IsActive()) {
    return usb_kinect::MotorStatus(angle_deg, moving, nullptr);
  }
  SkeletonReceiver& receiver = SkeletonReceiver::Get();
  return receiver.started() && receiver.TiltStatus(angle_deg, moving, 3.0);
}

// What the Kinect Tuner shows (kinect_tuner.h): the bridge's latest depth
// image, 8x smaller, with everyone's outline, and the motor.
bool KinectGetTunerView(KinectTunerView* view) {
  *view = KinectTunerView();
  view->real_kinect = usb_kinect::IsActive();
  view->have_tilt = KinectBridgeTiltStatus(&view->tilt_degrees,
                                           &view->tilt_moving);
  if (!cvars::kinect_skeleton_udp_port) {
    return view->have_tilt;
  }
  SkeletonReceiver& receiver = SkeletonReceiver::Get();
  ExternalBodies bodies;
  if (receiver.InterpolatedBodies(&bodies, 1.0, 0.0)) {
    view->bridge = true;
    for (uint32_t b = 0; b < bodies.count; ++b) {
      ++view->people;
      view->full_skeletons += bodies.body[b].state == 2;
    }
  }
  std::vector<uint16_t> depth;
  std::vector<uint8_t> mask;
  if (!receiver.LatestImage(&depth, &mask, 1.0) ||
      depth.size() < size_t(kDepthWidth) * kDepthHeight) {
    return view->bridge || view->have_tilt;
  }
  view->bridge = true;
  view->have_image = true;
  constexpr uint32_t kStepX = kDepthWidth / KinectTunerView::kWidth;
  constexpr uint32_t kStepY = kDepthHeight / KinectTunerView::kHeight;
  for (uint32_t y = 0; y < KinectTunerView::kHeight; ++y) {
    for (uint32_t x = 0; x < KinectTunerView::kWidth; ++x) {
      const size_t s = size_t(y * kStepY + kStepY / 2) * kDepthWidth +
                       x * kStepX + kStepX / 2;
      const uint32_t mm = depth[s];
      const size_t d = size_t(y) * KinectTunerView::kWidth + x;
      view->grey[d] =
          mm ? uint8_t(255 - std::min<uint32_t>(255, mm * 255 / 4500)) : 0;
      view->player[d] = mask[s];
    }
  }
  return true;
}

dword_result_t PsCamDeviceRequest_entry(lpvoid_t request,
                                        const ppc_context_t& context) {
  if (!request) {
    return kStatusInvalidDeviceRequest;
  }
  // A real Kinect (allow_kinect_camera) takes priority; without one the
  // emulated sensor answers as before.
  uint32_t usb_status;
  if (usb_kinect::TryHandleRequest(request.guest_address(), &usb_status)) {
    return usb_status;
  }
  if (!cvars::kinect_connected) {
    XELOGD("PsCamDeviceRequest({:08X}) -> C000009D (no camera):{}",
           request.guest_address(),
           HexDump(request.as<const uint8_t*>(), 0x40));
    return kStatusDeviceNotConnected;
  }
  return EmulatedNuiCamera::Get().HandleRequest(context.value(),
                                                request.guest_address());
}
DECLARE_XBOXKRNL_EXPORT1(PsCamDeviceRequest, kNone, kSketchy);

// ETX event-tracing producer API used by the NUI runtime. Tracing is not
// emulated; report success so callers continue.
dword_result_t EtxProducerRegister_entry() { return X_STATUS_SUCCESS; }
DECLARE_XBOXKRNL_EXPORT1(EtxProducerRegister, kNone, kStub);

dword_result_t EtxProducerUnregister_entry() { return X_STATUS_SUCCESS; }
DECLARE_XBOXKRNL_EXPORT1(EtxProducerUnregister, kNone, kStub);

dword_result_t EtxProducerLog_entry() { return X_STATUS_SUCCESS; }
DECLARE_XBOXKRNL_EXPORT2(EtxProducerLog, kNone, kStub, kHighFrequency);

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

namespace xe {
namespace kernel {
namespace xboxkrnl {
namespace {

// --- Title workarounds -------------------------------------------------------
// Dance Central 3 runs its main loop from inside the C runtime's exception
// filter: after initialising, it registers a callback (which calls App::Run)
// as the unhandled-exception handler and then deliberately writes to address
// 0. Xenia does not dispatch guest access violations to guest exception
// handlers, so the title just crashed after its splash screens. Call the
// callback directly at the faulting instruction instead.

struct DeliberateFaultFix {
  const char* title;
  uint32_t fault_address;
  uint32_t fault_instruction;  // expected word at fault_address
  uint32_t handler;            // function the exception filter would call
};

constexpr DeliberateFaultFix kDeliberateFaultFixes[] = {
    {"Dance Central 3", 0x82311A94, 0x91400000 /* stw r10, 0(0) */,
     0x82311A40},
};

// The known fixes plus any ScanTitleForHooks finds (Dance Central 2 too).
std::vector<DeliberateFaultFix> active_fault_fixes = [] {
  std::vector<DeliberateFaultFix> fixes;
  fixes.reserve(8);
  fixes.assign(std::begin(kDeliberateFaultFixes),
               std::end(kDeliberateFaultFixes));
  return fixes;
}();

void DeliberateFaultHook(cpu::ppc::PPCContext_s* ctx, uint32_t address) {
  for (const DeliberateFaultFix& fix : active_fault_fixes) {
    if (fix.fault_address != address) {
      continue;
    }
    const uint32_t instruction =
        xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(address));
    const uint32_t handler_prologue =
        xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(fix.handler));
    if (instruction != fix.fault_instruction ||
        handler_prologue != 0x7D8802A6 /* mflr r12 */) {
      return;  // a different title
    }
    XELOGI("{} workaround: calling exception callback {:08X} instead of "
           "faulting at {:08X}",
           fix.title, fix.handler, address);
    uint64_t args[] = {0};
    kernel_state()->processor()->Execute(
        XThread::GetCurrentThread()->thread_state(), fix.handler, args, 1);
    XELOGW("{} workaround: exception callback returned", fix.title);
    return;
  }
}

// Just Dance 4 diagnostic (kinect_debug_jd4_hand): at 82835D28 (per joint;
// r30 = player object, r31 = skeleton slot, r26 = joint), f29-f31 hold the
// position the game uses. For the hands it comes from another table, not
// from the skeleton frame at *(r30 + 0x6F4); log both for the right hand.
void Jd4HandDebugHook(cpu::ppc::PPCContext_s* ctx, uint32_t address) {
  static std::atomic<int> status{0};
  if (status == 2) {
    return;
  }
  if (status == 0) {
    status = xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(address - 4)) ==
                     0xC3A10060  // lfs f29, 0x60(r1)
                 ? 1
                 : 2;
    if (status == 2) {
      return;
    }
  }
  if (uint32_t(ctx->r[26]) != 11) {
    return;
  }
  static std::atomic<uint32_t> calls{0};
  const uint32_t n = calls.fetch_add(1);
  if (n >= 1200 || n % 2) {
    return;
  }
  const uint32_t player = uint32_t(ctx->r[30]);
  const uint32_t slot = uint32_t(ctx->r[31]);
  if (slot > 5) {
    return;
  }
  const uint32_t frame =
      xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(player + 0x6F4));
  if (!frame) {
    return;
  }
  const uint8_t* f = ctx->TranslateVirtual(frame);
  const uint8_t* joint = f + 0x30 + slot * 0x1C0 + 0x20 + 11 * 0x10;
  XELOGI("JD4 hand: t={} frame={} slot={} game ({:.3f}, {:.3f}, {:.3f}) "
         "skeleton ({:.3f}, {:.3f}, {:.3f})",
         xe::load_and_swap<uint64_t>(f), xe::load_and_swap<uint32_t>(f + 8),
         slot, ctx->f[29], ctx->f[30], ctx->f[31],
         xe::load_and_swap<float>(joint), xe::load_and_swap<float>(joint + 4),
         xe::load_and_swap<float>(joint + 8));
}

// Just Dance 4 diagnostic (kinect_debug_jd4_cursor): 82840ED0 updates the
// menu cursor object (r3 = *(0x82DE9B44) + 0x5DC) once per game frame.
void Jd4CursorDebugHook(cpu::ppc::PPCContext_s* ctx, uint32_t address) {
  static std::atomic<int> status{0};
  if (status == 2) {
    return;
  }
  if (status == 0) {
    status = xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(address)) ==
                         0x7D8802A6 &&
                     xe::load_and_swap<uint32_t>(
                         ctx->TranslateVirtual(address + 4)) == 0x4BB63801
                 ? 1
                 : 2;
    if (status == 2) {
      return;
    }
  }
  static std::atomic<int64_t> last_ms{0};
  static std::atomic<uint32_t> lines{0};
  const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  if (now - last_ms.load() < 50 || lines.load() >= 1200) {
    return;
  }
  // Only while a player drives the cursor (slot != -1).
  if (xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(uint32_t(ctx->r[3]))) ==
      0xFFFFFFFFu) {
    return;
  }
  last_ms = now;
  ++lines;
  const uint32_t object_address = uint32_t(ctx->r[3]);
  const uint8_t* object = ctx->TranslateVirtual(object_address);
  static const char digits[] = "0123456789ABCDEF";
  auto to_hex = [](const uint8_t* data, uint32_t length) {
    std::string hex;
    hex.reserve(length * 2);
    for (uint32_t i = 0; i < length; ++i) {
      hex += digits[data[i] >> 4];
      hex += digits[data[i] & 15];
    }
    return hex;
  };
  XELOGI("JD4 cursor: t={} {}", now, to_hex(object, 0x120));
  // The body state the cursor is scaled by: the owner (cursor - 0x5DC) keeps
  // per-slot reference points (+0x10E0), tracking IDs (+0x1140) and torso
  // lengths (+0x1158); and the skeleton frame it reads (*(owner + 0x6F4)):
  // its header and the cursor's slot.
  const uint32_t owner = object_address - 0x5DC;
  const uint32_t frame =
      xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(owner + 0x6F4));
  uint32_t slot = xe::load_and_swap<uint32_t>(object);
  if (!frame) {
    return;
  }
  const uint8_t* f = ctx->TranslateVirtual(frame);
  if (slot > 5) {
    slot = 0;
    for (uint32_t i = 0; i < 6; ++i) {
      if (xe::load_and_swap<uint32_t>(f + 0x30 + i * 0x1C0) == 2) {
        slot = i;
        break;
      }
    }
  }
  XELOGI("JD4 body: t={} slot={} {} {} {}", now, slot,
         to_hex(ctx->TranslateVirtual(owner + 0x10E0), 0x120), to_hex(f, 0x30),
         to_hex(f + 0x30 + slot * 0x1C0, 0x1C0));
}

// Just Dance 4 smooths 9 joints per player (8298CFA0 -> filter 82A95560).
// For the hands the filter's "unreliable" argument (r6, from r28) comes from
// the depth/colour hand tracker's flag at player + 0x1DC (left) / 0x25C
// (right); while it is set, the filter resets a 0.3 s timer that limits the
// joint to 2 m/s and 0.1 m/s^2, so a hand that is always flagged crawls.
// At 8298D2EC: r30 = joint, r28 = unreliable, r6 = the flag read for hands,
// f30 = time step (s).
void Jd4FilterDebugHook(cpu::ppc::PPCContext_s* ctx, uint32_t address) {
  static std::atomic<int> status{0};
  if (status == 2) {
    return;
  }
  if (status == 0) {
    status = xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(address)) ==
                     0x7F86E378  // or r6, r28, r28
                 ? 1
                 : 2;
    if (status == 2) {
      XELOGW("JD4 filter: unexpected code at {:08X}; not logging", address);
      return;
    }
  }
  static std::mutex mutex;
  static auto window_start = std::chrono::steady_clock::now();
  static uint32_t hand_calls[2], hand_flagged[2], other_calls, other_flagged;
  static uint32_t flag_values[2][4];  // 0, 1, 2, other
  static double dt_sum, dt_min = 1e9, dt_max;
  static uint32_t dt_count;
  std::lock_guard<std::mutex> lock(mutex);
  const uint32_t joint = uint32_t(ctx->r[30]);
  const bool unreliable = uint32_t(ctx->r[28]) != 0;
  if (joint == 7 || joint == 11) {
    const int hand = joint == 11;
    ++hand_calls[hand];
    hand_flagged[hand] += unreliable;
    const uint32_t value = uint32_t(ctx->r[6]);
    ++flag_values[hand][value < 3 ? value : 3];
  } else {
    ++other_calls;
    other_flagged += unreliable;
  }
  if (joint == 11) {
    const double dt = ctx->f[30];
    dt_sum += dt;
    dt_min = std::min(dt_min, dt);
    dt_max = std::max(dt_max, dt);
    ++dt_count;
  }
  const auto now = std::chrono::steady_clock::now();
  if (now - window_start < std::chrono::seconds(5)) {
    return;
  }
  window_start = now;
  auto percent = [](uint32_t part, uint32_t total) {
    return total ? 100.0 * part / total : 0.0;
  };
  XELOGI("JD4 filter: hands flagged unreliable: left {:.0f}% of {} (flag "
         "0/1/2/other {}/{}/{}/{}), right {:.0f}% of {} ({}/{}/{}/{}); other "
         "joints not tracked {:.0f}% of {}; time step {:.1f} ms ({:.1f} to "
         "{:.1f})",
         percent(hand_flagged[0], hand_calls[0]), hand_calls[0],
         flag_values[0][0], flag_values[0][1], flag_values[0][2],
         flag_values[0][3], percent(hand_flagged[1], hand_calls[1]),
         hand_calls[1], flag_values[1][0], flag_values[1][1],
         flag_values[1][2], flag_values[1][3],
         percent(other_flagged, other_calls), other_calls,
         dt_count ? dt_sum / dt_count * 1000 : 0.0,
         dt_count ? dt_min * 1000 : 0.0, dt_count ? dt_max * 1000 : 0.0);
  hand_calls[0] = hand_calls[1] = hand_flagged[0] = hand_flagged[1] = 0;
  other_calls = other_flagged = dt_count = 0;
  std::memset(flag_values, 0, sizeof(flag_values));
  dt_sum = dt_max = 0;
  dt_min = 1e9;
}

// kinect_jd4_skeleton_hands only applies to the bridges: with a real Kinect
// the game's own hand tracker sees real images and works as on a console
// (scoring felt better with it).
bool Jd4SkeletonHands() {
  return cvars::kinect_jd4_skeleton_hands && !cvars::allow_kinect_camera;
}

// kinect_jd4_hand_flag: at 8298D254 / 8298D2A4 (lwz r6, 0x1DC / 0x25C(r11),
// r11 = player state), overwrite the flag in memory before it is read. With
// title update 1 the same code is at 8298D944 / 8298D994.
constexpr uint32_t kJd4HandFlagSites[] = {0x8298D254, 0x8298D2A4, 0x8298D944,
                                          0x8298D994};
void Jd4HandFlagHook(cpu::ppc::PPCContext_s* ctx, uint32_t address) {
  size_t site = 0;
  while (site < std::size(kJd4HandFlagSites) &&
         kJd4HandFlagSites[site] != address) {
    ++site;
  }
  if (site == std::size(kJd4HandFlagSites)) {
    return;
  }
  // Even sites are the left hand (flag at +0x1DC), odd ones the right.
  const bool left = (site & 1) == 0;
  const uint32_t expected = left ? 0x80CB01DC : 0x80CB025C;
  static std::atomic<int> status[std::size(kJd4HandFlagSites)] = {};
  std::atomic<int>& state = status[site];
  if (state == 2) {
    return;
  }
  if (state == 0) {
    state = xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(address)) ==
                    expected
                ? 1
                : 2;
    if (state == 2) {
      XELOGW("JD4 hand flag: unexpected code at {:08X}; not changing it",
             address);
      return;
    }
  }
  const uint32_t field = uint32_t(ctx->r[11]) + (expected & 0xFFFF);
  if (Jd4SkeletonHands()) {
    // The tracker's position for this hand sits 0x5C bytes before its flag
    // (player + 0x180 / 0x200); r29 = skeleton slot, r3 = the skeleton frame
    // the filter reads the other joints from. Copy the frame's hand.
    const uint32_t slot = uint32_t(ctx->r[29]);
    const uint32_t frame = uint32_t(ctx->r[3]);
    if (frame && slot < 6) {
      const uint32_t joint = left ? 7 : 11;
      std::memcpy(ctx->TranslateVirtual(field - 0x5C),
                  ctx->TranslateVirtual(frame + 0x30 + slot * 0x1C0 + 0x20 +
                                        joint * 0x10),
                  12);
      xe::store_and_swap<uint32_t>(ctx->TranslateVirtual(field), 0);
    }
    return;
  }
  xe::store_and_swap<uint32_t>(ctx->TranslateVirtual(field),
                               uint32_t(cvars::kinect_jd4_hand_flag));
}

// Just Dance 2014: 82C51600 and 82C51888 map a joint into the colour image
// (NuiTransformSkeletonToDepthImage, then the runtime's
// NuiImageGetColorPixelCoordinatesFromDepthPixel at 82A79668) to place the
// face crop. Right after the second call: r3 = HRESULT, the depth pixel and
// value are in registers / on the stack, the colour pixel on the stack.
void Jd2014ColourMappingHook(cpu::ppc::PPCContext_s* ctx, uint32_t address) {
  struct Site {
    uint32_t address;
    uint32_t x_register, y_register;
    uint32_t colour_x_offset, colour_y_offset;
  };
  static constexpr Site kSites[] = {
      {0x82C516E0, 30, 31, 0x68, 0x6C},
      // Here the colour X goes to sp+0x60 and Y to sp+0x50 (r8, r9).
      {0x82C51938, 29, 30, 0x60, 0x50},
  };
  const Site* site = nullptr;
  for (const Site& s : kSites) {
    if (s.address == address) {
      site = &s;
    }
  }
  static std::atomic<int> status[2] = {0, 0};
  std::atomic<int>& state = status[site == &kSites[0] ? 0 : 1];
  if (!site || state == 2) {
    return;
  }
  if (state == 0) {
    // The call just before: bl 0x82A79668.
    const uint32_t call = xe::load_and_swap<uint32_t>(
        ctx->TranslateVirtual(address - 4));
    const int32_t offset = int32_t(call << 6) >> 6 & ~3;
    state = (call >> 26) == 18 && (call & 3) == 1 &&
                    address - 4 + uint32_t(offset) == 0x82A79668
                ? 1
                : 2;
    if (state == 2) {
      XELOGW("JD2014 colour mapping: unexpected code at {:08X}", address);
      return;
    }
  }
  const int32_t result = int32_t(uint32_t(ctx->r[3]));
  const uint32_t stack = uint32_t(ctx->r[1]);
  const uint32_t x = uint32_t(ctx->r[site->x_register]);
  const uint32_t y = uint32_t(ctx->r[site->y_register]);
  const uint16_t depth =
      xe::load_and_swap<uint16_t>(ctx->TranslateVirtual(stack + 0x58));
  uint8_t* colour_x = ctx->TranslateVirtual(stack + site->colour_x_offset);
  uint8_t* colour_y = ctx->TranslateVirtual(stack + site->colour_y_offset);
  const int32_t runtime_x = int32_t(xe::load_and_swap<uint32_t>(colour_x));
  const int32_t runtime_y = int32_t(xe::load_and_swap<uint32_t>(colour_y));
  static std::atomic<uint32_t> calls{0};
  const uint32_t n = calls.fetch_add(1);
  if (n < 10 || n % 600 == 0) {
    XELOGI("JD2014 colour mapping at {:08X}: depth pixel ({}, {}) value {} "
           "(= {} mm) -> runtime colour pixel ({}, {}), result {:08X}",
           address, int32_t(x), int32_t(y), depth, depth >> 3, runtime_x,
           runtime_y, uint32_t(result));
  }
  // Points just outside the depth image (the top of the face box above a
  // head near the top edge) are scaled the same way.
  const int32_t depth_x = int32_t(x), depth_y = int32_t(y);
  if (cvars::kinect_jd2014_colour_mapping != "nominal" || result < 0 ||
      depth_x < -320 || depth_x > 640 || depth_y < -240 || depth_y > 480) {
    return;
  }
  // Nominal Kinect v1 intrinsics: depth focal length 285.63 px at 320x240,
  // colour 531.15 px at 640x480, both centred (the cameras' 2.5 cm offset is
  // a few pixels at dancing distance). The colour camera sees a wider field,
  // so a point near the top of the depth image is lower in the colour image
  // than plain x2 scaling puts it.
  constexpr double kScale = 531.15 / 285.63;
  const int32_t mapped_x = int32_t(320.0 + (depth_x - 160) * kScale);
  // Shift down by kinect_jd2014_face_offset_m at the point's distance.
  const double distance_m = (depth >> 3) * 0.001;
  const double offset_px =
      distance_m > 0.3 ? 531.15 * cvars::kinect_jd2014_face_offset_m / distance_m
                       : 0.0;
  const int32_t mapped_y =
      int32_t(240.0 + (depth_y - 120) * kScale + offset_px);
  xe::store_and_swap<uint32_t>(colour_x, uint32_t(mapped_x));
  xe::store_and_swap<uint32_t>(colour_y, uint32_t(mapped_y));
  if (n < 10 || n % 600 == 0) {
    XELOGI("JD2014 colour mapping: using colour pixel ({}, {})", mapped_x,
           mapped_y);
  }
}

// Just Dance 2015 and 2017: each player's face in the player slots is the
// game's own crop of the camera image around where the runtime's
// NuiImageGetColorPixelCoordinatesFromDepthPixel puts their head. These
// hooks sit right after its two callers:
// - in the skeleton-point-to-colour function (JD2017 826C9030, reached
//   through an interface at 82DA2EA0): depth pixel in r29 / r30, packed
//   depth (mm << 3) at sp+0x58, colour X at sp+0x50, Y at sp+0x68, the
//   caller's return address at sp+0xA8;
// - in a loop over pairs of points (the box around each head: its centre
//   and its top edge): depth pixel at sp+0x5C / sp+0x64, the point's
//   distance (float, metres) at sp+0x98, colour X at sp+0x54, Y at sp+0x50.
struct JdColourMappingSite {
  const char* title;
  uint32_t address;
  uint32_t mapper;  // the bl just before calls this
  bool single;      // the first kind above
};

constexpr JdColourMappingSite kJdColourMappingSites[] = {
    {"Just Dance 2017", 0x826C90D0, 0x82B74940, true},
    {"Just Dance 2017", 0x82836FD4, 0x82B74940, false},
    {"Just Dance 2015", 0x82CD6DE8, 0x82AE3A98, true},
    {"Just Dance 2015", 0x82CF016C, 0x82AE3A98, false},
};

void JdColourMappingHook(cpu::ppc::PPCContext_s* ctx, uint32_t address) {
  constexpr size_t kSites = std::size(kJdColourMappingSites);
  size_t index = 0;
  while (index < kSites && kJdColourMappingSites[index].address != address) {
    ++index;
  }
  if (index == kSites) {
    return;
  }
  const JdColourMappingSite& site = kJdColourMappingSites[index];
  static std::atomic<int> status[kSites] = {};
  std::atomic<int>& state = status[index];
  if (state == 2) {
    return;
  }
  if (state == 0) {
    const uint32_t call =
        xe::load_and_swap<uint32_t>(ctx->TranslateVirtual(address - 4));
    const int32_t offset = int32_t(call << 6) >> 6 & ~3;
    state = (call >> 26) == 18 && (call & 3) == 1 &&
                    address - 4 + uint32_t(offset) == site.mapper
                ? 1
                : 2;
    if (state == 2) {
      return;  // another title
    }
    XELOGI("{} colour mapping: hook at {:08X} active ({})", site.title,
           address, cvars::kinect_jd_colour_mapping);
  }
  // Mode and face offset: the bridge's settings win in "auto".
  bool whole_image = false;
  float bridge_offset = 0.0f;
  const bool have_bridge =
      cvars::kinect_skeleton_udp_port &&
      SkeletonReceiver::Get().FaceSettings(&whole_image, &bridge_offset, 3.0);
  const std::string& mode = cvars::kinect_jd_colour_mapping;
  const bool use_nominal =
      mode == "nominal" || (mode == "auto" && have_bridge && whole_image);
  const double offset_m =
      have_bridge ? double(bridge_offset) : cvars::kinect_jd_face_offset_m;

  const int32_t result = int32_t(uint32_t(ctx->r[3]));
  const uint32_t stack = uint32_t(ctx->r[1]);
  auto load = [&](uint32_t offset) {
    return int32_t(xe::load_and_swap<uint32_t>(
        ctx->TranslateVirtual(stack + offset)));
  };
  int32_t depth_x, depth_y;
  uint32_t mm;
  uint32_t colour_x_offset, colour_y_offset;
  if (site.single) {
    depth_x = int32_t(uint32_t(ctx->r[29]));
    depth_y = int32_t(uint32_t(ctx->r[30]));
    mm = xe::load_and_swap<uint16_t>(ctx->TranslateVirtual(stack + 0x58)) >> 3;
    colour_x_offset = 0x50;
    colour_y_offset = 0x68;
  } else {
    depth_x = load(0x5C);
    depth_y = load(0x64);
    const float z =
        xe::load_and_swap<float>(ctx->TranslateVirtual(stack + 0x98));
    mm = z > 0.0f && z < 10.0f ? uint32_t(z * 1000.0f) : 0;
    colour_x_offset = 0x54;
    colour_y_offset = 0x50;
  }
  const int32_t runtime_x = load(colour_x_offset);
  const int32_t runtime_y = load(colour_y_offset);
  const bool apply = use_nominal && result >= 0 && mm > 300 &&
                     depth_x >= -320 && depth_x <= 640 && depth_y >= -240 &&
                     depth_y <= 480;
  // Nominal Kinect v1 intrinsics, as for Just Dance 2014 above, moved down
  // by the face offset at the point's distance.
  constexpr double kScale = 531.15 / 285.63;
  const double offset_px = mm > 300 ? 531.15 * offset_m / (mm * 0.001) : 0.0;
  const int32_t mapped_x = int32_t(320.0 + (depth_x - 160) * kScale);
  const int32_t mapped_y =
      int32_t(240.0 + (depth_y - 120) * kScale + offset_px);
  if (apply) {
    xe::store_and_swap<uint32_t>(ctx->TranslateVirtual(stack + colour_x_offset),
                                 uint32_t(mapped_x));
    xe::store_and_swap<uint32_t>(ctx->TranslateVirtual(stack + colour_y_offset),
                                 uint32_t(mapped_y));
  }
  static std::atomic<uint32_t> calls[kSites] = {};
  const uint32_t n = calls[index].fetch_add(1);
  if (n < 10 || n % 900 == 0) {
    XELOGI("{} colour mapping #{} at {:08X} (for {:08X}): depth pixel ({}, "
           "{}) {} mm -> runtime colour pixel ({}, {}) result {:08X}; "
           "nominal ({}, {}){}",
           site.title, n, address, site.single ? uint32_t(load(0xA8)) : 0u,
           depth_x, depth_y, mm, runtime_x, runtime_y, uint32_t(result),
           mapped_x, mapped_y, apply ? " used" : "");
  }
}

// jd4_debug_print: Just Dance 4's debug print (printf-style, format in r3,
// arguments from r4) is a bare blr in the retail game; log the messages.
// Only %s, %c and integer conversions are expanded (floats would be in the
// FP registers).
void Jd4DebugPrintHook(cpu::ppc::PPCContext_s* ctx, uint32_t address) {
  static std::atomic<int> state = 0;  // 0 unknown, 1 Just Dance 4, 2 not
  if (state == 0) {
    const uint32_t blr = xe::load_and_swap<uint32_t>(
        ctx->TranslateVirtual(0x82211F20));
    // addi r3, r11, 0x2B94 ("[DLC] MBF registration : %s") at 82A55C28.
    const uint32_t check = xe::load_and_swap<uint32_t>(
        ctx->TranslateVirtual(0x82A55C28));
    state = blr == 0x4E800020 && check == 0x386B2B94 ? 1 : 2;
    if (state == 1) {
      XELOGI("Just Dance 4: logging the game's debug messages");
    }
  }
  if (state != 1) {
    return;
  }
  // Arguments aren't always what the format says (some callers pass other
  // things), so only read guest memory that is actually readable.
  const auto readable = [](uint32_t guest) {
    if (guest < 0x10000) {
      return false;
    }
    Memory* memory = kernel_state()->memory();
    BaseHeap* heap = memory->LookupHeap(guest);
    uint32_t protect = 0;
    return heap && heap->QueryProtect(guest, &protect) &&
           (protect & kMemoryProtectRead);
  };
  const auto guest_string = [ctx, &readable](uint32_t guest, size_t limit) {
    std::string out;
    if (!guest) {
      return std::string("(null)");
    }
    if (!readable(guest)) {
      return std::string("(bad pointer)");
    }
    const char* p = ctx->TranslateVirtual<const char*>(guest);
    while (out.size() < limit) {
      const uint32_t at = guest + uint32_t(out.size());
      // Crossing into the next page: check that one too.
      if ((at & 0xFFF) == 0 && at != guest && !readable(at)) {
        break;
      }
      const char c = p[out.size()];
      if (!c) {
        break;
      }
      out.push_back(c);
    }
    return out;
  };
  const std::string format = guest_string(uint32_t(ctx->r[3]), 512);
  std::string text;
  uint32_t next_arg = 4;
  const auto arg = [&]() -> uint32_t {
    return next_arg <= 10 ? uint32_t(ctx->r[next_arg++]) : 0;
  };
  for (size_t i = 0; i < format.size(); ++i) {
    if (format[i] != '%') {
      text.push_back(format[i]);
      continue;
    }
    size_t j = i + 1;
    std::string spec = "%";
    while (j < format.size() && std::strchr("-+ #0123456789.lhI", format[j])) {
      if (format[j] != 'l' && format[j] != 'h' && format[j] != 'I') {
        spec.push_back(format[j]);
      }
      ++j;
    }
    if (j >= format.size()) {
      break;
    }
    const char conversion = format[j];
    char buffer[64];
    switch (conversion) {
      case '%':
        text.push_back('%');
        break;
      case 's':
        text += guest_string(arg(), 256);
        break;
      case 'c':
        text.push_back(char(arg()));
        break;
      case 'd':
      case 'i':
      case 'u':
      case 'x':
      case 'X':
      case 'o':
        spec.push_back(conversion);
        std::snprintf(buffer, sizeof(buffer), spec.c_str(), arg());
        text += buffer;
        break;
      case 'p':
        std::snprintf(buffer, sizeof(buffer), "%08X", arg());
        text += buffer;
        break;
      default:
        text += "?";
        break;
    }
    i = j;
  }
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
    text.pop_back();
  }
  static std::atomic<uint32_t> logged = 0, dlc_logged = 0;
  const bool dlc = text.find("[DLC]") != std::string::npos ||
                   text.find("[BIGfile]") != std::string::npos;
  if ((dlc ? dlc_logged : logged).fetch_add(1) < 4000) {
    XELOGI("Just Dance 4 says: {}", text);
  }
}

void InstallTitleWorkarounds() {
  cpu::ppc::RegisterGuestHook(0x82C516E0, Jd2014ColourMappingHook);
  cpu::ppc::RegisterGuestHook(0x82C51938, Jd2014ColourMappingHook);
  for (const JdColourMappingSite& site : kJdColourMappingSites) {
    cpu::ppc::RegisterGuestHook(site.address, JdColourMappingHook);
  }
  if (cvars::kinect_debug_jd4_filter) {
    cpu::ppc::RegisterGuestHook(0x8298D2EC, Jd4FilterDebugHook);
  }
  if (cvars::kinect_jd4_hand_flag >= 0 || Jd4SkeletonHands()) {
    for (uint32_t site : kJd4HandFlagSites) {
      cpu::ppc::RegisterGuestHook(site, Jd4HandFlagHook);
    }
  }
  if (cvars::kinect_debug_jd4_cursor) {
    cpu::ppc::RegisterGuestHook(0x82840ED0, Jd4CursorDebugHook);
  }
  if (cvars::jd4_debug_print) {
    cpu::ppc::RegisterGuestHook(0x82211F20, Jd4DebugPrintHook);
  }
  if (cvars::kinect_debug_jd4_hand) {
    cpu::ppc::RegisterGuestHook(0x82835D28, Jd4HandDebugHook);
  }
  if (!cvars::kinect_title_workarounds) {
    return;
  }
  for (const DeliberateFaultFix& fix : kDeliberateFaultFixes) {
    cpu::ppc::RegisterGuestHook(fix.fault_address, DeliberateFaultHook);
  }
}

}  // namespace
// --- Finding the hooks in any title -------------------------------------------
// The hooks above sit in Microsoft's NUI runtime, which every Kinect title
// links its own copy of: the code is the same in each NUI version, only its
// address differs (and moves with title updates). So when a title loads, look
// for the code itself, and hook where it is found. Matching code was checked
// against Kinect Adventures, Just Dance 3, 4 (with and without its title
// update), 2015, 2017 and Dance Central 3: one match each, at the known
// address.

struct CodePattern {
  uint32_t mask;
  uint32_t value;
};

// NuiSkeletonGetNextFrame, around the memcpy of the 0xAB0-byte frame into the
// caller's buffer; the hook is the instruction after the memcpy (index 9).
constexpr CodePattern kSkeletonCopyPattern[] = {
    {0xFFFFFFFF, 0x1D6B0AE0},  // mulli r11, r11, 0xAE0
    {0xFFFFFFFF, 0x7FCB5214},  // add r30, r11, r10
    {0xFFFFFFFF, 0x7D6B502E},  // lwzx r11, r11, r10
    {0xFFFFFFFF, 0x2F0B0001},  // cmpwi cr6, r11, 1
    {0xFFFF0003, 0x409A0000},  // bne cr6, ...
    {0xFFFFFFFF, 0x38A00AB0},  // li r5, 0xAB0
    {0xFFFFFFFF, 0x389E0020},  // addi r4, r30, 0x20
    {0xFC1F07FF, 0x7C030378},  // mr r3, rFrame
    {0xFC000003, 0x48000001},  // bl memcpy
    {0xFC1FFFFF, 0x901E0000},  // stw rN, 0(r30)  <- hook
    {0xFFFFFFFF, 0x7D6C42E6},  // mftb r11
    {0xFFFFFFFF, 0x815E0AD0},  // lwz r10, 0xAD0(r30)
    {0xFFFFFFFF, 0x5569003E},  // slwi r9, r11, 0
    {0xFFFFFFFF, 0x3D00000F},  // lis r8, 0xF
    {0xFFFFFFFF, 0x7CEA4850},  // subf r7, r10, r9
};
constexpr size_t kSkeletonCopyHookIndex = 9;

// NuiImageStreamGetNextFrame, where it hands the frame (r9) to the caller;
// the hook is the instruction after "stw r9, 0(r28)" (index 15). Not in Kinect
// Adventures' older runtime.
constexpr CodePattern kImageFramePattern[] = {
    {0xFFFFFFFF, 0x93AB0038}, {0xFFFFFFFF, 0x392B0008},
    {0xFFFFFFFF, 0x81410050}, {0xFFFFFFFF, 0x7F63DB78},
    {0xFFFFFFFF, 0x810A0030}, {0xFFFFFFFF, 0x394A0030},
    {0xFFFFFFFF, 0x910B0000}, {0xFFFFFFFF, 0x914B0004},
    {0xFFFFFFFF, 0x91680004}, {0xFFFFFFFF, 0x916A0000},
    {0xFFFFFFFF, 0x81610050}, {0xFFFFFFFF, 0x814B002C},
    {0xFFFFFFFF, 0x38EA0001}, {0xFFFFFFFF, 0x90EB002C},
    {0xFFFFFFFF, 0x913C0000},  // stw r9, 0(r28)
    {0xFC000003, 0x48000001},  // bl  <- hook
    {0xFC000003, 0x48000000},  // b
};
constexpr size_t kImageFrameHookIndex = 15;

template <size_t N>
std::vector<uint32_t> FindCode(const uint8_t* image, uint32_t base,
                               uint32_t size, const CodePattern (&pattern)[N]) {
  std::vector<uint32_t> found;
  const uint32_t words = size / 4;
  for (uint32_t i = 0; i + N <= words; ++i) {
    if ((xe::load_and_swap<uint32_t>(image + i * 4) & pattern[0].mask) !=
        pattern[0].value) {
      continue;
    }
    bool match = true;
    for (size_t k = 1; k < N && match; ++k) {
      match = (xe::load_and_swap<uint32_t>(image + (i + k) * 4) &
               pattern[k].mask) == pattern[k].value;
    }
    if (match) {
      found.push_back(base + i * 4);
    }
  }
  return found;
}

// Harmonix's Milo engine (Dance Central 2 and 3) runs its main loop from the
// unhandled-exception filter: it registers the filter (lis r9 / addi r3, r9
// = the callback; bl SetUnhandledExceptionFilter), then faults on purpose
// with "li r10, 1; stw r10, 0(0)". Find that, so DeliberateFaultHook calls
// the callback instead of the title crashing.
void ScanForDeliberateFaults(const uint8_t* image, uint32_t base,
                             uint32_t size) {
  const auto word = [&](uint32_t address) {
    return xe::load_and_swap<uint32_t>(image + (address - base));
  };
  for (uint32_t offset = 8 * 4; offset + 4 <= size; offset += 4) {
    const uint32_t address = base + offset;
    if (word(address) != 0x91400000 /* stw r10, 0(0) */ ||
        word(address - 4) != 0x39400001 /* li r10, 1 */) {
      continue;
    }
    uint32_t high = 0, low = 0;
    bool have_high = false, have_low = false, have_call = false;
    for (uint32_t back = 2; back <= 8; ++back) {
      const uint32_t w = word(address - back * 4);
      if ((w >> 16) == 0x3869 && !have_low) {  // addi r3, r9, low
        low = w & 0xFFFF;
        have_low = true;
      } else if ((w >> 16) == 0x3D20 && !have_high) {  // lis r9, high
        high = w & 0xFFFF;
        have_high = true;
      } else if ((w & 0xFC000003) == 0x48000001) {  // bl
        have_call = true;
      }
    }
    if (!have_high || !have_low || !have_call) {
      continue;
    }
    const uint32_t handler = (high << 16) + uint32_t(int32_t(int16_t(low)));
    if (handler < base || handler >= base + size ||
        word(handler) != 0x7D8802A6 /* mflr r12 */) {
      continue;
    }
    bool known = false;
    for (const DeliberateFaultFix& fix : active_fault_fixes) {
      known |= fix.fault_address == address;
    }
    if (known || active_fault_fixes.size() >= 8) {
      continue;
    }
    active_fault_fixes.push_back(
        {"this title (found by code scan)", address, 0x91400000, handler});
    cpu::ppc::RegisterGuestHook(address, DeliberateFaultHook);
    XELOGI("Found a deliberate fault at {:08X} that runs {:08X} from the "
           "exception filter; it will be called directly",
           address, handler);
  }
}

void ScanTitleForHooks(Memory* memory, uint32_t base, uint32_t size) {
  if (!size) {
    return;
  }
  const uint8_t* image = memory->TranslateVirtual(base);
  ScanForDeliberateFaults(image, base, size);
  if (!cvars::kinect_auto_hooks || !skeleton_hooks_installed) {
    return;
  }
  const auto word = [&](uint32_t address) {
    return xe::load_and_swap<uint32_t>(image + (address - base));
  };
  const auto in_image = [&](uint32_t address) {
    return address >= base + 4 && address < base + size;
  };

  for (uint32_t start : FindCode(image, base, size, kSkeletonCopyPattern)) {
    const uint32_t address = start + uint32_t(kSkeletonCopyHookIndex) * 4;
    const uint32_t mr = word(address - 8);
    const uint32_t frame_register = (mr >> 21) & 31;
    if (((mr >> 11) & 31) != frame_register) {
      continue;  // not "mr r3, rX"
    }
    bool known = false;
    for (const SkeletonHook& hook : active_skeleton_hooks) {
      known |= hook.address == address;
    }
    if (known) {
      XELOGI("NUI skeleton: found NuiSkeletonGetNextFrame's frame copy at "
             "{:08X} (a known hook)",
             address);
      continue;
    }
    if (active_skeleton_hooks.size() >= kMaxSkeletonHooks) {
      break;
    }
    active_skeleton_hooks.push_back(
        {"this title (found by code scan)", address, frame_register,
         word(address - 4)});
    cpu::ppc::RegisterGuestHook(address, SkeletonHookHandler);
    XELOGI("NUI skeleton: found NuiSkeletonGetNextFrame's frame copy at "
           "{:08X}, frame in r{}; hooked",
           address, frame_register);
  }

  if (!cvars::kinect_player_index) {
    return;
  }
  // Titles with a hook on their own copy of the image keep that one.
  for (const PlayerImageHook& hook : kKnownPlayerImageHooks) {
    if (in_image(hook.address) &&
        word(hook.address - 4) == hook.expected_previous) {
      return;
    }
  }
  for (uint32_t start : FindCode(image, base, size, kImageFramePattern)) {
    const uint32_t address = start + uint32_t(kImageFrameHookIndex) * 4;
    bool known = false;
    for (const ImageFrameHook& hook : active_image_frame_hooks) {
      known |= hook.address == address;
    }
    if (known) {
      XELOGI("NUI image frame: found NuiImageStreamGetNextFrame at {:08X} (a "
             "known hook)",
             address);
      continue;
    }
    if (active_image_frame_hooks.size() >= kMaxImageFrameHooks) {
      break;
    }
    active_image_frame_hooks.push_back(
        {"this title (found by code scan)", address, word(address - 4), 9,
         (1u << 0) | (1u << 7) | (1u << 4)});
    cpu::ppc::RegisterGuestHook(address, ImageFrameHookHandler);
    XELOGI("NUI image frame: found NuiImageStreamGetNextFrame at {:08X}; "
           "hooked",
           address);
  }
}

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

// Installs the emulated-skeleton guest hook (if enabled) while the kernel is
// set up, before any title code is translated.
void xe::kernel::xboxkrnl::RegisterUsbcamExports(
    xe::cpu::ExportResolver* export_resolver,
    xe::kernel::KernelState* kernel_state) {
  InstallSkeletonHook();
  InstallTitleWorkarounds();
}
