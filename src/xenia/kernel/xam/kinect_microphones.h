/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XAM_KINECT_MICROPHONES_H_
#define XENIA_KERNEL_XAM_KINECT_MICROPHONES_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "xenia/kernel/xobject.h"
#include "xenia/xbox.h"

namespace xe {
namespace kernel {
class XHostThread;
namespace xam {

// 16 ms of the Kinect's microphone array at 16 kHz, as the console's audio
// driver hands it to titles: the array's processed mix, and each of its four
// microphones.
struct MicrophoneWindow {
  static constexpr uint32_t kSamples = 256;
  static constexpr uint32_t kMicrophones = 4;
  int16_t processed[kSamples];
  int32_t microphones[kMicrophones][kSamples];
};

// A recording device on the PC (on Windows: WASAPI), 16 kHz mono 16-bit.
class HostMicrophone {
 public:
  // `match`: part of the device's name; empty = a device whose name has
  // "Kinect" in it, else the default recording device. Null if none opens.
  static std::unique_ptr<HostMicrophone> Open(const std::string& match,
                                              uint32_t sample_rate);
  virtual ~HostMicrophone() = default;
  virtual std::string name() const = 0;
  // Samples recorded since the last call, up to `max`.
  virtual size_t Read(int16_t* samples, size_t max) = 0;
};

// The Kinect's microphone array as a title's NUI runtime reads it. Ported
// from dcwestra/xenia-edge (nui_microphones.cc), itself after Davi Carlos's
// research into xam in kixenia (https://github.com/davicarlos/kixenia). The
// title asks for 16 ms windows with XamVoiceGetMicArrayAudioEx, and does its
// own echo cancellation in a (MEC) client the console's audio render driver
// calls every 256-sample frame of 5.33 ms, a tic; a window spans 3 tics and
// carries the tic it ends at. A host thread keeps the tic, completes a
// window request every 3 tics with the next window recorded (silence if
// none has come) and calls the MEC client. The windows come from a PC
// recording device: on Windows, the Kinect's microphone array is one once
// its drivers are installed (kinect_host_microphone).
class KinectMicrophones {
 public:
  static KinectMicrophones& Get();

  // False with kinect_microphone = "off": titles then see no array.
  // Otherwise there is one, silent if nothing can be recorded from.
  bool available() const;

  // XamVoiceGetMicArrayAudioEx: queues a window request, completed by
  // setting the X_KEVENT at event_ptr.
  X_STATUS ReadWindow(uint32_t request_ptr, uint32_t event_ptr);

  // XAudioRegisterRenderDriverMECClient and the rest.
  X_STATUS RegisterMecClient(uint32_t callback, uint32_t callback_arg,
                             uint32_t* handle);
  void UnregisterMecClient(uint32_t handle);
  uint16_t render_tic() const { return render_tic_; }

 private:
  struct Request {
    uint32_t request_ptr;
    uint32_t event_ptr;
  };

  void OpenSource();
  bool ReadSourceWindow(MicrophoneWindow* window);
  void StartThread();
  void ThreadMain();
  void Tick();
  void CompleteRequest(const Request& request, uint16_t tic);
  void LogLevels(const MicrophoneWindow& window, bool have_window);

  std::mutex mutex_;
  std::thread opener_;
  std::unique_ptr<HostMicrophone> source_;
  bool reading_ = false;
  std::deque<Request> requests_;
  uint32_t mec_callback_ = 0;
  uint32_t mec_callback_arg_ = 0;

  // On the thread only: samples not yet in a window, windows not yet read.
  std::vector<int16_t> pending_;
  std::deque<MicrophoneWindow> windows_;
  bool primed_ = false;
  uint32_t log_requests_ = 0;
  uint32_t log_windows_ = 0;
  int32_t log_peak_ = 0;
  uint32_t logs_ = 0;

  std::atomic<uint16_t> render_tic_ = 0;
  object_ref<XHostThread> thread_;
};

}  // namespace xam
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XAM_KINECT_MICROPHONES_H_
