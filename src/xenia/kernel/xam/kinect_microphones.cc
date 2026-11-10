/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/xam/kinect_microphones.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "xenia/base/byte_order.h"
#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/memory.h"
#include "xenia/base/platform.h"
#include "xenia/cpu/processor.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_threading.h"
#include "xenia/kernel/xevent.h"
#include "xenia/kernel/xthread.h"
#include "xenia/memory.h"

#if XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
// After windows.h: the property keys are defined here, not just declared.
#include <initguid.h>
#include <mmreg.h>
#include <audioclient.h>
#include <functiondiscoverykeys_devpkey.h>
#include <mmdeviceapi.h>
#endif

DEFINE_string(kinect_microphone, "auto",
              "The Kinect microphone array titles hear (voice commands): "
              "\"auto\" = a recording device on the PC (see "
              "kinect_host_microphone), silent if none opens; \"off\" = "
              "titles see no microphone array.",
              "Kernel");
DEFINE_string(kinect_host_microphone, "",
              "The PC recording device Kinect titles hear, by part of its "
              "name (e.g. \"Kinect\" or \"Headset\"). Empty = a device with "
              "\"Kinect\" in its name (the Kinect's microphone array, with its "
              "Windows drivers installed), else the default recording device.",
              "Kernel");

namespace xe {
namespace kernel {
namespace xam {

namespace {

// A window request, big-endian (xam 0x816FBCF8, from kixenia's notes): its
// size at +0, the buffers for the processed mix (16-bit) and the four
// microphones (32-bit), their sizes in bytes, and only when the size says the
// request has room for them (xam checks > 0x40 and > 0x64) the tic the window
// was captured at and the time.
constexpr uint32_t kRequestSize = 0x00;
constexpr uint32_t kRequestBuffers = 0x04;
constexpr uint32_t kRequestSizes = 0x18;
constexpr uint32_t kRequestTic = 0x40;
constexpr uint32_t kRequestTimeHigh = 0x44;
constexpr uint32_t kRequestTimeLow = 0x48;
constexpr uint32_t kRequestBufferCount = 1 + MicrophoneWindow::kMicrophones;
constexpr uint32_t kRequestSizeWithTic = 0x40;
constexpr uint32_t kRequestSizeWithTime = 0x64;

constexpr X_STATUS kStatusDeviceNotConnected = X_STATUS(0xC000009D);
constexpr uint32_t kSampleRate = 16000;
constexpr uint32_t kTicsPerWindow = 3;
constexpr auto kTicPeriod = std::chrono::nanoseconds(5333333);
constexpr uint32_t kMecClientHandle = 0x4D454301;
// Windows held back before the first is read (so arrival jitter doesn't
// leave the title without one), and the most kept before the oldest go.
constexpr size_t kPrimeWindows = 2;
constexpr size_t kMaxWindows = 6;

#if XE_PLATFORM_WIN32
template <typename T>
void SafeRelease(T*& p) {
  if (p) {
    p->Release();
    p = nullptr;
  }
}

std::string Utf8(const wchar_t* text) {
  if (!text) {
    return {};
  }
  const int size =
      WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
  if (size <= 1) {
    return {};
  }
  std::string out(size_t(size - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), size, nullptr,
                      nullptr);
  return out;
}

std::string Lower(std::string text) {
  for (char& c : text) {
    c = char(std::tolower(static_cast<unsigned char>(c)));
  }
  return text;
}

std::string DeviceName(IMMDevice* device) {
  IPropertyStore* properties = nullptr;
  std::string name;
  if (SUCCEEDED(device->OpenPropertyStore(STGM_READ, &properties))) {
    PROPVARIANT value;
    PropVariantInit(&value);
    if (SUCCEEDED(properties->GetValue(PKEY_Device_FriendlyName, &value)) &&
        value.vt == VT_LPWSTR) {
      name = Utf8(value.pwszVal);
    }
    PropVariantClear(&value);
  }
  SafeRelease(properties);
  return name;
}

// WASAPI, shared mode, with Windows converting to 16 kHz mono 16-bit.
class WasapiMicrophone : public HostMicrophone {
 public:
  ~WasapiMicrophone() override {
    if (client_) {
      client_->Stop();
    }
    SafeRelease(capture_);
    SafeRelease(client_);
  }

  bool Open(const std::string& match, uint32_t sample_rate) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IMMDeviceEnumerator* enumerator = nullptr;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                                CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                reinterpret_cast<void**>(&enumerator)))) {
      XELOGW("Kinect microphones: no Windows audio device enumerator");
      return false;
    }
    IMMDevice* device = FindDevice(enumerator, match);
    SafeRelease(enumerator);
    if (!device) {
      return false;
    }
    name_ = DeviceName(device);
    HRESULT hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                                  reinterpret_cast<void**>(&client_));
    SafeRelease(device);
    if (FAILED(hr)) {
      XELOGW("Kinect microphones: can't open \"{}\" ({:08X})", name_,
             uint32_t(hr));
      return false;
    }
    WAVEFORMATEX format = {};
    format.wFormatTag = 1;  // WAVE_FORMAT_PCM
    format.nChannels = 1;
    format.nSamplesPerSec = sample_rate;
    format.wBitsPerSample = 16;
    format.nBlockAlign = 2;
    format.nAvgBytesPerSec = sample_rate * 2;
    // 200 ms of buffer, in 100 ns units.
    hr = client_->Initialize(AUDCLNT_SHAREMODE_SHARED,
                             AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                                 AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                             2000000, 0, &format, nullptr);
    if (FAILED(hr)) {
      XELOGW("Kinect microphones: \"{}\" won't record 16 kHz mono ({:08X})",
             name_, uint32_t(hr));
      return false;
    }
    if (FAILED(client_->GetService(__uuidof(IAudioCaptureClient),
                                   reinterpret_cast<void**>(&capture_))) ||
        FAILED(client_->Start())) {
      XELOGW("Kinect microphones: can't start recording from \"{}\"", name_);
      return false;
    }
    return true;
  }

  std::string name() const override { return name_; }

  size_t Read(int16_t* samples, size_t max) override {
    size_t count = 0;
    UINT32 packet = 0;
    while (SUCCEEDED(capture_->GetNextPacketSize(&packet)) && packet) {
      BYTE* data = nullptr;
      UINT32 frames = 0;
      DWORD flags = 0;
      if (FAILED(capture_->GetBuffer(&data, &frames, &flags, nullptr,
                                     nullptr))) {
        break;
      }
      const size_t take = std::min<size_t>(frames, max - count);
      if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
        std::memset(samples + count, 0, take * sizeof(int16_t));
      } else {
        std::memcpy(samples + count, data, take * sizeof(int16_t));
      }
      count += take;
      capture_->ReleaseBuffer(frames);
      if (count == max) {
        break;
      }
    }
    return count;
  }

 private:
  static IMMDevice* FindDevice(IMMDeviceEnumerator* enumerator,
                               const std::string& match) {
    const std::string wanted = Lower(match.empty() ? "kinect" : match);
    IMMDeviceCollection* devices = nullptr;
    IMMDevice* found = nullptr;
    if (SUCCEEDED(enumerator->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE,
                                                 &devices))) {
      UINT count = 0;
      devices->GetCount(&count);
      for (UINT i = 0; i < count && !found; ++i) {
        IMMDevice* device = nullptr;
        if (FAILED(devices->Item(i, &device))) {
          continue;
        }
        if (Lower(DeviceName(device)).find(wanted) != std::string::npos) {
          found = device;
        } else {
          SafeRelease(device);
        }
      }
    }
    SafeRelease(devices);
    if (!found && match.empty()) {
      enumerator->GetDefaultAudioEndpoint(eCapture, eConsole, &found);
    }
    if (!found) {
      XELOGW("Kinect microphones: no recording device{}",
             match.empty() ? "" : " named like \"" + match + "\"");
    }
    return found;
  }

  std::string name_;
  IAudioClient* client_ = nullptr;
  IAudioCaptureClient* capture_ = nullptr;
};
#endif  // XE_PLATFORM_WIN32

}  // namespace

std::unique_ptr<HostMicrophone> HostMicrophone::Open(const std::string& match,
                                                     uint32_t sample_rate) {
#if XE_PLATFORM_WIN32
  auto microphone = std::make_unique<WasapiMicrophone>();
  if (microphone->Open(match, sample_rate)) {
    return microphone;
  }
#endif
  return nullptr;
}

KinectMicrophones& KinectMicrophones::Get() {
  static KinectMicrophones microphones;
  return microphones;
}

bool KinectMicrophones::available() const {
  return cvars::kinect_microphone != "off";
}

// On opener_: opening a device can take a moment. Requests get silence
// meanwhile.
void KinectMicrophones::OpenSource() {
  auto source =
      HostMicrophone::Open(cvars::kinect_host_microphone, kSampleRate);
  if (source) {
    XELOGI("Kinect microphones: recording from \"{}\"", source->name());
  } else {
    XELOGW("Kinect microphones: nothing to record from; titles hear silence");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  source_ = std::move(source);
}

X_STATUS KinectMicrophones::ReadWindow(uint32_t request_ptr,
                                       uint32_t event_ptr) {
  if (!available()) {
    return kStatusDeviceNotConnected;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!reading_) {
      reading_ = true;
      XELOGI("Kinect microphones: the title is listening");
      opener_ = std::thread([this]() { OpenSource(); });
      opener_.detach();
    }
    // A title that stops waiting would leave them piling up.
    if (requests_.size() < 64) {
      requests_.push_back({request_ptr, event_ptr});
    }
  }
  StartThread();
  return X_STATUS_SUCCESS;
}

X_STATUS KinectMicrophones::RegisterMecClient(uint32_t callback,
                                              uint32_t callback_arg,
                                              uint32_t* handle) {
  if (!available()) {
    return kStatusDeviceNotConnected;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    mec_callback_ = callback;
    mec_callback_arg_ = callback_arg;
  }
  XELOGI("Kinect microphones: echo cancellation client {:08X}", callback);
  *handle = kMecClientHandle;
  StartThread();
  return X_STATUS_SUCCESS;
}

void KinectMicrophones::UnregisterMecClient(uint32_t handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  mec_callback_ = 0;
  mec_callback_arg_ = 0;
}

void KinectMicrophones::StartThread() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (thread_) {
    return;
  }
  KernelState* state = kernel_state();
  thread_ = object_ref<XHostThread>(new XHostThread(
      state, 64 * 1024, 0,
      [this]() {
        ThreadMain();
        return 0;
      },
      state->GetSystemProcess()));
  thread_->set_can_debugger_suspend(true);
  thread_->set_name("Kinect Microphones");
  thread_->Create();
}

// The tic on a fixed schedule of 5.33 ms, as the console's audio render
// driver keeps it.
void KinectMicrophones::ThreadMain() {
#if XE_PLATFORM_WIN32
  CoInitializeEx(nullptr, COINIT_MULTITHREADED);
#endif
  auto deadline = std::chrono::steady_clock::now();
  while (true) {
    deadline += kTicPeriod;
    const auto now = std::chrono::steady_clock::now();
    if (deadline > now) {
      std::this_thread::sleep_until(deadline);
    } else if (now - deadline > kTicPeriod * 8) {
      // Far behind (stopped in a debugger): don't race to catch up.
      deadline = now;
    }
    Tick();
  }
}

// Every third tic completes the oldest window request, then the echo
// cancellation client runs, on this thread as on the console's driver thread.
void KinectMicrophones::Tick() {
  const uint16_t tic = uint16_t(render_tic_ + 1);
  render_tic_ = tic;
  uint32_t callback = 0;
  uint32_t callback_arg = 0;
  Request request = {};
  bool complete = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tic % kTicsPerWindow == 0 && !requests_.empty()) {
      request = requests_.front();
      requests_.pop_front();
      complete = true;
    }
    callback = mec_callback_;
    callback_arg = mec_callback_arg_;
  }
  if (complete) {
    CompleteRequest(request, tic);
  }
  if (callback) {
    uint64_t args[] = {callback_arg};
    kernel_state()->processor()->Execute(thread_->thread_state(), callback,
                                         args, xe::countof(args));
  }
}

// Whatever was recorded goes into 16 ms windows; the oldest is handed over.
bool KinectMicrophones::ReadSourceWindow(MicrophoneWindow* window) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!source_) {
      return false;
    }
    int16_t samples[MicrophoneWindow::kSamples * 8];
    size_t count;
    while ((count = source_->Read(samples, xe::countof(samples))) > 0) {
      pending_.insert(pending_.end(), samples, samples + count);
    }
  }
  size_t offset = 0;
  while (pending_.size() - offset >= MicrophoneWindow::kSamples) {
    MicrophoneWindow filled;
    for (uint32_t i = 0; i < MicrophoneWindow::kSamples; ++i) {
      const int16_t sample = pending_[offset + i];
      filled.processed[i] = sample;
      for (uint32_t m = 0; m < MicrophoneWindow::kMicrophones; ++m) {
        filled.microphones[m][i] = int32_t(sample) * 65536;
      }
    }
    windows_.push_back(filled);
    offset += MicrophoneWindow::kSamples;
  }
  pending_.erase(pending_.begin(), pending_.begin() + offset);
  // A recording clock a little faster than the title's: catch up.
  while (windows_.size() > kMaxWindows) {
    windows_.pop_front();
  }
  if (!primed_) {
    if (windows_.size() < kPrimeWindows) {
      return false;
    }
    primed_ = true;
  }
  if (windows_.empty()) {
    primed_ = false;
    return false;
  }
  *window = windows_.front();
  windows_.pop_front();
  return true;
}

// Every 10 seconds for the first minute, then every minute: how many
// windows came and the loudest sample, to see what the title hears.
void KinectMicrophones::LogLevels(const MicrophoneWindow& window,
                                  bool have_window) {
  for (int16_t sample : window.processed) {
    log_peak_ = std::max(log_peak_, std::abs(int32_t(sample)));
  }
  log_windows_ += have_window;
  const uint32_t windows_per_log = logs_ < 6 ? 625 : 3750;
  if (++log_requests_ < windows_per_log) {
    return;
  }
  ++logs_;
  XELOGI("Kinect microphones: {} of {} windows recorded, peak {:.1f} dBFS",
         log_windows_, log_requests_,
         log_peak_ ? 20.0 * std::log10(log_peak_ / 32768.0) : -120.0);
  log_requests_ = 0;
  log_windows_ = 0;
  log_peak_ = 0;
}

void KinectMicrophones::CompleteRequest(const Request& request, uint16_t tic) {
  Memory* memory = kernel_state()->memory();
  uint8_t* fields = memory->TranslateVirtual<uint8_t*>(request.request_ptr);
  MicrophoneWindow window;
  const bool have_window = ReadSourceWindow(&window);
  // A request is due every third tic whether a window has come or not;
  // without one it gets silence.
  if (!have_window) {
    std::memset(&window, 0, sizeof(window));
  }
  LogLevels(window, have_window);
  for (uint32_t buffer_index = 0; buffer_index < kRequestBufferCount;
       ++buffer_index) {
    const uint32_t buffer = xe::load_and_swap<uint32_t>(
        fields + kRequestBuffers + buffer_index * 4);
    const uint32_t size =
        xe::load_and_swap<uint32_t>(fields + kRequestSizes + buffer_index * 4);
    if (!buffer) {
      continue;
    }
    uint8_t* dest = memory->TranslateVirtual<uint8_t*>(buffer);
    if (buffer_index == 0) {
      const uint32_t count = std::min<uint32_t>(size / sizeof(int16_t),
                                                MicrophoneWindow::kSamples);
      for (uint32_t i = 0; i < count; ++i) {
        xe::store_and_swap<int16_t>(dest + i * sizeof(int16_t),
                                    window.processed[i]);
      }
    } else {
      const int32_t* samples = window.microphones[buffer_index - 1];
      const uint32_t count = std::min<uint32_t>(size / sizeof(int32_t),
                                                MicrophoneWindow::kSamples);
      for (uint32_t i = 0; i < count; ++i) {
        xe::store_and_swap<int32_t>(dest + i * sizeof(int32_t), samples[i]);
      }
    }
  }
  const uint32_t request_size =
      xe::load_and_swap<uint32_t>(fields + kRequestSize);
  if (request_size > kRequestSizeWithTic) {
    xe::store_and_swap<uint16_t>(fields + kRequestTic, tic);
  }
  if (request_size > kRequestSizeWithTime) {
    const uint64_t time = Clock::QueryGuestSystemTime();
    xe::store_and_swap<uint32_t>(fields + kRequestTimeHigh,
                                 uint32_t(time >> 32));
    xe::store_and_swap<uint32_t>(fields + kRequestTimeLow, uint32_t(time));
  }
  if (request.event_ptr) {
    xboxkrnl::xeKeSetEvent(
        memory->TranslateVirtual<X_KEVENT*>(request.event_ptr), 1, 0);
  }
}

}  // namespace xam
}  // namespace kernel
}  // namespace xe
