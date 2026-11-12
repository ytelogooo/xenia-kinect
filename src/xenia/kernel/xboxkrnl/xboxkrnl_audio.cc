/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <cstring>

#include "xenia/apu/audio_system.h"
#include "xenia/base/cvar.h"
#include "xenia/emulator.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/xam/kinect_microphones.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_private.h"
#include "xenia/xbox.h"

DEFINE_bool(kinect_echo_reference, true,
            "Give Kinect titles' echo cancellation the sound Xenia plays, so "
            "they can remove it from what the microphones hear (game music "
            "from speakers). Off: silence, nothing is removed.",
            "Kernel");

DECLARE_uint32(audio_flag);

namespace xe {
namespace kernel {
namespace xboxkrnl {

dword_result_t XAudioGetSpeakerConfig_entry(lpdword_t config_ptr) {
  kernel_state()->xconfig()->ReadSetting(
      XCONFIG_USER_CATEGORY,
      XCONFIG_USER_CATEGORY_ENTRIES::XCONFIG_USER_AUDIO_FLAGS, config_ptr);
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioGetSpeakerConfig, kAudio, kImplemented);

dword_result_t XAudioGetVoiceCategoryVolumeChangeMask_entry(
    lpunknown_t driver_ptr, lpdword_t out_ptr) {
  assert_true((driver_ptr.guest_address() & 0xFFFF0000) == 0x41550000);

  xe::threading::MaybeYield();

  // Checking these bits to see if any voice volume changed.
  // I think.
  *out_ptr = 0;
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT2(XAudioGetVoiceCategoryVolumeChangeMask, kAudio, kStub,
                         kHighFrequency);

dword_result_t XAudioGetVoiceCategoryVolume_entry(dword_t unk,
                                                  lpfloat_t out_ptr) {
  // Expects a floating point single. Volume %?
  *out_ptr = 1.0f;

  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT2(XAudioGetVoiceCategoryVolume, kAudio, kStub,
                         kHighFrequency);

dword_result_t XAudioEnableDucker_entry(dword_t unk) { return X_ERROR_SUCCESS; }
DECLARE_XBOXKRNL_EXPORT1(XAudioEnableDucker, kAudio, kStub);

dword_result_t XAudioRegisterRenderDriverClient_entry(lpdword_t callback_ptr,
                                                      lpdword_t driver_ptr) {
  if (!callback_ptr) {
    return X_E_INVALIDARG;
  }

  uint32_t callback = callback_ptr[0];

  if (!callback) {
    return X_E_INVALIDARG;
  }
  uint32_t callback_arg = callback_ptr[1];

  auto audio_system = kernel_state()->emulator()->audio_system();

  size_t index;
  auto result = audio_system->RegisterClient(callback, callback_arg, &index);
  if (XFAILED(result)) {
    return result;
  }

  assert_true(!(index & ~0x0000FFFF));
  *driver_ptr = 0x41550000 | (static_cast<uint32_t>(index) & 0x0000FFFF);
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioRegisterRenderDriverClient, kAudio,
                         kImplemented);

dword_result_t XAudioUnregisterRenderDriverClient_entry(
    lpunknown_t driver_ptr) {
  assert_true((driver_ptr.guest_address() & 0xFFFF0000) == 0x41550000);

  auto audio_system = kernel_state()->emulator()->audio_system();
  audio_system->UnregisterClient(driver_ptr.guest_address() & 0x0000FFFF);
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioUnregisterRenderDriverClient, kAudio,
                         kImplemented);

dword_result_t XAudioSubmitRenderDriverFrame_entry(lpunknown_t driver_ptr,
                                                   lpunknown_t samples_ptr) {
  assert_true((driver_ptr.guest_address() & 0xFFFF0000) == 0x41550000);

  auto audio_system = kernel_state()->emulator()->audio_system();
  auto samples =
      kernel_state()->memory()->TranslateVirtual<float*>(samples_ptr);
  audio_system->SubmitFrame(driver_ptr.guest_address() & 0x0000FFFF, samples);

  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT2(XAudioSubmitRenderDriverFrame, kAudio, kImplemented,
                         kHighFrequency);

// The echo cancellation (MEC) client of a Kinect title's microphone
// processing, {callback, argument}: called every 256-sample frame of the
// render driver (5.33 ms, a tic) by xam::KinectMicrophones.
dword_result_t XAudioRegisterRenderDriverMECClient_entry(lpdword_t client,
                                                         lpdword_t handle) {
  if (!client || !client[0] || !handle) {
    return X_E_INVALIDARG;
  }
  uint32_t mec_handle = 0;
  const X_STATUS result = xam::KinectMicrophones::Get().RegisterMecClient(
      client[0], client[1], &mec_handle);
  if (XFAILED(result)) {
    return result;
  }
  *handle = mec_handle;
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioRegisterRenderDriverMECClient, kAudio,
                         kImplemented);

dword_result_t XAudioUnregisterRenderDriverMECClient_entry(dword_t handle) {
  xam::KinectMicrophones::Get().UnregisterMecClient(handle);
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioUnregisterRenderDriverMECClient, kAudio,
                         kImplemented);

dword_result_t XAudioGetRenderDriverTic_entry() {
  return xam::KinectMicrophones::Get().render_tic();
}
DECLARE_XBOXKRNL_EXPORT2(XAudioGetRenderDriverTic, kAudio, kImplemented,
                         kHighFrequency);

// The frame last played, which the title cancels from the microphones: 6
// channels of 256 big-endian floats (as in Kixenia, by Davi Carlos).
dword_result_t XAudioCaptureRenderDriverFrame_entry(dword_t handle,
                                                    lpvoid_t buffer) {
  if (!buffer) {
    return X_ERROR_SUCCESS;
  }
  if (cvars::kinect_echo_reference) {
    kernel_state()->emulator()->audio_system()->CaptureLastFrame(
        buffer.as<uint8_t*>());
  } else {
    std::memset(buffer.as<uint8_t*>(), 0, 6 * 256 * sizeof(float));
  }
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT2(XAudioCaptureRenderDriverFrame, kAudio, kImplemented,
                         kHighFrequency);

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

DECLARE_XBOXKRNL_EMPTY_REGISTER_EXPORTS(Audio);
