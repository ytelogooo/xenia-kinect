/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/xam/apps/xam_app.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/nui.h"
#include "xenia/kernel/xam/nui_data_store.h"
#include "xenia/kernel/xam/xam_content_device.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_usb_kinect.h"
#include "xenia/kernel/xenumerator.h"

/* Notes:
   - Messages ids that start with 0x00021xxx are UI calls
   - Messages ids that start with 0x00023xxx are used for the user profile
   - Messages ids that start with 0x0002Bxxx are used by the Kinect device
   usually for camera related functions
   - Messages ids that start with 0x0002Cxxx are used by the XamNuiIdentity
   functions
*/

DEFINE_bool(kinect_id, true,
            "Kinect titles' identity (Kinect ID) system services: the data "
            "their identity runtime stores (in <content folder>/nui), its "
            "player matching, and the skeleton frames they hand the system "
            "each frame. Off, these fail as before.",
            "HID");

DEFINE_bool(kinect_system_tracking, true,
            "With a real Kinect, answer Kinect titles' per-frame skeleton "
            "message as the console's system does: it chooses which two "
            "players they track. Kinect Adventures needs it for its title "
            "screen cursor.",
            "HID");

namespace xe {
namespace kernel {
namespace xam {
namespace apps {

// BEGIN_HEXDUMP_HELPER
// Formats up to |length| bytes as 16-byte hex rows. Used to capture the raw
// payloads of Kinect/NUI XAM messages whose layouts are not yet known.
static std::string FormatHexDump(const uint8_t* data, size_t length) {
  static const char kHex[] = "0123456789ABCDEF";
  std::string out;
  for (size_t i = 0; i < length; ++i) {
    if (i % 16 == 0) {
      out += '\n';
      out += kHex[(i >> 8) & 0xF];
      out += kHex[(i >> 4) & 0xF];
      out += kHex[i & 0xF];
      out += ": ";
    }
    out += kHex[data[i] >> 4];
    out += kHex[data[i] & 0xF];
    out += ' ';
  }
  return out;
}
// END_HEXDUMP_HELPER

static constexpr uint32_t kMaxNuiDumpBytes = 0x100;

XamApp::XamApp(KernelState* kernel_state) : App(kernel_state, 0xFE) {}

X_HRESULT XamApp::DispatchMessageSync(uint32_t message, uint32_t buffer_ptr,
                                      uint32_t buffer_length) {
  // NOTE: buffer_length may be zero or valid.
  auto buffer = memory_->TranslateVirtual(buffer_ptr);
  switch (message) {
    case 0x0002000E: {
      X_ENUMERATE_PARAM* data_ptr =
          reinterpret_cast<X_ENUMERATE_PARAM*>(buffer);

      XELOGD(
          "XEnumerateCrossTitle({:04X}, {:04X}, {:04X}, {:04X}, {}, {}, "
          "{:04X})",
          data_ptr->user_index.get(), data_ptr->flags.get(),
          data_ptr->private_enum_structure_ptr.get(),
          data_ptr->buffer_ptr.get(), data_ptr->buffer_size.get(),
          data_ptr->items_requested.get(), data_ptr->items_returned_ptr.get());

      if (!data_ptr->buffer_ptr || !data_ptr->private_enum_structure_ptr) {
        return X_E_INVALIDARG;
      }

      auto enum_struct =
          memory_->TranslateVirtual<X_KENUMERATOR_CONTENT_AGGREGATE*>(
              data_ptr->private_enum_structure_ptr);

      auto e = kernel_state_->object_table()->LookupObject<XEnumerator>(
          enum_struct->handle);

      if (!e) {
        return X_E_INVALIDARG;
      }

      assert_true(enum_struct->magic == kXObjSignature);
      assert_true(data_ptr->buffer_size == sizeof(XCONTENT_DATA_INTERNAL));

      XCONTENT_DATA_INTERNAL* content_data_ptr =
          memory_->TranslateVirtual<XCONTENT_DATA_INTERNAL*>(
              data_ptr->buffer_ptr);

      uint32_t item_count = 0;
      std::memset(content_data_ptr, 0, data_ptr->buffer_size);

      X_RESULT result =
          e->WriteItems(reinterpret_cast<uint8_t*>(content_data_ptr),
                        data_ptr->buffer_size, &item_count);

      result = X_HRESULT_FROM_WIN32(result);

      xe::be<uint32_t>* items_returned_ptr =
          memory_->TranslateVirtual<xe::be<uint32_t>*>(
              data_ptr->items_returned_ptr);

      *items_returned_ptr = item_count;

      return result;
    }
    case 0x00020021: {
      struct XContentQueryVolumeDeviceType {
        char root_name[64];
        xe::be<uint32_t> is_title_process;
        xe::be<DeviceType> device_type_ptr;
        xe::be<uint32_t> overlapped_ptr;
      }* data = reinterpret_cast<XContentQueryVolumeDeviceType*>(buffer);
      assert_true(buffer_length == sizeof(XContentQueryVolumeDeviceType));

      std::string target;
      if (!kernel_state_->file_system()->FindSymbolicLink(
              std::string(data->root_name) + ':', target)) {
        return X_E_INVALIDARG;
      }

      // Only apply this check to XContent packages
      if (!target.starts_with("\\Device\\Package_")) {
        return X_E_INVALIDARG;
      }

      xe::be<DeviceType>* device_type_ptr =
          memory_->TranslateVirtual<xe::be<DeviceType>*>(
              static_cast<uint32_t>(data->device_type_ptr.get()));

      switch (kernel_state_->deployment_type_) {
        case XDeploymentType::kDownload:
        case XDeploymentType::kInstalledToHDD: {
          *device_type_ptr = DeviceType::HDD;
        } break;
        case XDeploymentType::kOpticalDisc: {
          *device_type_ptr = DeviceType::ODD;
        } break;
        default: {
          *device_type_ptr = DeviceType::Invalid;
        } break;
      }

      XELOGD("XContentQueryVolumeDeviceType('{}', {:08X}, {:08X}, {:08X})",
             data->root_name,
             static_cast<uint32_t>(data->is_title_process.get()),
             static_cast<uint32_t>(data->device_type_ptr.get()),
             static_cast<uint32_t>(data->overlapped_ptr.get()));

      return X_E_SUCCESS;
    }
    case 0x00021012: {
      uint32_t enabled = xe::load_and_swap<uint32_t>(buffer);
      XELOGD("XEnableGuestSignin: {}", enabled ? "true" : "false");
      return X_E_SUCCESS;
    }
    case 0x00022005: {
      struct XTITLE_GET_DEPLOYMENT_TYPE {
        xe::be<uint32_t> deployment_type_ptr;
        xe::be<uint32_t> overlapped_ptr;
      }* data = reinterpret_cast<XTITLE_GET_DEPLOYMENT_TYPE*>(buffer);
      assert_true(!buffer_length ||
                  buffer_length == sizeof(XTITLE_GET_DEPLOYMENT_TYPE));
      auto deployment_type =
          memory_->TranslateVirtual<uint32_t*>(data->deployment_type_ptr);
      *deployment_type = static_cast<uint32_t>(kernel_state_->deployment_type_);
      XELOGD("XTitleGetDeploymentType({:08X}, {:08X}",
             data->deployment_type_ptr.get(), data->overlapped_ptr.get());
      return X_E_SUCCESS;
    }
    case 0x0002B001: {
      // Games used in:
      // 42560819, 373307D9
      uint32_t ms_delay = xe::load_and_swap<uint32_t>(buffer);

      XELOGD("XNuiDelayUI unimplemented({:08X})", ms_delay);
      return X_E_SUCCESS;
    }
    case 0x0002B003: {
      // Games used in:
      // 4D5309C9
      // It only receives buffer
      X_NUI_TITLE_INITIALIZE_DATA* args =
          reinterpret_cast<X_NUI_TITLE_INITIALIZE_DATA*>(buffer);
      XELOGD("XNuiTitleInitialize({:016X}, {:016X}, flags: 0x{:08X}), Stubbed",
             args->xam_version.get(), args->xtl_version.get(),
             args->flags.get());
      if (buffer_ptr && buffer_length) {
        XELOGD("XNuiTitleInitialize payload ({} bytes):{}", buffer_length,
               FormatHexDump(buffer,
                             std::min<uint32_t>(buffer_length,
                                                kMaxNuiDumpBytes)));
      }
      X_HRESULT result = X_E_FAIL;

      if (args->xam_version == 5) {
        kernel_state_->nui()->SetNUIVersion(uint64_t(args->xam_version), 0);
        kernel_state_->nui()->SetNUIVersion(uint64_t(args->xtl_version), 1);
        kernel_state_->nui()->SetInitFlags(uint32_t(args->flags));
        if ((args->flags & 0x80000000) == 0) {
          XELOGD("Todo: write data to two pointers");
          result = X_E_SUCCESS;
        }
      }
      if ((args->flags & 0x40000000) == 0) {
        XELOGD(
            "Todo: call XamNuiGetDeviceStatus and if status does not have flag "
            "1 set but has flags 4 and 0x40 call XamShowNuiTroubleshooterUI");
      }

      return result;
    }
    case 0x0002B004: {
      // Games used in:
      // Emerald Dashboard
      // Called with no buffer

      kernel_state_->nui()->SetNUIVersion(0, 0);
      kernel_state_->nui()->SetNUIVersion(0, 1);
      kernel_state_->nui()->SetInitFlags(0);
      kernel_state_->nui()->SetTiltCallback(0);

      XELOGD("XNuiTitleShutdown stubbed");
      return X_E_SUCCESS;
    }
    case 0x0002C009: {
      // Registers (non-zero) or removes (zero) a Kinect identity callback.
      // TODO(knuckleslee): Invoke the callback for identity notifications.
      if (!buffer_ptr) {
        return X_E_INVALIDARG;
      }
      XELOGD("XamApp: Kinect identity callback {:08X}",
             xe::load_and_swap<uint32_t>(buffer));
      return X_E_SUCCESS;
    }
    case 0x0002A004:
    case 0x0002A005:
    case 0x0002A006:
    case 0x0002A007:
    case 0x0002A008:
    case 0x0002A009: {
      // The console's store for Kinect titles' identity (Kinect ID) runtime:
      // a blob and its version by name (0x2A004 writes, 0x2A005 reads,
      // 0x2A006 deletes) or by profile (0x2A007-0x2A009).
      if (!cvars::kinect_id || !buffer_ptr) {
        break;
      }
      struct X_NUI_DATA_REQUEST {
        // A profile's XUID, or a name's address in the upper half.
        xe::be<uint64_t> key;
        // To read, the address the version goes to.
        xe::be<uint64_t> version;
        xe::be<uint32_t> buffer_ptr;
        // To read, the address of the buffer's size, replaced with the data's.
        xe::be<uint32_t> size;
      };
      const auto* request =
          memory_->TranslateVirtual<const X_NUI_DATA_REQUEST*>(buffer_ptr);
      const bool named = message <= 0x0002A006;
      const NuiDataStore::Kind kind = named ? NuiDataStore::Kind::kNamed
                                            : NuiDataStore::Kind::kProfileCache;
      std::string key;
      if (named) {
        const uint32_t name_ptr = uint32_t(request->key >> 32);
        if (!name_ptr) {
          return X_E_INVALIDARG;
        }
        key = memory_->TranslateVirtual<const char*>(name_ptr);
      } else {
        key = NuiDataStore::ProfileKey(request->key);
      }
      NuiDataStore* store = NuiDataStore::Get();
      switch ((message - 0x0002A004) % 3) {
        case 0: {
          const uint32_t size = request->size;
          const uint8_t* data = request->buffer_ptr
                                    ? memory_->TranslateVirtual<const uint8_t*>(
                                          request->buffer_ptr)
                                    : nullptr;
          XELOGI("Kinect ID: data {} written, version {:X}, {} bytes", key,
                 request->version.get(), size);
          return store->Write(kind, key, request->version, data,
                              data ? size : 0)
                     ? X_E_SUCCESS
                     : X_E_FAIL;
        }
        case 1: {
          uint64_t version = 0;
          std::vector<uint8_t> data;
          const bool found = store->Read(kind, key, version, data);
          static uint32_t logged = 0;
          if (logged < 16) {
            ++logged;
            XELOGI("Kinect ID: data {} read: {}", key,
                   found ? fmt::format("version {:X}, {} bytes", version,
                                       data.size())
                         : std::string("none yet"));
          }
          if (const uint32_t version_ptr = uint32_t(request->version)) {
            xe::store_and_swap<uint64_t>(memory_->TranslateVirtual(version_ptr),
                                         version);
          }
          if (!request->size) {
            return X_E_INVALIDARG;
          }
          auto* size_ptr =
              memory_->TranslateVirtual<xe::be<uint32_t>*>(request->size);
          const uint32_t capacity = *size_ptr;
          *size_ptr = uint32_t(data.size());
          if (request->buffer_ptr) {
            if (capacity < data.size()) {
              return X_E_INSUFFICIENT_BUFFER;
            }
            std::memcpy(memory_->TranslateVirtual(request->buffer_ptr),
                        data.data(), data.size());
          }
          return X_E_SUCCESS;
        }
        default:
          XELOGI("Kinect ID: data {} deleted", key);
          store->Delete(kind, key);
          return X_E_SUCCESS;
      }
    }
    case 0x0002C00C: {
      // A service of Kinect titles' identity runtime: an operation on a
      // buffer, 0 and 1 decrypting what the data store gave back, which is
      // kept unencrypted here, and 2 looking up a tuning setting by name.
      if (!cvars::kinect_id || !buffer_ptr) {
        break;
      }
      struct X_NUI_IDENTITY_SERVICE_REQUEST {
        xe::be<uint32_t> operation;
        xe::be<uint32_t> unknown_04;
        xe::be<uint64_t> parameter;
        xe::be<uint32_t> in_ptr;
        xe::be<uint32_t> in_size;
        xe::be<uint32_t> out_ptr;
        xe::be<uint32_t> out_size_ptr;
      };
      const auto* request =
          memory_->TranslateVirtual<const X_NUI_IDENTITY_SERVICE_REQUEST*>(
              buffer_ptr);
      switch (request->operation) {
        case 0:
        case 1: {
          if (!request->out_size_ptr ||
              (request->in_size && !request->in_ptr)) {
            return X_E_INVALIDARG;
          }
          auto* out_size = memory_->TranslateVirtual<xe::be<uint32_t>*>(
              request->out_size_ptr);
          const uint32_t capacity = *out_size;
          *out_size = request->in_size;
          if (!request->out_ptr) {
            return X_E_SUCCESS;
          }
          if (capacity < request->in_size) {
            return X_E_INSUFFICIENT_BUFFER;
          }
          std::memmove(memory_->TranslateVirtual(request->out_ptr),
                       memory_->TranslateVirtual(request->in_ptr),
                       request->in_size);
          return X_E_SUCCESS;
        }
        case 2:
          // Titles fall back to their own defaults.
          return X_E_NOTFOUND;
        default:
          XELOGW("Kinect ID: identity service operation {} ({:X})",
                 request->operation.get(), request->parameter.get());
          return X_E_FAIL;
      }
    }
    case 0x0002C00D: {
      // A Kinect title's identity runtime asking which player it sees: for
      // each of its eight identity slots it sends four (status, score)
      // pairs it measured against the slot's biometric data - status 4 and
      // score -1 where it has none - and xam answers with a slot. One the
      // title has a player in means it's them, an empty one means someone
      // new, to enroll there, and 8 or more means not decided yet.
      if (!cvars::kinect_id || !buffer_ptr) {
        break;
      }
      struct X_NUI_IDENTITY_MATCH_REQUEST {
        xe::be<uint32_t> buffer_ptr;
        xe::be<uint32_t> unknown_04;
        xe::be<uint64_t> parameter;
        xe::be<uint32_t> result_ptr;
        xe::be<uint32_t> buffer_size;
      };
      struct X_NUI_IDENTITY_MATCH_BUFFER {
        xe::be<uint32_t> status[8][4];
        xe::be<float> score[8][4];
        xe::be<uint32_t> tracking_id;
        xe::be<uint32_t> flags;
        xe::be<uint32_t> slot_flags[8];
      };
      static_assert_size(X_NUI_IDENTITY_MATCH_BUFFER, 296);
      const auto* request =
          memory_->TranslateVirtual<const X_NUI_IDENTITY_MATCH_REQUEST*>(
              buffer_ptr);
      if (!request->buffer_ptr || !request->result_ptr ||
          request->buffer_size < sizeof(X_NUI_IDENTITY_MATCH_BUFFER)) {
        return X_E_INVALIDARG;
      }
      const auto* match =
          memory_->TranslateVirtual<const X_NUI_IDENTITY_MATCH_BUFFER*>(
              request->buffer_ptr);
      // The slot whose measurements agree best, or the first slot with none
      // for someone new.
      int32_t best_slot = -1;
      float best_score = -1.0f;
      int32_t free_slot = -1;
      for (int32_t slot = 0; slot < 8; ++slot) {
        float total = 0.0f;
        uint32_t measured = 0;
        for (int i = 0; i < 4; ++i) {
          if (match->status[slot][i] != 4 && match->score[slot][i] >= 0.0f) {
            total += match->score[slot][i];
            ++measured;
          }
        }
        if (!measured) {
          if (free_slot < 0 && !(match->slot_flags[slot] & 1)) {
            free_slot = slot;
          }
          continue;
        }
        const float score = total / float(measured);
        if (score > best_score) {
          best_score = score;
          best_slot = slot;
        }
      }
      const uint32_t answer = best_slot >= 0   ? uint32_t(best_slot)
                              : free_slot >= 0 ? uint32_t(free_slot)
                                               : 0x80000000u;
      static uint32_t logged = 0;
      if (logged < 32) {
        ++logged;
        XELOGI(
            "Kinect ID: who is tracking ID {:X} (flags {:X})? answered {:X} "
            "(best slot {}, score {:.3f}, free slot {})",
            match->tracking_id.get(), match->flags.get(), answer, best_slot,
            best_score, free_slot);
      }
      xe::store_and_swap<uint32_t>(
          memory_->TranslateVirtual(request->result_ptr), answer);
      return X_E_SUCCESS;
    }
    case 0x0002C003:
    case 0x0002C004: {
      // A Kinect title's identity runtime answering what the system asked of
      // it: 0x2C003 an enrollment, 0x2C004 an identification. {result,
      // tracking ID, enrollment slot, ...}.
      if (!cvars::kinect_id || !buffer_ptr || buffer_length < 12) {
        break;
      }
      XELOGI("Kinect ID: {} result {:08X}, tracking ID {:X}, slot {}",
             message == 0x0002C003 ? "enrollment" : "identification",
             xe::load_and_swap<uint32_t>(buffer),
             xe::load_and_swap<uint32_t>(buffer + 4),
             xe::load_and_swap<uint32_t>(buffer + 8));
      return X_E_SUCCESS;
    }
    case 0x0002B002: {
      // A Kinect title's NUI runtime hands the system its skeleton frame each
      // frame: a 48-byte header, NUI_SKELETON_DATA[6] (0x1C0 each), then 0x30
      // bytes for the answer. Answering 0x8301000B with two tracking IDs at
      // +0xACC/+0xAD0 tells the runtime the system manages its players and
      // which two skeletons to track fully (Kinect Adventures 82D34A70: it
      // then tracks those, +0x60 set, instead of choosing itself). Kinect
      // Adventures, a launch title, expects the system to do this: without
      // it, it skips its Kinect Guide lesson and its title screen cursor
      // never follows a hand. Done with a real Kinect (kinect_system_tracking)
      // - with the SDK bridge, the bridge and the title choose.
      if (!cvars::kinect_id) {
        break;
      }
      if (!cvars::kinect_system_tracking || !buffer_ptr ||
          !xboxkrnl::usb_kinect::IsActive()) {
        return X_E_SUCCESS;
      }
      constexpr uint32_t kSkeletons = 48;
      constexpr uint32_t kSkeletonSize = 0x1C0;
      constexpr uint32_t kAnswer = 0xACC;
      static uint32_t chosen[2] = {};
      struct Seen {
        uint32_t id;
        uint32_t state;
        float z;
      };
      std::vector<Seen> seen;
      for (uint32_t i = 0; i < 6; ++i) {
        const uint8_t* skeleton = buffer + kSkeletons + i * kSkeletonSize;
        const uint32_t state = xe::load_and_swap<uint32_t>(skeleton);
        const uint32_t id = xe::load_and_swap<uint32_t>(skeleton + 4);
        if ((state == 1 || state == 2) && id) {
          seen.push_back({id, state, xe::load_and_swap<float>(skeleton + 0x18)});
        }
      }
      // Players already chosen stay; free places go to the tracked ones,
      // then the nearest.
      std::sort(seen.begin(), seen.end(), [](const Seen& a, const Seen& b) {
        return a.state != b.state ? a.state > b.state : a.z < b.z;
      });
      uint32_t next[2] = {};
      for (int k = 0; k < 2; ++k) {
        for (const Seen& player : seen) {
          if (player.id == chosen[k]) {
            next[k] = chosen[k];
          }
        }
      }
      for (const Seen& player : seen) {
        if (player.id == next[0] || player.id == next[1]) {
          continue;
        }
        if (!next[0]) {
          next[0] = player.id;
        } else if (!next[1]) {
          next[1] = player.id;
        }
      }
      static uint32_t logged = 0;
      if ((next[0] != chosen[0] || next[1] != chosen[1]) && logged < 40) {
        ++logged;
        XELOGI("Kinect: the system tracks players {} and {} ({} seen)",
               next[0], next[1], seen.size());
      }
      chosen[0] = next[0];
      chosen[1] = next[1];
      xe::store_and_swap<uint32_t>(buffer + kAnswer, next[0]);
      xe::store_and_swap<uint32_t>(buffer + kAnswer + 4, next[1]);
      return X_HRESULT(0x8301000B);
    }
    // Causes dashboard to correctly process language/region change. It does not
    // contain any buffer.
    case 0x8000000D: {
      const bool is_pc_enabled =
          (kernel_state_->xconfig()->ReadSetting<uint8_t>(
               XCONFIG_USER_CATEGORY, XCONFIG_USER_PC_FLAGS) &
           X_PC_FLAGS::PCEnabled) != 0;

      return is_pc_enabled ? X_E_ACCESS_DENIED : X_E_SUCCESS;
    }
  }
  XELOGE(
      "Unimplemented XAM message app={:08X}, msg={:08X}, arg1={:08X}, "
      "arg2={:08X}",
      app_id(), message, buffer_ptr, buffer_length);
  const uint32_t message_range = message & 0xFFFFF000;
  if ((message_range == 0x0002B000 || message_range == 0x0002C000) &&
      buffer_ptr && buffer_length) {
    XELOGE("Unimplemented NUI message {:08X} payload ({} bytes):{}", message,
           buffer_length,
           FormatHexDump(buffer, std::min<uint32_t>(buffer_length,
                                                    kMaxNuiDumpBytes)));
  }
  return X_E_FAIL;
}

}  // namespace apps
}  // namespace xam
}  // namespace kernel
}  // namespace xe
