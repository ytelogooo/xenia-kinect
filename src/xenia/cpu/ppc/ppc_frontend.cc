/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/ppc/ppc_frontend.h"

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/atomic.h"
#include "xenia/base/logging.h"
#include "xenia/base/mutex.h"
#include "xenia/base/utf8.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <cstring>
#include <unordered_map>
#include <vector>
#include "xenia/cpu/cpu_flags.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/ppc/ppc_emit.h"
#include "xenia/cpu/ppc/ppc_opcode_info.h"
#include "xenia/cpu/ppc/ppc_translator.h"
#include "xenia/cpu/processor.h"

namespace xe {
namespace cpu {
namespace ppc {

void InitializeIfNeeded();
void CleanupOnShutdown();

void InitializeIfNeeded() {
  static bool has_initialized = false;
  if (has_initialized) {
    return;
  }
  has_initialized = true;

  RegisterEmitCategoryAltivec();
  RegisterEmitCategoryALU();
  RegisterEmitCategoryControl();
  RegisterEmitCategoryFPU();
  RegisterEmitCategoryMemory();

  atexit(CleanupOnShutdown);
}

void CleanupOnShutdown() {}

PPCFrontend::PPCFrontend(Processor* processor) : processor_(processor) {
  InitializeIfNeeded();
}

PPCFrontend::~PPCFrontend() {
  // Force cleanup now before we deinit.
  translator_pool_.Reset();
}

Memory* PPCFrontend::memory() const { return processor_->memory(); }

// Checks the state of the global lock and sets scratch to the current MSR
// value.
void CheckGlobalLock(PPCContext* ppc_context, void* arg0, void* arg1) {
  auto global_mutex = reinterpret_cast<global_mutex_type*>(arg0);
  auto global_lock_count = reinterpret_cast<int32_t*>(arg1);
  std::lock_guard<global_mutex_type> lock(*global_mutex);
  ppc_context->scratch = *global_lock_count ? 0 : 0x8000;
}

// Enters the global lock. Safe to recursion.
void EnterGlobalLock(PPCContext* ppc_context, void* arg0, void* arg1) {
  auto global_mutex = reinterpret_cast<global_mutex_type*>(arg0);
  auto global_lock_count = reinterpret_cast<int32_t*>(arg1);
  global_mutex->lock();
  xe::atomic_inc(global_lock_count);
}

// Leaves the global lock. Safe to recursion.
void LeaveGlobalLock(PPCContext* ppc_context, void* arg0, void* arg1) {
  auto global_mutex = reinterpret_cast<global_mutex_type*>(arg0);
  auto global_lock_count = reinterpret_cast<int32_t*>(arg1);
  auto new_lock_count = xe::atomic_dec(global_lock_count);
  assert_true(new_lock_count >= 0);
  global_mutex->unlock();
}

void SyscallHandler(PPCContext* ppc_context, void* arg0, void* arg1) {
  uint64_t syscall_number = ppc_context->r[0];
  switch (syscall_number) {
    default:
      assert_unhandled_case(syscall_number);
      XELOGE("Unhandled syscall {}!", syscall_number);
      break;
#pragma warning(suppress : 4065)
  }
}

// Logs executions of a --probe_guest_addresses entry. arg0 is the address.
void ProbeHandler(PPCContext* ppc_context, void* arg0, void* arg1) {
  static std::mutex probe_mutex;
  static std::unordered_map<uint32_t, uint64_t> probe_counts;
  const uint32_t address = uint32_t(reinterpret_cast<uintptr_t>(arg0));
  uint64_t count;
  {
    std::lock_guard<std::mutex> lock(probe_mutex);
    count = ++probe_counts[address];
  }
  static const std::vector<uint32_t> threads = [] {
    std::vector<uint32_t> ids;
    for (auto part : xe::utf8::split(cvars::probe_thread_ids, ",; ")) {
      ids.push_back(
          uint32_t(std::strtoul(std::string(part).c_str(), nullptr, 10)));
    }
    return ids;
  }();
  if (!threads.empty() &&
      std::find(threads.begin(), threads.end(), ppc_context->thread_id) ==
          threads.end()) {
    return;
  }
  static const std::pair<uint32_t, uint32_t> dump = [] {
    const auto parts = xe::utf8::split(cvars::probe_dump, ":");
    if (parts.size() != 2) {
      return std::pair<uint32_t, uint32_t>(0, 0);
    }
    const uint32_t addr =
        uint32_t(std::strtoul(std::string(parts[0]).c_str(), nullptr, 16));
    const uint32_t len =
        uint32_t(std::strtoul(std::string(parts[1]).c_str(), nullptr, 16));
    return std::pair<uint32_t, uint32_t>(addr, std::min(len, 0x1000u));
  }();
  if (cvars::probe_log_limit) {
    // With a dump, a hit is logged only when the dumped memory changed since
    // that probe's last logged hit, and only the 32-byte rows that changed.
    std::string changed_rows;
    if (dump.first && dump.second) {
      static std::mutex dump_mutex;
      static std::unordered_map<uint32_t, std::vector<uint8_t>> last_dumps;
      const uint8_t* bytes = ppc_context->TranslateVirtual(dump.first);
      std::lock_guard<std::mutex> lock(dump_mutex);
      std::vector<uint8_t>& last = last_dumps[address];
      const bool first = last.empty();
      if (!first && std::memcmp(last.data(), bytes, dump.second) == 0) {
        return;
      }
      for (uint32_t row = 0; row < dump.second; row += 32) {
        const uint32_t row_length = std::min(32u, dump.second - row);
        if (!first &&
            std::memcmp(last.data() + row, bytes + row, row_length) == 0) {
          continue;
        }
        changed_rows += fmt::format("\n  dump {:08X}:", dump.first + row);
        for (uint32_t i = 0; i < row_length; ++i) {
          changed_rows += fmt::format(" {:02X}", bytes[row + i]);
        }
      }
      last.assign(bytes, bytes + dump.second);
    }
    static std::atomic<uint32_t> logged{0};
    if (logged.fetch_add(1) >= cvars::probe_log_limit) {
      return;
    }
    // A LARGE_INTEGER timeout or interval is usually on the stack.
    auto stack_value = [&](uint64_t pointer) -> std::string {
      const uint32_t p = uint32_t(pointer);
      if (p < 0x70000000 || p >= 0x7F000000 || (p & 7)) {
        return "-";
      }
      return fmt::format("{}", int64_t(xe::load_and_swap<uint64_t>(
                                   ppc_context->TranslateVirtual(p))));
    };
    XELOGI(
        "probe {:08X} hit #{} thread {}: r3={:08X} r4={:08X} r5={:08X} "
        "r6={:08X} r7={:08X} r8={:08X} [r5]={} [r6]={} lr={:08X}",
        address, count, ppc_context->thread_id, uint32_t(ppc_context->r[3]),
        uint32_t(ppc_context->r[4]), uint32_t(ppc_context->r[5]),
        uint32_t(ppc_context->r[6]), uint32_t(ppc_context->r[7]),
        uint32_t(ppc_context->r[8]), stack_value(ppc_context->r[5]),
        stack_value(ppc_context->r[6]), uint32_t(ppc_context->lr));
    if (!changed_rows.empty()) {
      XELOGI("probe {:08X} memory:{}", address, changed_rows);
    }
    return;
  }
  if (count <= 5 || count % 300 == 0) {
    XELOGI("probe {:08X} hit #{}: r3={:08X} r4={:08X} r5={:08X} r6={:08X} "
           "lr={:08X}",
           address, count, uint32_t(ppc_context->r[3]),
           uint32_t(ppc_context->r[4]), uint32_t(ppc_context->r[5]),
           uint32_t(ppc_context->r[6]), uint32_t(ppc_context->lr));
    if (dump.first && dump.second) {
      const uint8_t* bytes = ppc_context->TranslateVirtual(dump.first);
      std::string line;
      for (uint32_t i = 0; i < dump.second; ++i) {
        if (i % 32 == 0) {
          if (!line.empty()) {
            XELOGI("{}", line);
          }
          line = fmt::format("  dump {:08X}:", dump.first + i);
        }
        line += fmt::format(" {:02X}", bytes[i]);
      }
      if (!line.empty()) {
        XELOGI("{}", line);
      }
    }
  }
}

namespace {
std::mutex guest_hook_mutex;
std::unordered_map<uint32_t, GuestHookHandler> guest_hooks;
std::unordered_map<uint32_t, Function*> guest_hook_functions;
std::atomic<bool> any_guest_hooks{false};

void GuestHookThunk(PPCContext* ppc_context, void* arg0, void* arg1) {
  reinterpret_cast<GuestHookHandler>(arg0)(
      ppc_context, uint32_t(reinterpret_cast<uintptr_t>(arg1)));
}
}  // namespace

void RegisterGuestHook(uint32_t address, GuestHookHandler handler) {
  std::lock_guard<std::mutex> lock(guest_hook_mutex);
  guest_hooks[address] = handler;
  any_guest_hooks = true;
  XELOGI("Guest hook registered at {:08X}", address);
}

Function* PPCFrontend::GetGuestHookFunction(uint32_t address) {
  if (!any_guest_hooks.load(std::memory_order_relaxed)) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(guest_hook_mutex);
  auto hook = guest_hooks.find(address);
  if (hook == guest_hooks.end()) {
    return nullptr;
  }
  auto& function = guest_hook_functions[address];
  if (!function) {
    function = processor_->DefineBuiltin(
        fmt::format("GuestHook_{:08X}", address), GuestHookThunk,
        reinterpret_cast<void*>(hook->second),
        reinterpret_cast<void*>(uintptr_t(address)));
  }
  return function;
}

bool PPCFrontend::Initialize() {
  void* arg0 = reinterpret_cast<void*>(&xe::global_critical_region::mutex());
  void* arg1 = reinterpret_cast<void*>(&builtins_.global_lock_count);
  builtins_.check_global_lock =
      processor_->DefineBuiltin("CheckGlobalLock", CheckGlobalLock, arg0, arg1);
  builtins_.enter_global_lock =
      processor_->DefineBuiltin("EnterGlobalLock", EnterGlobalLock, arg0, arg1);
  builtins_.leave_global_lock =
      processor_->DefineBuiltin("LeaveGlobalLock", LeaveGlobalLock, arg0, arg1);
  builtins_.syscall_handler = processor_->DefineBuiltin(
      "SyscallHandler", SyscallHandler, nullptr, nullptr);
  for (const auto& token : xe::utf8::split(
           cvars::probe_guest_addresses, ",; ")) {
    const uint32_t address =
        uint32_t(std::strtoul(std::string(token).c_str(), nullptr, 16));
    if (!address) {
      continue;
    }
    builtins_.probes[address] = processor_->DefineBuiltin(
        fmt::format("Probe_{:08X}", address), ProbeHandler,
        reinterpret_cast<void*>(uintptr_t(address)), nullptr);
    XELOGI("Guest probe installed at {:08X}", address);
  }
  return true;
}

bool PPCFrontend::DeclareFunction(GuestFunction* function) {
  // chrispy: make sure we aren't declaring a function that is actually padding
  // data, this will mess up PPCScanner and is hard to debug wow, this halo
  // reach actually has branches into 0 opcodes, look into further
  // xenia_assert(*reinterpret_cast<const uint32_t*>(
  //                  this->memory()->TranslateVirtual(function->address())) !=
  //                  0);
  //  Could scan or something here.
  //  Could also check to see if it's a well-known function type and classify
  //  for later.
  //  Could also kick off a precompiler, since we know it's likely the function
  //  will be demanded soon.
  return true;
}

bool PPCFrontend::DefineFunction(GuestFunction* function,
                                 uint32_t debug_info_flags) {
  auto translator = translator_pool_.Allocate(this);
  bool result = translator->Translate(function, debug_info_flags);
  translator->Reset();
  translator_pool_.Release(translator);
  return result;
}

}  // namespace ppc
}  // namespace cpu
}  // namespace xe
