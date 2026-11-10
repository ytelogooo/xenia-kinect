/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_PPC_PPC_FRONTEND_H_
#define XENIA_CPU_PPC_PPC_FRONTEND_H_

#include <memory>
#include <unordered_map>

#include "xenia/base/type_pool.h"
#include "xenia/cpu/function.h"
#include "xenia/memory.h"

namespace xe {
namespace cpu {
class Processor;
}  // namespace cpu
}  // namespace xe

namespace xe {
namespace cpu {
namespace ppc {

class PPCTranslator;
struct PPCContext_s;

// Host callbacks that run just before a guest instruction executes. Emulated
// peripherals use them to supply data at a known point in title code (for
// example the emulated Kinect filling in skeleton frames). Register before
// the containing guest function is first translated, e.g. while kernel
// modules are being created.
using GuestHookHandler = void (*)(PPCContext_s* context, uint32_t address);
void RegisterGuestHook(uint32_t address, GuestHookHandler handler);

struct PPCBuiltins {
  int32_t global_lock_count;
  Function* check_global_lock;
  Function* enter_global_lock;
  Function* leave_global_lock;
  Function* syscall_handler;
  // Diagnostic probes from --probe_guest_addresses, keyed by guest address.
  std::unordered_map<uint32_t, Function*> probes;
};

class PPCFrontend {
 public:
  explicit PPCFrontend(Processor* processor);
  ~PPCFrontend();

  bool Initialize();

  Processor* processor() const { return processor_; }
  Memory* memory() const;
  PPCBuiltins* builtins() { return &builtins_; }

  // Builtin calling the hook registered for address, or nullptr.
  Function* GetGuestHookFunction(uint32_t address);

  bool DeclareFunction(GuestFunction* function);
  bool DefineFunction(GuestFunction* function, uint32_t debug_info_flags);

 private:
  Processor* processor_;
  PPCBuiltins builtins_ = {0};
  TypePool<PPCTranslator, PPCFrontend*> translator_pool_;
};
// Checks the state of the global lock and sets scratch to the current MSR
// value.
void CheckGlobalLock(PPCContext* ppc_context, void* arg0, void* arg1);
}  // namespace ppc
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_PPC_PPC_FRONTEND_H_
