/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// LDI* - the kernel's block-based LZX decompressor. Titles reach it through
// XMemCreateDecompressionContext/XMemDecompress, and the Kinect NUI runtime
// uses it directly to unpack its skeletal tracking database
// (stexemplar:\database.gmsodf). Each LDIDecompress call hands over one
// compressed block that expands to at most 32 KB (one LZX frame); decoder
// state (window, trees, repeat offsets) carries over between calls until
// LDIResetDecompression.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>

#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_private.h"
#include "xenia/xbox.h"

#include "third_party/mspack/lzx.h"
#include "third_party/mspack/mspack.h"

namespace xe {
namespace kernel {
namespace xboxkrnl {

namespace {

// MDI/LDI result codes.
constexpr uint32_t kLdiOk = 0;
constexpr uint32_t kLdiNoMemory = 1;
constexpr uint32_t kLdiBadParameters = 2;
constexpr uint32_t kLdiFailed = 4;

constexpr uint32_t kLdiFrameSize = 0x8000;
// Largest compressed block for a 32 KB frame (frame + 6 KB worst case);
// titles check for at least this.
constexpr uint32_t kLdiSourceBufferMin = kLdiFrameSize + 0x1800;
// Zero bytes appended after each block so the bit reader can prefetch past
// the end of the block without hitting end-of-input.
constexpr size_t kInputPadding = 32;

// Doubles as the mspack_file for both the input and the output stream.
struct LdiStream {
  const uint8_t* in = nullptr;
  size_t in_size = 0;
  size_t in_pos = 0;
  uint8_t* out = nullptr;
  size_t out_capacity = 0;
  size_t out_pos = 0;
};

int LdiRead(mspack_file* file, void* buffer, int bytes) {
  auto* s = reinterpret_cast<LdiStream*>(file);
  const size_t total = s->in_size + kInputPadding;
  if (bytes <= 0 || s->in_pos >= total) {
    return 0;
  }
  const size_t n = std::min<size_t>(size_t(bytes), total - s->in_pos);
  auto* dst = static_cast<uint8_t*>(buffer);
  for (size_t i = 0; i < n; ++i) {
    const size_t p = s->in_pos + i;
    dst[i] = p < s->in_size ? s->in[p] : 0;
  }
  s->in_pos += n;
  return int(n);
}

int LdiWrite(mspack_file* file, void* buffer, int bytes) {
  auto* s = reinterpret_cast<LdiStream*>(file);
  if (bytes < 0 || s->out_pos + size_t(bytes) > s->out_capacity) {
    return -1;
  }
  std::memcpy(s->out + s->out_pos, buffer, size_t(bytes));
  s->out_pos += size_t(bytes);
  return bytes;
}

void LdiMessage(mspack_file*, const char*, ...) {}
void* LdiAlloc(mspack_system*, size_t bytes) { return std::calloc(1, bytes); }
void LdiFree(void* ptr) { std::free(ptr); }
void LdiCopy(void* src, void* dest, size_t bytes) {
  std::memmove(dest, src, bytes);
}

mspack_system ldi_system = {
    nullptr,  // open
    nullptr,  // close
    LdiRead,  LdiWrite,
    nullptr,  // seek
    nullptr,  // tell
    LdiMessage, LdiAlloc, LdiFree, LdiCopy,
    nullptr,  // null_ptr
};

struct LdiContext {
  int window_bits = 17;
  LdiStream input;
  LdiStream output;
  lzxd_stream* lzx = nullptr;

  bool Init() {
    Free();
    lzx = lzxd_init(&ldi_system, reinterpret_cast<mspack_file*>(&input),
                    reinterpret_cast<mspack_file*>(&output), window_bits, 0,
                    0x10000, 0, 0);
    return lzx != nullptr;
  }
  void Free() {
    if (lzx) {
      lzxd_free(lzx);
      lzx = nullptr;
    }
  }
  ~LdiContext() { Free(); }
};

std::mutex ldi_mutex;
std::map<uint32_t, std::unique_ptr<LdiContext>> ldi_contexts;
uint32_t ldi_next_handle = 0x4C440001;  // 'LD'..

LdiContext* LookupContext(uint32_t handle) {
  auto it = ldi_contexts.find(handle);
  return it == ldi_contexts.end() ? nullptr : it->second.get();
}

}  // namespace

// int LDICreateDecompression(UINT* data_block_max, void* configuration,
//     alloc, free, memory, UINT* src_buffer_min, LDI_HANDLE* handle)
// configuration starts with the LZX window size in bytes.
dword_result_t LDICreateDecompression_entry(
    lpdword_t data_block_max, lpdword_t configuration, dword_t alloc_fn,
    dword_t free_fn, dword_t memory, lpdword_t src_buffer_min,
    lpdword_t handle_out) {
  if (!handle_out) {
    return kLdiBadParameters;
  }
  *handle_out = 0;

  uint32_t window_size = configuration ? uint32_t(*configuration) : 0x20000;
  int window_bits = 17;
  if (window_size >= 15 && window_size <= 21) {
    window_bits = int(window_size);
  } else {
    uint32_t bit = 0;
    if (!xe::bit_scan_forward(window_size, &bit) || bit < 15 || bit > 21) {
      XELOGE("LDICreateDecompression: unsupported window size {:08X}",
             window_size);
      return kLdiBadParameters;
    }
    window_bits = int(bit);
  }

  auto context = std::make_unique<LdiContext>();
  context->window_bits = window_bits;
  if (!context->Init()) {
    return kLdiNoMemory;
  }

  if (data_block_max &&
      (*data_block_max == 0 || *data_block_max > kLdiFrameSize)) {
    *data_block_max = kLdiFrameSize;
  }
  if (src_buffer_min) {
    *src_buffer_min = kLdiSourceBufferMin;
  }

  std::lock_guard<std::mutex> lock(ldi_mutex);
  const uint32_t handle = ldi_next_handle++;
  ldi_contexts[handle] = std::move(context);
  *handle_out = handle;
  XELOGD("LDICreateDecompression: window {:08X} (bits {}) -> handle {:08X}",
         window_size, window_bits, handle);
  return kLdiOk;
}
DECLARE_XBOXKRNL_EXPORT1(LDICreateDecompression, kNone, kImplemented);

// int LDIDecompress(LDI_HANDLE handle, BYTE* src, UINT src_size, BYTE* dst,
//     UINT* dst_size)  -- *dst_size: in = expected output, out = produced.
dword_result_t LDIDecompress_entry(dword_t handle, lpvoid_t src,
                                   dword_t src_size, lpvoid_t dst,
                                   lpdword_t dst_size) {
  std::lock_guard<std::mutex> lock(ldi_mutex);
  LdiContext* context = LookupContext(handle);
  if (!context || !src || !dst || !dst_size) {
    return kLdiBadParameters;
  }
  const uint32_t wanted = *dst_size;
  if (wanted == 0 || wanted > kLdiFrameSize) {
    return kLdiBadParameters;
  }

  context->input = {src.as<const uint8_t*>(), src_size, 0};
  context->output = {nullptr, 0, 0, dst.as<uint8_t*>(), wanted, 0};

  lzxd_stream* lzx = context->lzx;
  // Every block starts a fresh bitstream: drop whatever the bit reader
  // prefetched from the previous block's padding.
  lzx->i_ptr = lzx->i_end = lzx->inbuf;
  lzx->bit_buffer = 0;
  lzx->bits_left = 0;
  lzx->input_end = 0;
  // Keep mspack's frame counter in step with the bytes produced so far, and
  // tell it exactly how much to produce so it never decodes ahead into the
  // next (not yet supplied) block.
  lzx->frame = uint32_t(lzx->offset / kLdiFrameSize);
  lzxd_set_output_length(lzx, lzx->offset + wanted);

  const int result = lzxd_decompress(lzx, wanted);
  *dst_size = uint32_t(context->output.out_pos);
  if (result != MSPACK_ERR_OK) {
    XELOGE("LDIDecompress({:08X}): lzx error {} ({} -> {} of {} bytes)",
           handle.value(), result, src_size.value(),
           context->output.out_pos, wanted);
    lzx->error = MSPACK_ERR_OK;
    return kLdiFailed;
  }
  return kLdiOk;
}
DECLARE_XBOXKRNL_EXPORT2(LDIDecompress, kNone, kImplemented, kHighFrequency);

dword_result_t LDIResetDecompression_entry(dword_t handle) {
  std::lock_guard<std::mutex> lock(ldi_mutex);
  LdiContext* context = LookupContext(handle);
  if (!context) {
    return kLdiBadParameters;
  }
  return context->Init() ? kLdiOk : kLdiNoMemory;
}
DECLARE_XBOXKRNL_EXPORT2(LDIResetDecompression, kNone, kImplemented,
                         kHighFrequency);

dword_result_t LDIDestroyDecompression_entry(dword_t handle) {
  std::lock_guard<std::mutex> lock(ldi_mutex);
  if (ldi_contexts.erase(handle) == 0) {
    return kLdiBadParameters;
  }
  return kLdiOk;
}
DECLARE_XBOXKRNL_EXPORT1(LDIDestroyDecompression, kNone, kImplemented);

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

DECLARE_XBOXKRNL_EMPTY_REGISTER_EXPORTS(Ldi);
