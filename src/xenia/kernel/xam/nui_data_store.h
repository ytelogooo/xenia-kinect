/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XAM_NUI_DATA_STORE_H_
#define XENIA_KERNEL_XAM_NUI_DATA_STORE_H_

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace xe {
namespace kernel {
namespace xam {

// What the console keeps for Kinect titles' identity (Kinect ID) runtime,
// each blob with a version the title gives it:
// - named data, the identity database ("NuiIdentity"),
// - each profile's identity data the console caches,
// - each profile's biometric data, which on the console travels with it.
// The console encrypts these, and titles have XAM decrypt what they read
// back; here they're kept as written, so decrypting is copying. Files go to
// <content folder>/nui. Based on the xenia-edge fork's NuiDataStore.
class NuiDataStore {
 public:
  enum class Kind {
    kNamed,
    kProfileCache,
    kProfileBiometric,
  };

  // The store under the emulator's content folder.
  static NuiDataStore* Get();

  explicit NuiDataStore(std::filesystem::path root) : root_(std::move(root)) {}

  // False if there's nothing stored.
  bool Read(Kind kind, const std::string& key, uint64_t& version_out,
            std::vector<uint8_t>& data_out);
  bool Write(Kind kind, const std::string& key, uint64_t version,
             const uint8_t* data, size_t size);
  void Delete(Kind kind, const std::string& key);

  static std::string ProfileKey(uint64_t xuid);

 private:
  std::filesystem::path PathFor(Kind kind, const std::string& key) const;

  std::filesystem::path root_;
  std::mutex mutex_;
};

}  // namespace xam
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XAM_NUI_DATA_STORE_H_
