/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/vfs/devices/null_device.h"

#include <cctype>

#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/string.h"
#include "xenia/kernel/xfile.h"
#include "xenia/vfs/devices/null_entry.h"

namespace xe {
namespace vfs {

NullDevice::NullDevice(const std::string& mount_path,
                       const std::initializer_list<std::string>& null_paths)
    : Device(mount_path), name_("NullDevice"), null_paths_(null_paths) {}

NullDevice::~NullDevice() = default;

bool NullDevice::Initialize() {
  auto root_entry = new NullEntry(this, nullptr, mount_path_);
  root_entry->attributes_ = kFileAttributeDirectory;
  root_entry_ = std::unique_ptr<Entry>(root_entry);

  for (auto path : null_paths_) {
    auto child = NullEntry::Create(this, root_entry, path);
    root_entry->children_.push_back(std::unique_ptr<Entry>(child));
  }
  return true;
}

void NullDevice::Dump(StringBuffer* string_buffer) {
  auto global_lock = global_critical_region_.Acquire();
  root_entry_->Dump(string_buffer, 0);
}

Entry* NullDevice::ResolvePath(const std::string_view path) {
  XELOGFS("NullDevice::ResolvePath({})", path);

  auto root = root_entry_.get();
  if (path.empty()) {
    return root_entry_.get();
  }

  for (auto& child : root->children()) {
    if (!xe_strcasecmp(child->path().c_str(), path.data())) {
      return child.get();
    }
  }

  return nullptr;
}

namespace {
constexpr uint64_t kCachePartitionBytes = 0x80000000ull;  // 2 GiB

// "\\Cache1" from an entry path such as "\\Cache1" or "\\Cache1\\".
std::string VolumeOf(const std::string& path) {
  std::string volume = path;
  while (volume.size() > 1 && (volume.back() == '\\' || volume.back() == '/')) {
    volume.pop_back();
  }
  const size_t next = volume.find('\\', 1);
  if (next != std::string::npos) {
    volume.resize(next);
  }
  for (char& c : volume) {
    c = char(std::tolower(uint8_t(c)));
  }
  return volume;
}
}  // namespace

void NullDevice::NoteFatxFormat(const std::string& volume,
                                uint32_t sectors_per_cluster) {
  std::lock_guard<std::mutex> lock(formats_mutex_);
  auto& noted = formatted_sectors_per_cluster_[VolumeOf(volume)];
  if (noted != sectors_per_cluster) {
    XELOGI("NullDevice: {}{} formatted as FATX with {}-byte clusters",
           mount_path_, volume, sectors_per_cluster * bytes_per_sector());
  }
  noted = sectors_per_cluster;
}

uint32_t NullDevice::FormattedSectorsPerCluster(const Entry* entry) const {
  if (!entry) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(formats_mutex_);
  auto it = formatted_sectors_per_cluster_.find(VolumeOf(entry->path()));
  return it == formatted_sectors_per_cluster_.end() ? 0 : it->second;
}

uint64_t NullDevice::TotalAllocationUnitsFor(const Entry* entry) const {
  const uint32_t sectors = FormattedSectorsPerCluster(entry);
  return sectors ? kCachePartitionBytes / (uint64_t(sectors) * bytes_per_sector())
                 : total_allocation_units();
}

uint64_t NullDevice::AvailableAllocationUnitsFor(const Entry* entry) const {
  const uint32_t sectors = FormattedSectorsPerCluster(entry);
  return sectors ? kCachePartitionBytes / (uint64_t(sectors) * bytes_per_sector())
                 : available_allocation_units();
}

uint32_t NullDevice::SectorsPerAllocationUnitFor(const Entry* entry) const {
  const uint32_t sectors = FormattedSectorsPerCluster(entry);
  return sectors ? sectors : sectors_per_allocation_unit();
}

}  // namespace vfs
}  // namespace xe
