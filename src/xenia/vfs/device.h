/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_VFS_DEVICE_H_
#define XENIA_VFS_DEVICE_H_

#include <memory>
#include <string>

#include "xenia/base/mutex.h"
#include "xenia/base/string_buffer.h"
#include "xenia/vfs/entry.h"

namespace xe {
namespace vfs {

class Device {
 public:
  explicit Device(const std::string_view mount_path);
  virtual ~Device();

  virtual bool Initialize() = 0;

  const std::string_view mount_path() const { return mount_path_; }
  virtual bool is_read_only() const { return true; }
  virtual bool is_io_queued() const { return false; }

  virtual void Dump(StringBuffer* string_buffer) = 0;
  virtual Entry* ResolvePath(const std::string_view path) = 0;

  virtual const std::string& name() const = 0;
  virtual uint32_t attributes() const = 0;
  virtual uint32_t component_name_max_length() const = 0;

  virtual uint32_t total_allocation_units() const = 0;
  virtual uint32_t available_allocation_units() const = 0;
  virtual uint32_t sectors_per_allocation_unit() const = 0;
  virtual uint32_t bytes_per_sector() const = 0;

  // The volume size for a file on this device; a device holding several
  // volumes (NullDevice's cache partitions) may answer per volume.
  virtual uint64_t TotalAllocationUnitsFor(const Entry* entry) const {
    return total_allocation_units();
  }
  virtual uint64_t AvailableAllocationUnitsFor(const Entry* entry) const {
    return available_allocation_units();
  }
  virtual uint32_t SectorsPerAllocationUnitFor(const Entry* entry) const {
    return sectors_per_allocation_unit();
  }

 protected:
  xe::global_critical_region global_critical_region_;
  std::string mount_path_;
};

}  // namespace vfs
}  // namespace xe

#endif  // XENIA_VFS_DEVICE_H_
