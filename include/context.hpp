/*
 *  Copyright 2018 Digital Media Professionals Inc.

 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at

 *      http://www.apache.org/licenses/LICENSE-2.0

 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
/// @file
/// @brief dmp_dv_context implementation.
#pragma once

#include "base.hpp"

#include <string>


#ifndef ERESTARTSYS
#define ERESTARTSYS 512
#endif


/// @brief Implementation of dmp_dv_context.
class CDMPDVContext : public CDMPDVBase {
 public:
  /// @brief Constructor.
  CDMPDVContext() : CDMPDVBase() {
    fd_ion_ = -1;
    dma_heap_id_mask_ = 0;
    ub_size_ = 0;
    max_kernel_size_ = 3;
    conv_freq_ = 0;
    fc_freq_ = 0;
    max_fc_vector_size_ = 16384;
    mac_num_ = 0;
    svn_version_ = 0;
    zia_c2_ = false;
  }

  /// @brief Called when the object is about to be destroyed.
  virtual ~CDMPDVContext() {
    Cleanup();
  }

  /// @brief Releases held resources.
  void Cleanup() {
    if (fd_ion_ != -1) {
      close(fd_ion_);
      fd_ion_ = -1;
    }
    dma_heap_id_mask_ = 0;
  }

  /// @brief Initializes the DV device.
  bool Initialize() {
    Cleanup();

    fd_ion_ = open("/dev/ion", O_RDONLY | O_CLOEXEC);  // O_CLOEXEC is suggested for security
    if (fd_ion_ == -1) {
      SET_ERR("open() failed for /dev/ion: %s", strerror(errno));
      return false;
    }

    // Get ION heap counts
    struct ion_heap_query query;
    memset(&query, 0, sizeof(query));
    int res = ioctl(fd_ion_, ION_IOC_HEAP_QUERY, &query);
    if (res < 0) {
      SET_IOCTL_ERR(res, "/dev/ion", "ION_IOC_HEAP_QUERY");
      return false;
    }
    const int n = query.cnt;
    if ((n < 1) || (n > 32)) {
      SET_ERR("Got unexpected number of ION heaps: %d", n);
      return false;
    }

    // Get ION heap informations
    dma_heap_id_mask_ = 0;
    struct ion_heap_data heaps[32];
    query.heaps = (size_t)&heaps[0];
    res = ioctl(fd_ion_, ION_IOC_HEAP_QUERY, &query);
    if (res < 0) {
      SET_IOCTL_ERR(res, "/dev/ion", "ION_IOC_HEAP_QUERY");
      return false;
    }
    for (int i = 0; i < n; ++i) {
      switch (heaps[i].type) {
        case ION_HEAP_TYPE_SYSTEM:
        case ION_HEAP_TYPE_SYSTEM_CONTIG:
        case ION_HEAP_TYPE_CARVEOUT:
        case ION_HEAP_TYPE_CHUNK:
        case ION_HEAP_TYPE_CUSTOM:
          break;
        case ION_HEAP_TYPE_DMA:
          dma_heap_id_mask_ |= (1 << heaps[i].heap_id);
          break;
        default:
          break;
      }
    }
    if (!dma_heap_id_mask_) {
      SET_ERR("ION heaps doesn\'t contain ION_HEAP_TYPE_DMA");
      return false;
    }

    ub_size_ = sysfs_read_int("conv/ub_size", 0);
    max_kernel_size_ = sysfs_read_int("conv/max_kernel_size", 3);
    conv_freq_ = sysfs_read_int("conv/conv_freq", 0);
    fc_freq_ = sysfs_read_int("fc/fc_freq", 0);
    max_fc_vector_size_ = sysfs_read_int("fc/max_fc_vector_size", 16384);
    mac_num_ = sysfs_read_int("conv/mac_num", 0);
    svn_version_ = sysfs_read_int("conv/svn_version", 0);

    zia_c2_ = (sizeof(size_t) == 4) && (ub_size_ == 524288) && (mac_num_ == 576);  // TODO: add proper detection.

    char s[256];
    snprintf(s, sizeof(s), "DMP DV: ub_size=%d max_kernel_size=%d conv_freq=%d fc_freq=%d max_fc_vector_size=%d",
             ub_size_, max_kernel_size_, conv_freq_, fc_freq_, max_fc_vector_size_);
    info_ = s;

    return true;
  }

  /// @brief Reads single int value from sysfs file.
  int sysfs_read_int(const char *key, int def) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/class/dmp_dv/dv_%s", key);
    FILE *fin = fopen(path, "r");
    if (!fin) {
      return def;
    }
    int res = def;
    if (fscanf(fin, "%d", &res) != 1) {
      res = def;
    }
    fclose(fin);
    return res;
  }

  /// @brief Returns handle to ION file descriptor.
  inline int get_fd_ion() const {
    return fd_ion_;
  }

  /// @brief Returns ION DMA heap id mask.
  inline uint32_t get_dma_heap_id_mask() const {
    return dma_heap_id_mask_;
  }

  /// @brief Returns device information string.
  inline const char *GetInfoString() const {
    return info_.c_str();
  }

  /// @brief Returns maximum supported kernel size.
  inline int get_max_kernel_size() const {
    return max_kernel_size_;
  }

  /// @brief Returns maximum supported fully connected block input size.
  inline int get_max_fc_vector_size() const {
    return max_fc_vector_size_;
  }

  /// @brief Returns unified buffer size in bytes.
  inline int get_ub_size() const {
    return ub_size_;
  }

  /// @brief Returns frequency of CONV
  inline int get_conv_freq() const {
    return conv_freq_;
  }

  /// @brief Returns frequency of FC
  inline int get_fc_freq() const {
    return fc_freq_;
  }

  /// @brief Returns hardware SVN version of CONV.
  inline int get_svn_version() const {
    return svn_version_;
  }

  /// @brief Returns true if the hardware is ZIA-C2.
  inline bool is_zia_c2() const {
    return zia_c2_;
  }

  int GetInfo(struct dmp_dv_info *p_info) {
    if (p_info->size < 8) {
      SET_ERR("Invalid argument: info->size is too small: %u", p_info->size);
      return -1;
    }
    p_info->version = 0;
    if (p_info->size >= sizeof(struct dmp_dv_info_v0)) {
      struct dmp_dv_info_v0 *info = (struct dmp_dv_info_v0*)p_info;
      info->ub_size = ub_size_;
      info->max_kernel_size = max_kernel_size_;
      info->conv_freq = conv_freq_;
      info->fc_freq = fc_freq_;
      info->max_fc_vector_size = max_fc_vector_size_;
    }
    return 0;
  }

  /// @brief If specified device exists.
  inline int DeviceExists(int dev_type_id) {
    switch (dev_type_id) {
      case DMP_DV_DEV_CONV:
        return get_conv_freq() ? 1 : 0;
      case DMP_DV_DEV_FC:
        return get_fc_freq() ? 1 : 0;
      case DMP_DV_DEV_IPU:
        {
          struct stat s;
          memset(&s, 0, sizeof(s));
          if (stat(DMP_DV_DEV_PATH_IPU, &s) != 0) {
            return 0;
          }
          return S_ISCHR(s.st_mode) ? 1 : 0;
        }
      default:
        SET_ERR("Invalid argument: unsupported device type %d", dev_type_id);
        break;
    }
    return -1;
  }

 private:
  /// @brief Size of the Unified Buffer.
  int ub_size_;

  /// @brief Size of the maximum kernel.
  int max_kernel_size_;

  /// @brief Convolutional block frequency in MHz.
  int conv_freq_;

  /// @brief Fully Connected block frequency in MHz.
  int fc_freq_;

  /// @brief Fully Connected block maximum input vector size in elements.
  int max_fc_vector_size_;

  /// @brief Number of MACs.
  int mac_num_;

  /// @brief Hardware SVN version.
  int svn_version_;

  /// @brief If the hardware is ZIA-C2.
  bool zia_c2_;

  /// @brief Device information.
  std::string info_;

  /// @brief File handle for ION memory allocator.
  int fd_ion_;

  /// @brief ION heap selector.
  uint32_t dma_heap_id_mask_;
};
