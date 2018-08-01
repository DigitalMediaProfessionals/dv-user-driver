/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief dmp_dv_context implementation.
 */
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
  /// @param path Path to the device, can be NULL or empty. Currently returns default device regardless of path.
  bool Initialize(const char *path) {
    Cleanup();

    path_ = path ? path : "";

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
      SET_IOCTL_ERR("/dev/ion", "ION_IOC_HEAP_QUERY");
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
      SET_IOCTL_ERR("/dev/ion", "ION_IOC_HEAP_QUERY");
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

    ub_size_ = sysfs_read_int("/sys/devices/platform/dmp_dv/ub_size", 0);
    max_kernel_size_ = sysfs_read_int("/sys/devices/platform/dmp_dv/max_kernel_size", 3);
    conv_freq_ = sysfs_read_int("/sys/devices/platform/dmp_dv/conv_freq", 0);
    fc_freq_ = sysfs_read_int("/sys/devices/platform/dmp_dv/fc_freq", 0);

    return true;
  }

  /// @brief Reads single int value from sysfs file.
  int sysfs_read_int(const char *path, int def) {
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
  const char *GetInfoString() {
    if (!info_.length()) {
      char s_path[256];
      if (path_.length()) {
        snprintf(s_path, sizeof(s_path), " (%s)", path_.c_str());
      }
      else {
        s_path[0] = 0;
      }
      char s[256];
      snprintf(s, sizeof(s), "DMP DV%s: ub_size=%d max_kernel_size=%d conv_freq=%d fc_freq=%d",
               s_path, ub_size_, max_kernel_size_, conv_freq_, fc_freq_);
      info_ = s;
    }
    return info_.c_str();
  }

  /// @brief Returns maximum supported kernel size.
  inline int get_max_kernel_size() const {
    return max_kernel_size_;
  }

  int GetInfo(dmp_dv_info *p_info) {
    if (p_info->size < 8) {
      SET_ERR("Invalid argument: info->size is too small: %u", p_info->size);
      return -1;
    }
    p_info->version = 0;
    if (p_info->size >= sizeof(dmp_dv_info_v0)) {
      dmp_dv_info_v0 *info = (dmp_dv_info_v0*)p_info;
      info->ub_size = ub_size_;
      info->max_kernel_size = max_kernel_size_;
      info->conv_freq = conv_freq_;
      info->fc_freq = fc_freq_;
    }
    return 0;
  }

 private:
  /// @brief Mutex for critical section for setting last execution id.
  pthread_mutex_t mt_last_exec_id_;

  /// @brief Path to the device.
  std::string path_;

  /// @brief Size of the Unified Buffer.
  int ub_size_;

  /// @brief Size of the maximum kernel.
  int max_kernel_size_;

  /// @brief Convolutional block frequency in MHz.
  int conv_freq_;

  /// @brief Fully Connected block frequency in MHz.
  int fc_freq_;

  /// @brief Device information.
  std::string info_;

  /// @brief File handle for ION memory allocator.
  int fd_ion_;

  /// @brief ION heap selector.
  uint32_t dma_heap_id_mask_;
};
