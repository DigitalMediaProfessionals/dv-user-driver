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
class CDMPDVContext : public virtual CDMPDVBase {
 public:
  /// @brief Constructor.
  CDMPDVContext() : CDMPDVBase() {
    pthread_mutex_init(&mt_last_exec_id_, NULL);
    fd_ion_ = -1;
    dma_heap_id_mask_ = 0;
    last_exec_id_ = -1;
    last_wait_id_ = -1;
    fd_conv_ = -1;
    ub_size_ = 0;
    max_kernel_size_ = 3;
    conv_freq_ = 0;
    fc_freq_ = 0;
  }

  /// @brief Called when the object is about to be destroyed.
  virtual ~CDMPDVContext() {
    Cleanup();
    pthread_mutex_destroy(&mt_last_exec_id_);
  }

  /// @brief Releases held resources.
  virtual void Cleanup() {
    if (fd_conv_ != -1) {
      close(fd_conv_);
      fd_conv_ = -1;
    }
    if (fd_ion_ != -1) {
      close(fd_ion_);
      fd_ion_ = -1;
    }
    dma_heap_id_mask_ = 0;
    last_exec_id_ = -1;
  }

  /// @brief Fills info string for debugging purposes.
  virtual void fill_debug_info(char *info, int length) {
    snprintf(info, length, "dmp_dv_context: addr=%zu, n_ref=%d", (size_t)this, n_ref_);
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

    fd_conv_ = open("/dev/dv_conv", O_RDONLY | O_CLOEXEC);
    if (fd_conv_ == -1) {
      SET_ERR("open() failed for /dev/dv_conv: %s", strerror(errno));
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

  /// @brief Acquires lock for working with last execution id.
  inline void last_exec_id_lock() {
    pthread_mutex_lock(&mt_last_exec_id_);
  }

  /// @brief Assigns last execution id.
  /// @details Must be called inside last_exec_id_lock(), last_exec_id_unlock().
  inline void set_last_exec_id(int64_t exec_id) {
    last_exec_id_ = exec_id;
  }

  /// @brief Releases lock for working with last execution id.
  inline void last_exec_id_unlock() {
    pthread_mutex_unlock(&mt_last_exec_id_);
  }

  /// @brief Waits for the specific execution id to be completed.
  int Wait(int64_t exec_id) {
    last_exec_id_lock();
    int last_wait_id = last_wait_id_;
    if (exec_id < 0) {
      exec_id = last_exec_id_;
    }
    last_exec_id_unlock();
    if ((exec_id < 0) || (exec_id <= last_wait_id)) {
      return 0;
    }
    for (;;) {
      int res = ioctl(fd_conv_, DMP_DV_IOC_WAIT, &exec_id);
      if (!res) {
        break;
      }
      switch (res) {
        case -EBUSY:       // timeout of 2 seconds reached
        case ERESTARTSYS:  // signal has interrupted the wait
          continue;  // repeat ioctl

        default:
          SET_IOCTL_ERR("/dev/dv_conv", "DMP_DV_IOC_WAIT");
          return res;
      }
    }
    last_exec_id_lock();
    last_wait_id_ = std::max(last_wait_id_, exec_id);
    last_exec_id_unlock();
    return 0;
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

  /// @brief Handle to the DV accelerator.
  int fd_conv_;

  /// @brief Id of the last executed command.
  int64_t last_exec_id_;

  /// @brief Id of the last succeeded waited command.
  int64_t last_wait_id_;
};
