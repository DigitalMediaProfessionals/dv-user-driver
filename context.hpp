/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief dv_context implementation.
 */
#pragma once

#include "common.hpp"


/// @brief Implementation of dv_context.
class CDVContext {
 public:
  CDVContext() {
    fd_ion_ = -1;
    dma_heap_id_mask_ = 0;
  }

  virtual ~CDVContext() {
    Cleanup();
  }

  void Cleanup() {
    if (fd_ion_ != -1) {
      close(fd_ion_);
      fd_ion_ = -1;
    }
    dma_heap_id_mask_ = 0;
  }

  bool Initialize(const char *path) {
    Cleanup();

    path_ = path ? path : "";

    fd_ion_ = open("/dev/ion", O_RDONLY | O_CLOEXEC);  // O_CLOEXEC is suggested for security
    if (fd_ion_ < 0) {
      SET_ERR("open() failed for /dev/ion");
      return false;
    }

    // Get ION heap counts
    struct ion_heap_query query;
    memset(&query, 0, sizeof(query));
    int res = ioctl(fd_ion_, ION_IOC_HEAP_QUERY, &query);
    if (res < 0) {
      SET_ERR("ioctl() failed for ION_IOC_HEAP_QUERY");
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
      SET_ERR("ioctl() failed for ION_IOC_HEAP_QUERY");
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

    info_ = std::string("DV700: UBUF=640Kb PATH=") + path_;

    return true;
  }

  inline int get_fd_ion() const {
    return fd_ion_;
  }

  inline uint32_t get_dma_heap_id_mask() const {
    return dma_heap_id_mask_;
  }

  int Sync() {
    // TODO: implement.
    return 0;
  }

  const char *GetInfoString() {
    return info_.c_str();
  }

 private:
  /// @brief Path to the device.
  std::string path_;

  /// @brief Device information.
  std::string info_;

  /// @brief File handle for ION memory allocator.
  int fd_ion_;

  /// @brief ION heap selector.
  uint32_t dma_heap_id_mask_;
};
