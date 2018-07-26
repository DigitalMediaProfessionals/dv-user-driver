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

#include "common.h"

#include <string>


/// @brief Implementation of dmp_dv_context.
class CDMPDVContext {
 public:
  CDMPDVContext() {
    // TODO: add reference counter.

    fd_ion_ = -1;
    dma_heap_id_mask_ = 0;

    last_executed_cmdlist_ = NULL;
  }

  virtual ~CDMPDVContext() {
    Cleanup();
  }

  void Release() {
    // TODO: adjust when reference counter will be implemented.
    delete this;
  }

  void Cleanup() {
    last_executed_cmdlist_ = NULL;
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

    return true;
  }

  inline int get_fd_ion() const {
    return fd_ion_;
  }

  inline uint32_t get_dma_heap_id_mask() const {
    return dma_heap_id_mask_;
  }

  void SetLastExecutedCmdList(dmp_dv_cmdlist *cmdlist) {
    last_executed_cmdlist_ = cmdlist;
    // TODO: add proper critical section.
  }

  int WaitAll() {
    if (last_executed_cmdlist_) {
      dmp_dv_cmdlist_wait(last_executed_cmdlist_, -1);
      last_executed_cmdlist_ = NULL;
      // TODO: decrease reference counter and add proper critical section.
    }
    return 0;
  }

  const char *GetInfoString() {
    if (!info_.length()) {
      info_ = std::string("DV700: UBUF=640Kb PATH=") + path_;
    }
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

  /// @brief Last executed command list.
  dmp_dv_cmdlist *last_executed_cmdlist_;
};
