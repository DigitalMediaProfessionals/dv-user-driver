/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief dv_mem implementation.
 */
#pragma once

#include "context.hpp"


/// @brief Implementation of dv_mem.
class CDVMem {
 public:
  CDVMem() {
    ctx_ = NULL;
    fd_mem_ = -1;
    requested_size_ = 0;
    real_size_ = 0;
    map_ptr_ = NULL;
    sync_flags_ = 0;
  }

  virtual ~CDVMem() {
    Cleanup();
  }

  bool Initialize(CDVContext *ctx, size_t size) {
    Cleanup();
    if (!ctx) {
      SET_ERR("Invalid argument: ctx is NULL");
      return NULL;
    }
    ctx_ = ctx;

    // Try to allocate a buffer
    struct ion_allocation_data alloc_param;
    memset(&alloc_param, 0, sizeof(alloc_param));
    alloc_param.len = size;
    alloc_param.heap_id_mask = ctx->get_dma_heap_id_mask();
    alloc_param.flags = ION_FLAG_CACHED;
    int res = ioctl(ctx->get_fd_ion(), ION_IOC_ALLOC, &alloc_param);
    if (res < 0) {
      SET_ERR("ioctl() for ION_IOC_ALLOC failed to allocate %zu bytes", size);
      return false;
    }
    fd_mem_ = alloc_param.fd;
    requested_size_ = size;
    off_t buf_size = lseek(fd_mem_, 0, SEEK_END);
    if ((buf_size < 0) || ((size_t)buf_size < size)) {
      SET_ERR("Could not confirm size of allocated continuous memory for %zu bytes", size);
      return false;
    }
    if (lseek(fd_mem_, 0, SEEK_SET)) {
      SET_ERR("Could not confirm size of allocated continuous memory for %zu bytes", size);
      return false;
    }
    real_size_ = buf_size;

    // TODO: increase reference counter on context.

    return true;
  }

  void Cleanup() {
    Unmap();
    if (fd_mem_ != -1) {
      close(fd_mem_);
      fd_mem_ = -1;
    }
    requested_size_ = 0;
    real_size_ = 0;
    ctx_ = NULL;
  }

  uint8_t* Map() {
    if (map_ptr_) {
      return map_ptr_;
    }
    map_ptr_ = (uint8_t*)mmap(NULL, real_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem_, 0);
    if (map_ptr_ == MAP_FAILED) {
      SET_ERR("mmap() on allocated from /dev/ion file descriptor failed for %zu bytes", real_size_);
      return NULL;
    }
    return map_ptr_;
  }

  void Unmap() {
    SyncEnd();
    if (!map_ptr_) {
      return;
    }
    munmap(map_ptr_, real_size_);
    map_ptr_ = NULL;
  }

  int SyncStart(int rd, int wr) {
    int res = SyncEnd();
    if (res) {
      return res;
    }
    if (!map_ptr_) {
      return 0;
    }
    sync_flags_ = (rd ? DMA_BUF_SYNC_READ : 0) | (wr ? DMA_BUF_SYNC_WRITE : 0);
    if (!sync_flags_) {
      SET_ERR("Invalid arguments: either rd or wr should be non-zero");
      return -1;
    }
    struct dma_buf_sync sync_args;
    memset(&sync_args, 0, sizeof(sync_args));
    sync_args.flags = DMA_BUF_SYNC_START | sync_flags_;
    res = ioctl(fd_mem_, DMA_BUF_IOCTL_SYNC, &sync_args);
    if (res < 0) {
      SET_ERR("ioctl() for DMA_BUF_SYNC_START failed");
      return res;
    }
    return 0;
  }

  int SyncEnd() {
    if (!sync_flags_) {
      return 0;
    }
    struct dma_buf_sync sync_args;
    memset(&sync_args, 0, sizeof(sync_args));
    sync_args.flags = DMA_BUF_SYNC_END | sync_flags_;
    int res = ioctl(fd_mem_, DMA_BUF_IOCTL_SYNC, &sync_args);
    if (res < 0) {
      SET_ERR("ioctl() for DMA_BUF_SYNC_END failed");
      return res;
    }
    sync_flags_ = 0;
    return 0;
  }

  inline size_t get_size() const {
    return real_size_;
  }

 private:
  /// @brief Pointer to dv context.
  CDVContext *ctx_;

  /// @brief File handle for allocated memory.
  int fd_mem_;

  /// @brief Requested size of allocated memory.
  size_t requested_size_;

  /// @brief Real size of allocated memory.
  size_t real_size_;

  /// @brief Mapped memory pointer.
  uint8_t *map_ptr_;

  /// @brief Last used DMA synchronization flags.
  int sync_flags_;
};
