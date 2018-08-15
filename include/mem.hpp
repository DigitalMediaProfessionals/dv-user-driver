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
/*
 * @brief dmp_dv_mem implementation.
 */
#pragma once

#include "context.hpp"


/// @brief Implementation of dmp_dv_mem.
class CDMPDVMem : public CDMPDVBase {
 public:
  /// @brief Constructor.
  CDMPDVMem() : CDMPDVBase() {
    ctx_ = NULL;
    fd_mem_ = -1;
    requested_size_ = 0;
    real_size_ = 0;
    map_ptr_ = NULL;
    sync_flags_ = 0;
  }

  /// @brief Destructor.
  virtual ~CDMPDVMem() {
    Cleanup();
  }

  /// @brief Allocates memory accessible by device associated with the provided context.
  bool Initialize(CDMPDVContext *ctx, size_t size) {
    Cleanup();
    if (!ctx) {
      SET_ERR("Invalid argument: ctx is NULL");
      return NULL;
    }

    // Try to allocate a buffer
    struct ion_allocation_data alloc_param;
    memset(&alloc_param, 0, sizeof(alloc_param));
    alloc_param.len = size;
    alloc_param.heap_id_mask = ctx->get_dma_heap_id_mask();
    alloc_param.flags = ION_FLAG_CACHED;
    int res = ioctl(ctx->get_fd_ion(), ION_IOC_ALLOC, &alloc_param);
    if (res < 0) {
      SET_IOCTL_ERR("/dev/ion", "ION_IOC_ALLOC");
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

    ctx->Retain();
    ctx_ = ctx;

    return true;
  }

  /// @brief Releases held resources.
  void Cleanup() {
    Unmap();
    if (fd_mem_ != -1) {
      close(fd_mem_);
      fd_mem_ = -1;
    }
    requested_size_ = 0;
    real_size_ = 0;
    if (ctx_) {
      ctx_->Release();
      ctx_ = NULL;
    }
  }

  /// @brief Maps allocated memory to user address space with READ and WRITE permissions.
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

  /// @brief Unmaps allocated memory from user address space.
  void Unmap() {
    SyncEnd();
    if (!map_ptr_) {
      return;
    }
    munmap(map_ptr_, real_size_);
    map_ptr_ = NULL;
  }

  /// @brief Starts CPU <-> Device memory syncronization.
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
      SET_IOCTL_ERR("/dev/ion", "DMA_BUF_SYNC_START");
      return res;
    }
    return 0;
  }

  /// @brief Ends CPU <-> Device memory syncronization.
  int SyncEnd() {
    if (!sync_flags_) {
      return 0;
    }
    struct dma_buf_sync sync_args;
    memset(&sync_args, 0, sizeof(sync_args));
    sync_args.flags = DMA_BUF_SYNC_END | sync_flags_;
    int res = ioctl(fd_mem_, DMA_BUF_IOCTL_SYNC, &sync_args);
    if (res < 0) {
      SET_IOCTL_ERR("/dev/ion", "DMA_BUF_SYNC_END");
      return res;
    }
    sync_flags_ = 0;
    return 0;
  }

  /// @brief Returns real size of the allocated memory which can be greater than requested.
  inline size_t get_size() const {
    return real_size_;
  }

  /// @brief Returns file descriptor from memory handle or -1 when memory handle is NULL.
  static inline int get_fd(dmp_dv_mem *mem) {
    return mem ? ((CDMPDVMem*)mem)->fd_mem_ : -1;
  }

 private:
  /// @brief Pointer to dv context.
  CDMPDVContext *ctx_;

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
