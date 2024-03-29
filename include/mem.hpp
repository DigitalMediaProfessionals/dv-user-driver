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
/// @brief dmp_dv_mem implementation.
#pragma once

#include "context.hpp"


#define CACHE_LINE_SIZE 64
#define CACHE_LINE_LOG2 6


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
      SET_IOCTL_ERR(res, "/dev/ion", "ION_IOC_ALLOC");
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
    __sync_add_and_fetch(&total_size_, (int64_t)real_size_);

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
      __sync_add_and_fetch(&total_size_, -(int64_t)real_size_);
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
    void *ptr = mmap(NULL, real_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem_, 0);
    if (ptr == MAP_FAILED) {
      SET_ERR("mmap() on allocated from /dev/ion file descriptor failed for %zu bytes", real_size_);
      return NULL;
    }
    map_ptr_ = (uint8_t*)ptr;
    return map_ptr_;
  }

  /// @brief Unmaps allocated memory from user address space.
  void Unmap() {
    if (!map_ptr_) {
      return;
    }
    SyncEnd();
    munmap(map_ptr_, real_size_);
    map_ptr_ = NULL;
  }

  /// @brief Starts CPU <-> Device memory syncronization.
  int SyncStart(int rd, int wr) {
    if (!map_ptr_) {
      SET_ERR("Memory must be mapped before starting synchronization");
      return EINVAL;
    }
    int new_sync_flags = (rd ? DMA_BUF_SYNC_READ : 0) | (wr ? DMA_BUF_SYNC_WRITE : 0);
    if (!new_sync_flags) {
      SET_ERR("Invalid arguments: either rd or wr must be non-zero");
      return EINVAL;
    }
    if ((sync_flags_ | new_sync_flags) == sync_flags_) {  // already mapped with same or greater set of flags
      return 0;
    }
    int res = SyncEnd();
    if (res) {
      return res;
    }
    sync_flags_ = new_sync_flags;
    struct dma_buf_sync sync_args;
    memset(&sync_args, 0, sizeof(sync_args));
    sync_args.flags = DMA_BUF_SYNC_START | sync_flags_;
    res = ioctl(fd_mem_, DMA_BUF_IOCTL_SYNC, &sync_args);
    if (res < 0) {
      SET_IOCTL_ERR(res, "/dev/ion", "DMA_BUF_SYNC_START");
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
      SET_IOCTL_ERR(res, "/dev/ion", "DMA_BUF_SYNC_END");
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
  static inline int get_fd(dmp_dv_mem mem) {
    return mem ? ((CDMPDVMem*)mem)->fd_mem_ : -1;
  }

  /// @brief Returns total per-process allocated device-accessible memory size in bytes.
  static inline int64_t get_total_size() {
    return __sync_add_and_fetch(&total_size_, 0);
  }

  /// @brief Returns pointer to mapped memory.
  inline uint8_t *get_ptr() const {
    return map_ptr_;
  }

  /// @brief Returns sync flags.
  inline int get_sync_flags() const {
    return sync_flags_;
  }

  int ToDevice(size_t offs, size_t size, int flags) {
    if (offs + size > real_size_) {
      SET_ERR("Invalid memory range specified: offs=%zu size=%zu while memory buffer size is %zu",
              offs, size, real_size_);
      return EINVAL;
    }
    if (!size) {
      return 0;
    }
    if (!map_ptr_) {
      SET_ERR("Memory must be mapped before starting synchronization");
      return EINVAL;
    }
    uint8_t *end = map_ptr_ + offs + size;
#ifdef __aarch64__
    if (flags & DMP_DV_MEM_CPU_WONT_READ) {
      for (uint8_t *addr = (uint8_t*)((((size_t)(map_ptr_ + offs)) >> CACHE_LINE_LOG2) << CACHE_LINE_LOG2);
           addr < end; addr += CACHE_LINE_SIZE) {
        asm("DC CIVAC, %0" /* Write changes to RAM and Invalidate cache */
            : /* No outputs */
            : "r" (addr));
      }
    }
    else {
      for (uint8_t *addr = (uint8_t*)((((size_t)(map_ptr_ + offs)) >> CACHE_LINE_LOG2) << CACHE_LINE_LOG2);
           addr < end; addr += CACHE_LINE_SIZE) {
        asm("DC CVAC, %0" /* Write changes to RAM and Leave data in cache */
            : /* No outputs */
            : "r" (addr));
      }
    }
    asm("DSB SY");  // data sync barrier
#else
    __builtin___clear_cache(map_ptr_ + offs, end);
#endif
    return 0;
  }

  int ToCPU(size_t offs, size_t size, int flags) {
    if (flags & DMP_DV_MEM_CPU_HADNT_READ) {
      return 0;
    }
    if (offs + size > real_size_) {
      SET_ERR("Invalid memory range specified: offs=%zu size=%zu while memory buffer size is %zu",
              offs, size, real_size_);
      return EINVAL;
    }
    if (!size) {
      return 0;
    }
    if (!map_ptr_) {
      SET_ERR("Memory must be mapped before starting synchronization");
      return EINVAL;
    }
    uint8_t *end = map_ptr_ + offs + size;
#ifdef __aarch64__
    for (uint8_t *addr = (uint8_t*)((((size_t)(map_ptr_ + offs)) >> CACHE_LINE_LOG2) << CACHE_LINE_LOG2);
         addr < end; addr += CACHE_LINE_SIZE) {
      asm("DC CIVAC, %0"
          : /* No outputs. */
          : "r" (addr));
    }
    asm("DSB SY");  // data sync barrier
#else
    __builtin___clear_cache(map_ptr_ + offs, end);
#endif
    return 0;
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

  /// @brief Total per-process allocated device-accessible memory size in bytes.
  static int64_t total_size_;
};
