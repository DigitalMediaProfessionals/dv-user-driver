/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
#include <stdio.h>
#include <string.h>

#include <string>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include <linux/dma-buf.h>
#include "ion.h"

#include "dv.h"


/// @brief Last error message.
static char s_last_error_message[256];


/// @brief Helper to set the last error message.
#define SET_ERR(...) snprintf(s_last_error_message, sizeof(s_last_error_message), __VA_ARGS__)


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


/// @brief Implementation of dv_mem.
class CDVMem {
 public:
  CDVMem() {
    ctx_ = NULL;
    fd_mem_ = -1;
    size_ = 0;
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
    size_ = size;
    off_t buf_size = lseek(fd_mem_, 0, SEEK_END);
    if ((buf_size < 0) || ((size_t)buf_size != size)) {
      SET_ERR("Could not confirm size of allocated continuous memory for %zu bytes", size);
      return false;
    }
    if (lseek(fd_mem_, 0, SEEK_SET)) {
      SET_ERR("Could not confirm size of allocated continuous memory for %zu bytes", size);
      return false;
    }

    return true;
  }

  void Cleanup() {
    Unmap();
    if (fd_mem_ != -1) {
      close(fd_mem_);
      fd_mem_ = -1;
    }
    size_ = 0;
    ctx_ = NULL;
  }

  uint8_t* Map() {
    if (map_ptr_) {
      return map_ptr_;
    }
    map_ptr_ = (uint8_t*)mmap(NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem_, 0);
    if (map_ptr_ == MAP_FAILED) {
      SET_ERR("mmap() on allocated from /dev/ion file descriptor failed for %zu bytes", size_);
      return NULL;
    }
    return map_ptr_;
  }

  void Unmap() {
    SyncEnd();
    if (!map_ptr_) {
      return;
    }
    munmap(map_ptr_, size_);
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

 private:
  /// @brief Pointer to dv context.
  CDVContext *ctx_;

  /// @brief File handle for allocated memory.
  int fd_mem_;

  /// @brief Size of allocated memory.
  size_t size_;

  /// @brief Mapped memory pointer.
  uint8_t *map_ptr_;

  /// @brief Last used DMA synchronization flags.
  int sync_flags_;
};


/// @brief Implementation of dv_cmdlist.
class CDVCmdList {
 public:
  CDVCmdList() {
    ctx_ = NULL;
  }

  virtual ~CDVCmdList() {
    Cleanup();
  }

  bool Initialize(CDVContext *ctx) {
    Cleanup();
    if (!ctx) {
      SET_ERR("Invalid argument: ctx is NULL");
      return NULL;
    }
    ctx_ = ctx;

    return true;
  }

  void Cleanup() {
    ctx_ = NULL;
  }

 private:
  CDVContext *ctx_;
};


extern "C"
const char *dv_get_last_error_message() {
  return s_last_error_message;
}


extern "C"
const char *dv_get_version_string() {
  return "0.1.0 Initial release.";
}


extern "C"
dv_context* dv_context_create(const char *path) {
  CDVContext *ctx = new CDVContext();
  if (!ctx) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDVContext));
    return NULL;
  }
  if (!ctx->Initialize(path)) {
    delete ctx;
    return NULL;
  }
  return (dv_context*)ctx;
}


extern "C"
const char *dv_context_get_info_string(dv_context* ctx) {
  if (!ctx) {
    SET_ERR("Invalid argument: ctx is NULL");
    return "";
  }
  return ((CDVContext*)ctx)->GetInfoString();
}


extern "C"
void dv_context_destroy(dv_context *ctx) {
  if (!ctx) {
    return;
  }
  free(ctx);
}


extern "C"
dv_mem* dv_mem_alloc(dv_context *ctx, size_t size) {
  CDVMem *mem = new CDVMem();
  if (!mem) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDVMem));
    return NULL;
  }
  if (!mem->Initialize((CDVContext*)ctx, size)) {
    delete mem;
    return NULL;
  }

  return (dv_mem*)mem;
}


extern "C"
void dv_mem_free(dv_mem *mem) {
  if (!mem) {
    return;
  }
  delete (CDVMem*)mem;
}


extern "C"
uint8_t *dv_mem_map(dv_mem *mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return NULL;
  }
  return ((CDVMem*)mem)->Map();
}


extern "C"
void dv_mem_unmap(dv_mem *mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return;
  }
  ((CDVMem*)mem)->Unmap();
}


extern "C"
int dv_mem_sync_start(dv_mem *mem, int rd, int wr) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return -1;
  }
  return ((CDVMem*)mem)->SyncStart(rd, wr);
}


extern "C"
int dv_mem_sync_end(dv_mem *mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return -1;
  }
  return ((CDVMem*)mem)->SyncEnd();
}


extern "C"
int dv_sync(dv_context *ctx) {
  if (!ctx) {
    SET_ERR("Invalid argument: ctx is NULL");
    return -1;
  }
  return ((CDVContext*)ctx)->Sync();
}


extern "C"
dv_cmdlist *dv_cmdlist_create(dv_context *ctx) {
  CDVCmdList *cmdlist = new CDVCmdList();
  if (!cmdlist) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDVCmdList));
    return NULL;
  }
  if (!cmdlist->Initialize((CDVContext*)ctx)) {
    delete cmdlist;
    return NULL;
  }
  return (dv_cmdlist*)cmdlist;
}


extern "C"
void dv_cmdlist_destroy(dv_cmdlist *cmdlist) {
  if (!cmdlist) {
    return;
  }
  delete (CDVCmdList*)cmdlist;
}
