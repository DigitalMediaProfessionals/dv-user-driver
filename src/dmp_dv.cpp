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
/// @brief Shared library exported functions implementation.

#include <string.h>
#include <stdarg.h>

#include "dmp_dv.h"
#include "common.h"
#include "context.hpp"
#include "mem.hpp"
#include "cmdlist.hpp"
#include "cmdlist_conv.hpp"
#include "cmdlist_fc.hpp"
#include "cmdlist_ipu.hpp"
#include "cmdlist_maximizer.hpp"


/// @brief Creators for the specific device types.
CDMPDVCmdListDeviceHelper* (*CDMPDVCmdListDeviceHelper::creators_[DMP_DV_DEV_COUNT])(CDMPDVContext *ctx) = {
    NULL,
    CDMPDVCmdListConvHelper::Create,
    CDMPDVCmdListFCHelper::Create,
    CDMPDVCmdListIPUHelper::Create,
    CDMPDVCmdListMaximizerHelper::Create
};


/// @brief Total per-process allocated device-accessible memory size in bytes instantiation.
int64_t CDMPDVMem::total_size_ = 0;


extern "C" {


/// @brief Last error message (instantiation).
char s_last_error_message[256];

/// @brief Verbosity level for debug messages.
static int s_verbosity_level = -1;


const char *dmp_dv_get_last_error_message() {
  return s_last_error_message;
}


void dmp_dv_set_last_error_message(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vsnprintf(s_last_error_message, sizeof(s_last_error_message), format, args);
  if (s_verbosity_level == -1) {
    const char *s = getenv("VERBOSITY");
    s_verbosity_level = s ? atoi(s) : 0;
  }
  if (s_verbosity_level >= 1) {
    fprintf(stderr, "%s\n", s_last_error_message);
    fflush(stderr);
  }
}


const char *dmp_dv_get_version_string() {
  return "7.2.20201106";
}


dmp_dv_context dmp_dv_context_create() {
  CDMPDVContext *ctx = new CDMPDVContext();
  if (!ctx) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDMPDVContext));
    return NULL;
  }
  if (!ctx->Initialize()) {
    delete ctx;
    return NULL;
  }
  return (dmp_dv_context)ctx;
}


const char *dmp_dv_context_get_info_string(dmp_dv_context ctx) {
  if (!ctx) {
    SET_ERR("Invalid argument: ctx is NULL");
    return "";
  }
  return ((CDMPDVContext*)ctx)->GetInfoString();
}


int dmp_dv_context_get_info(dmp_dv_context ctx, struct dmp_dv_info *info) {
  if (!ctx) {
    SET_ERR("Invalid argument: ctx is NULL");
    return EINVAL;
  }
  return ((CDMPDVContext*)ctx)->GetInfo(info);
}


int dmp_dv_context_release(dmp_dv_context ctx) {
  if (!ctx) {
    return 0;
  }
  return ((CDMPDVContext*)ctx)->Release();
}


int dmp_dv_context_retain(dmp_dv_context ctx) {
  if (!ctx) {
    return 0;
  }
  return ((CDMPDVContext*)ctx)->Retain();
}


dmp_dv_mem dmp_dv_mem_alloc(dmp_dv_context ctx, size_t size) {
  CDMPDVMem *mem = new CDMPDVMem();
  if (!mem) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDMPDVMem));
    return NULL;
  }
  if (!mem->Initialize((CDMPDVContext*)ctx, size)) {
    mem->Release();
    return NULL;
  }

  return (dmp_dv_mem)mem;
}


int dmp_dv_mem_release(dmp_dv_mem mem) {
  if (!mem) {
    return 0;
  }
  return ((CDMPDVMem*)mem)->Release();
}


int dmp_dv_mem_retain(dmp_dv_mem mem) {
  if (!mem) {
    return 0;
  }
  return ((CDMPDVMem*)mem)->Retain();
}


uint8_t *dmp_dv_mem_map(dmp_dv_mem mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return NULL;
  }
  return ((CDMPDVMem*)mem)->Map();
}


void dmp_dv_mem_unmap(dmp_dv_mem mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return;
  }
  ((CDMPDVMem*)mem)->Unmap();
}


int dmp_dv_mem_sync_start(dmp_dv_mem mem, int rd, int wr) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return EINVAL;
  }
  return ((CDMPDVMem*)mem)->SyncStart(rd, wr);
}


int dmp_dv_mem_sync_end(dmp_dv_mem mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return EINVAL;
  }
  return ((CDMPDVMem*)mem)->SyncEnd();
}


size_t dmp_dv_mem_get_size(dmp_dv_mem mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return 0;
  }
  return ((CDMPDVMem*)mem)->get_size();
}


int64_t dmp_dv_mem_get_total_size() {
  return CDMPDVMem::get_total_size();
}


int dmp_dv_mem_to_device(dmp_dv_mem mem, size_t offs, size_t size, int flags) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return 0;
  }
  return ((CDMPDVMem*)mem)->ToDevice(offs, size, flags);
}


int dmp_dv_mem_to_cpu(dmp_dv_mem mem, size_t offs, size_t size, int flags) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return 0;
  }
  return ((CDMPDVMem*)mem)->ToCPU(offs, size, flags);
}


dmp_dv_cmdlist dmp_dv_cmdlist_create(dmp_dv_context ctx) {
  CDMPDVCmdList *cmdlist = new CDMPDVCmdList();
  if (!cmdlist) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDMPDVCmdList));
    return NULL;
  }
  if (!cmdlist->Initialize((CDMPDVContext*)ctx)) {
    cmdlist->Release();
    return NULL;
  }
  return (dmp_dv_cmdlist)cmdlist;
}


int dmp_dv_cmdlist_release(dmp_dv_cmdlist cmdlist) {
  if (!cmdlist) {
    return 0;
  }
  return ((CDMPDVCmdList*)cmdlist)->Release();
}


int dmp_dv_cmdlist_retain(dmp_dv_cmdlist cmdlist) {
  if (!cmdlist) {
    return 0;
  }
  return ((CDMPDVCmdList*)cmdlist)->Retain();
}


int dmp_dv_cmdlist_commit(dmp_dv_cmdlist cmdlist) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return EINVAL;
  }
  return ((CDMPDVCmdList*)cmdlist)->Commit();
}


int64_t dmp_dv_cmdlist_exec(dmp_dv_cmdlist cmdlist) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return EINVAL;
  }
  return ((CDMPDVCmdList*)cmdlist)->Exec();
}


int dmp_dv_cmdlist_wait(dmp_dv_cmdlist cmdlist, int64_t exec_id) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return EINVAL;
  }
  return ((CDMPDVCmdList*)cmdlist)->Wait(exec_id);
}


int64_t dmp_dv_cmdlist_get_last_exec_time(dmp_dv_cmdlist cmdlist) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return -1;
  }
  return ((CDMPDVCmdList*)cmdlist)->GetLastExecTime();
}


int dmp_dv_cmdlist_add_raw(dmp_dv_cmdlist cmdlist, struct dmp_dv_cmdraw *cmd) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return EINVAL;
  }
  return ((CDMPDVCmdList*)cmdlist)->AddRaw(cmd);
}


int dmp_dv_device_exists(dmp_dv_context ctx, int dev_type_id) {
  if(!ctx) {
    SET_ERR("Invalid argument: ctx is NULL");
    return -1;
  }
  return ((CDMPDVContext*)ctx)->DeviceExists(dev_type_id);
}


int dmp_dv_fpga_device_exists(dmp_dv_context ctx, int dev_type_id) {
  return dmp_dv_device_exists(ctx, dev_type_id);
}

}  // extern "C"
