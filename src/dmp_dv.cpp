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
 * @brief Shared library exported functions implementation.
 */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

#include "dmp_dv.h"
#include "common.h"
#include "context.hpp"
#include "mem.hpp"
#include "cmdlist.hpp"
#include "cmdlist_conv.hpp"
#include "cmdlist_fc.hpp"
#include "cmdlist_ipu.hpp"


/// @brief Creators for the specific device types.
CDMPDVCmdListDeviceHelper* (*CDMPDVCmdListDeviceHelper::creators_[DMP_DV_DEV_COUNT])(CDMPDVContext *ctx) = {
    NULL,
    CDMPDVCmdListConvHelper::Create,
    CDMPDVCmdListFCHelper::Create,
    CDMPDVCmdListIPUHelper::Create
};


/// @brief Total per-process allocated device-accessible memory size in bytes instantiation.
int64_t CDMPDVMem::total_size_ = 0;


extern "C" {


/// @brief Last error message (instantiation).
char s_last_error_message[256];


const char *dmp_dv_get_last_error_message() {
  return s_last_error_message;
}


const char *dmp_dv_get_version_string() {
  return "0.3.2 20181101";
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


void dmp_dv_context_release(dmp_dv_context ctx) {
  if (!ctx) {
    return;
  }
  ((CDMPDVContext*)ctx)->Release();
}


void dmp_dv_context_retain(dmp_dv_context ctx) {
  if (!ctx) {
    return;
  }
  ((CDMPDVContext*)ctx)->Retain();
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


void dmp_dv_mem_release(dmp_dv_mem mem) {
  if (!mem) {
    return;
  }
  ((CDMPDVMem*)mem)->Release();
}


void dmp_dv_mem_retain(dmp_dv_mem mem) {
  if (!mem) {
    return;
  }
  ((CDMPDVMem*)mem)->Retain();
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


void dmp_dv_cmdlist_release(dmp_dv_cmdlist cmdlist) {
  if (!cmdlist) {
    return;
  }
  ((CDMPDVCmdList*)cmdlist)->Release();
}


void dmp_dv_cmdlist_retain(dmp_dv_cmdlist cmdlist) {
  if (!cmdlist) {
    return;
  }
  ((CDMPDVCmdList*)cmdlist)->Retain();
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


int dmp_dv_cmdlist_add_raw(dmp_dv_cmdlist cmdlist, struct dmp_dv_cmdraw *cmd) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return EINVAL;
  }
  return ((CDMPDVCmdList*)cmdlist)->AddRaw(cmd);
}

int dmp_dv_fpga_device_exists(dmp_dv_context ctx, int dev_type_id) {
  if(!ctx) {
    return -1;
  }
  switch (dev_type_id) {
    case DMP_DV_DEV_CONV:
      return ((CDMPDVContext*)ctx)->get_conv_freq() ? 1 : 0;
    case DMP_DV_DEV_FC:
      return ((CDMPDVContext*)ctx)->get_fc_freq() ? 1 : 0;
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
      return -1;
  }
}

}  // extern "C"
