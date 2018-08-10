/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief Shared library exported functions implementation.
 */
#include "dmp_dv.h"
#include "common.h"
#include "context.hpp"
#include "mem.hpp"
#include "cmdlist.hpp"


/// @brief Creators for the specific device types.
f_device_helper_creator CDMPDVCmdListDeviceHelper::creators_[DMP_DV_DEV_COUNT] = {
    NULL,
    CDMPDVCmdListConvHelper::Create,
    NULL
};


extern "C" {


/// @brief Last error message (instantiation).
char s_last_error_message[256];


const char *dmp_dv_get_last_error_message() {
  return s_last_error_message;
}


const char *dmp_dv_get_version_string() {
  return "0.1.0 Initial release.";
}


dmp_dv_context* dmp_dv_context_create() {
  CDMPDVContext *ctx = new CDMPDVContext();
  if (!ctx) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDMPDVContext));
    return NULL;
  }
  if (!ctx->Initialize()) {
    delete ctx;
    return NULL;
  }
  return (dmp_dv_context*)ctx;
}


const char *dmp_dv_context_get_info_string(dmp_dv_context* ctx) {
  if (!ctx) {
    SET_ERR("Invalid argument: ctx is NULL");
    return "";
  }
  return ((CDMPDVContext*)ctx)->GetInfoString();
}


int dmp_dv_context_get_info(dmp_dv_context* ctx, dmp_dv_info *info) {
  if (!ctx) {
    SET_ERR("Invalid argument: ctx is NULL");
    return EINVAL;
  }
  return ((CDMPDVContext*)ctx)->GetInfo(info);
}


void dmp_dv_context_release(dmp_dv_context *ctx) {
  if (!ctx) {
    return;
  }
  ((CDMPDVContext*)ctx)->Release();
}


void dmp_dv_context_retain(dmp_dv_context *ctx) {
  if (!ctx) {
    return;
  }
  ((CDMPDVContext*)ctx)->Retain();
}


dmp_dv_mem* dmp_dv_mem_alloc(dmp_dv_context *ctx, size_t size) {
  CDMPDVMem *mem = new CDMPDVMem();
  if (!mem) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDMPDVMem));
    return NULL;
  }
  if (!mem->Initialize((CDMPDVContext*)ctx, size)) {
    mem->Release();
    return NULL;
  }

  return (dmp_dv_mem*)mem;
}


void dmp_dv_mem_release(dmp_dv_mem *mem) {
  if (!mem) {
    return;
  }
  ((CDMPDVMem*)mem)->Release();
}


void dmp_dv_mem_retain(dmp_dv_mem *mem) {
  if (!mem) {
    return;
  }
  ((CDMPDVMem*)mem)->Retain();
}


uint8_t *dmp_dv_mem_map(dmp_dv_mem *mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return NULL;
  }
  return ((CDMPDVMem*)mem)->Map();
}


void dmp_dv_mem_unmap(dmp_dv_mem *mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return;
  }
  ((CDMPDVMem*)mem)->Unmap();
}


int dmp_dv_mem_sync_start(dmp_dv_mem *mem, int rd, int wr) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return EINVAL;
  }
  return ((CDMPDVMem*)mem)->SyncStart(rd, wr);
}


int dmp_dv_mem_sync_end(dmp_dv_mem *mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return EINVAL;
  }
  return ((CDMPDVMem*)mem)->SyncEnd();
}


size_t dmp_dv_mem_get_size(dmp_dv_mem *mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return 0;
  }
  return ((CDMPDVMem*)mem)->get_size();
}


dmp_dv_cmdlist *dmp_dv_cmdlist_create(dmp_dv_context *ctx) {
  CDMPDVCmdList *cmdlist = new CDMPDVCmdList();
  if (!cmdlist) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDMPDVCmdList));
    return NULL;
  }
  if (!cmdlist->Initialize((CDMPDVContext*)ctx)) {
    cmdlist->Release();
    return NULL;
  }
  return (dmp_dv_cmdlist*)cmdlist;
}


void dmp_dv_cmdlist_release(dmp_dv_cmdlist *cmdlist) {
  if (!cmdlist) {
    return;
  }
  ((CDMPDVCmdList*)cmdlist)->Release();
}


void dmp_dv_cmdlist_retain(dmp_dv_cmdlist *cmdlist) {
  if (!cmdlist) {
    return;
  }
  ((CDMPDVCmdList*)cmdlist)->Retain();
}


int dmp_dv_cmdlist_commit(dmp_dv_cmdlist *cmdlist) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return EINVAL;
  }
  return ((CDMPDVCmdList*)cmdlist)->Commit();
}


int64_t dmp_dv_cmdlist_exec(dmp_dv_cmdlist *cmdlist) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return EINVAL;
  }
  return ((CDMPDVCmdList*)cmdlist)->Exec();
}


int dmp_dv_cmdlist_wait(dmp_dv_cmdlist *cmdlist, int64_t exec_id) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return EINVAL;
  }
  return ((CDMPDVCmdList*)cmdlist)->Wait(exec_id);
}


int dmp_dv_cmdlist_add_raw(dmp_dv_cmdlist *cmdlist, dmp_dv_cmdraw *cmd) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return EINVAL;
  }
  return ((CDMPDVCmdList*)cmdlist)->AddRaw(cmd);
}


}  // extern "C"
