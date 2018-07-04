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
#include "dv.h"
#include "common.hpp"
#include "context.hpp"
#include "mem.hpp"
#include "cmdlist.hpp"


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


extern "C"
int dv_cmdlist_end(dv_cmdlist *cmdlist) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return -1;
  }
  return ((CDVCmdList*)cmdlist)->End();
}


extern "C"
int dv_cmdlist_exec(dv_cmdlist *cmdlist) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return -1;
  }
  return ((CDVCmdList*)cmdlist)->Exec();
}


extern "C"
int dv_cmdlist_add_raw(dv_cmdlist *cmdlist, dv_cmdraw *cmd) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return -1;
  }
  return ((CDVCmdList*)cmdlist)->AddRaw(cmd);
}


extern "C"
int32_t dv_get_cmdraw_max_version() {
  return CDVCmdList::get_cmdraw_max_version();
}
