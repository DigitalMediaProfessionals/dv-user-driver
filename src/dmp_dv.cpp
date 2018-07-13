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


/// @brief Last error message (instantiation).
char s_last_error_message[256];


extern "C"
const char *dmp_dv_get_last_error_message() {
  return s_last_error_message;
}


extern "C"
const char *dmp_dv_get_version_string() {
  return "0.1.0 Initial release.";
}


extern "C"
dmp_dv_context* dmp_dv_context_create(const char *path) {
  CDMPDVContext *ctx = new CDMPDVContext();
  if (!ctx) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDMPDVContext));
    return NULL;
  }
  if (!ctx->Initialize(path)) {
    delete ctx;
    return NULL;
  }
  return (dmp_dv_context*)ctx;
}


extern "C"
const char *dmp_dv_context_get_info_string(dmp_dv_context* ctx) {
  if (!ctx) {
    SET_ERR("Invalid argument: ctx is NULL");
    return "";
  }
  return ((CDMPDVContext*)ctx)->GetInfoString();
}


extern "C"
void dmp_dv_context_destroy(dmp_dv_context *ctx) {
  if (!ctx) {
    return;
  }
  free(ctx);
}


extern "C"
dmp_dv_mem* dmp_dv_mem_alloc(dmp_dv_context *ctx, size_t size) {
  CDMPDVMem *mem = new CDMPDVMem();
  if (!mem) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDMPDVMem));
    return NULL;
  }
  if (!mem->Initialize((CDMPDVContext*)ctx, size)) {
    delete mem;
    return NULL;
  }

  return (dmp_dv_mem*)mem;
}


extern "C"
void dmp_dv_mem_free(dmp_dv_mem *mem) {
  if (!mem) {
    return;
  }
  delete (CDMPDVMem*)mem;
}


extern "C"
uint8_t *dmp_dv_mem_map(dmp_dv_mem *mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return NULL;
  }
  return ((CDMPDVMem*)mem)->Map();
}


extern "C"
void dmp_dv_mem_unmap(dmp_dv_mem *mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return;
  }
  ((CDMPDVMem*)mem)->Unmap();
}


extern "C"
int dmp_dv_mem_sync_start(dmp_dv_mem *mem, int rd, int wr) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return -1;
  }
  return ((CDMPDVMem*)mem)->SyncStart(rd, wr);
}


extern "C"
int dmp_dv_mem_sync_end(dmp_dv_mem *mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return -1;
  }
  return ((CDMPDVMem*)mem)->SyncEnd();
}


extern "C"
size_t dmp_dv_mem_get_size(dmp_dv_mem *mem) {
  if (!mem) {
    SET_ERR("Invalid argument: mem is NULL");
    return -1;
  }
  return ((CDMPDVMem*)mem)->get_size();
}


extern "C"
int dmp_dv_sync(dmp_dv_context *ctx) {
  if (!ctx) {
    SET_ERR("Invalid argument: ctx is NULL");
    return -1;
  }
  return ((CDMPDVContext*)ctx)->Sync();
}


extern "C"
dmp_dv_cmdlist *dmp_dv_cmdlist_create(dmp_dv_context *ctx) {
  CDMPDVCmdList *cmdlist = new CDMPDVCmdList();
  if (!cmdlist) {
    SET_ERR("Failed to allocate %zu bytes of memory", sizeof(CDMPDVCmdList));
    return NULL;
  }
  if (!cmdlist->Initialize((CDMPDVContext*)ctx)) {
    delete cmdlist;
    return NULL;
  }
  return (dmp_dv_cmdlist*)cmdlist;
}


extern "C"
void dmp_dv_cmdlist_destroy(dmp_dv_cmdlist *cmdlist) {
  if (!cmdlist) {
    return;
  }
  delete (CDMPDVCmdList*)cmdlist;
}


extern "C"
int dmp_dv_cmdlist_end(dmp_dv_cmdlist *cmdlist) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return -1;
  }
  return ((CDMPDVCmdList*)cmdlist)->End();
}


extern "C"
int dmp_dv_cmdlist_exec(dmp_dv_cmdlist *cmdlist) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return -1;
  }
  return ((CDMPDVCmdList*)cmdlist)->Exec();
}


extern "C"
int dmp_dv_cmdlist_add_raw(dmp_dv_cmdlist *cmdlist, dmp_dv_cmdraw *cmd) {
  if (!cmdlist) {
    SET_ERR("Invalid argument: cmdlist is NULL");
    return -1;
  }
  return ((CDMPDVCmdList*)cmdlist)->AddRaw(cmd);
}


extern "C"
int32_t dmp_dv_get_cmdraw_max_version() {
  return CDMPDVCmdList::get_cmdraw_max_version();
}