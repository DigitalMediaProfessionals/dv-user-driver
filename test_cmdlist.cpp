/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>

#include <stdio.h>
#include <string.h>

#include <memory>

#include "dv.h"
#include "dv_cmdraw_v0.h"


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)


int test_cmdlist() {
  LOG("ENTER: test_cmdlist\n");
  LOG("dv_get_version_string(): %s\n", dv_get_version_string());

  dv_context *ctx = dv_context_create(NULL);
  if (!ctx) {
    ERR("dv_context_create() failed: %s\n", dv_get_last_error_message());
    return -1;
  }
  LOG("Successfully created context: %s\n", dv_context_get_info_string(ctx));

  const int size = 65536;
  dv_mem *mem = dv_mem_alloc(ctx, size);
  if (!mem) {
    ERR("dv_mem_alloc() failed: %s\n", dv_get_last_error_message());
    dv_context_destroy(ctx);
    return -1;
  }
  LOG("Successfully allocated %zu bytes of memory\n", size);

  uint8_t *arr = dv_mem_map(mem);
  if (!arr) {
    ERR("dv_mem_map() failed: %s\n", dv_get_last_error_message());
    dv_mem_free(mem);
    dv_context_destroy(ctx);
    return -1;
  }
  LOG("Successfully mapped to user space %zu bytes of memory, address is %zu\n", size, (size_t)arr);

  // Start Write memory transaction
  if (dv_mem_sync_start(mem, 0, 1)) {
    ERR("dv_mem_sync_start() failed for writing: %s\n", dv_get_last_error_message());
    dv_mem_free(mem);
    dv_context_destroy(ctx);
    return -1;
  }
  memset(arr, 0, size);
  if (dv_mem_sync_end(mem)) {
    ERR("dv_mem_sync_end() failed: %s\n", dv_get_last_error_message());
    dv_mem_free(mem);
    dv_context_destroy(ctx);
    return -1;
  }
  LOG("Successfully filled the buffer and synced with physical memory\n");

  dv_cmdlist *cmdlist = dv_cmdlist_create(ctx);
  if (!cmdlist) {
    ERR("dv_cmdlist_create() failed: %s\n", dv_get_last_error_message());
    dv_mem_free(mem);
    dv_context_destroy(ctx);
    return -1;
  }
  LOG("Created command list\n");

  const int32_t cmdraw_max_version = dv_get_cmdraw_max_version();
  if (cmdraw_max_version < 0) {
    ERR("dv_get_cmdraw_max_version() returned %d\n", (int)cmdraw_max_version);
    dv_cmdlist_destroy(cmdlist);
    dv_mem_free(mem);
    dv_context_destroy(ctx);
    return -1;
  }
  LOG("Maximum supported version for raw command is %d\n", (int)cmdraw_max_version);

  dv_cmdraw_v0 cmd;
  memset(&cmd, 0, sizeof(cmd));
  cmd.size = sizeof(cmd);
  cmd.version = 0;

  if (dv_cmdlist_add_raw(cmdlist, (dv_cmdraw*)&cmd)) {
    ERR("dv_cmdlist_add_raw() failed: %s\n", dv_get_last_error_message());
    dv_cmdlist_destroy(cmdlist);
    dv_mem_free(mem);
    dv_context_destroy(ctx);
    return -1;
  }

  // TODO: implement.

  if (dv_cmdlist_end(cmdlist)) {
    ERR("dv_cmdlist_end() failed: %s\n", dv_get_last_error_message());
    dv_cmdlist_destroy(cmdlist);
    dv_mem_free(mem);
    dv_context_destroy(ctx);
    return -1;
  }
  LOG("Ended the command list");

  if (dv_cmdlist_exec(cmdlist)) {
    ERR("dv_cmdlist_exec() failed: %s\n", dv_get_last_error_message());
    dv_cmdlist_destroy(cmdlist);
    dv_mem_free(mem);
    dv_context_destroy(ctx);
    return -1;
  }
  LOG("Executed the command list");

  dv_cmdlist_destroy(cmdlist);
  dv_mem_free(mem);
  dv_context_destroy(ctx);

  LOG("EXIT: test_cmdlist\n");
  return 0;
}


int main(int argc, char **argv) {
  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  for (int i = 0; i < 10; ++i) {
    res = test_cmdlist();
    if (res) {
      ++n_err;
    }
    else {
      ++n_ok;
    }
  }

  LOG("Tests succeeded: %d\n", n_ok);
  LOG("Tests failed: %d\n", n_err);
  return n_err;
}
