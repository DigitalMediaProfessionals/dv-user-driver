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

  int result = -1;
  dv_context *ctx = NULL;
  dv_cmdlist *cmdlist = NULL;
  dv_mem *io_mem = NULL, *weights_mem = NULL;
  size_t io_size, weights_size;
  int32_t cmdraw_max_version;

  LOG("dv_get_version_string(): %s\n", dv_get_version_string());

  ctx  = dv_context_create(NULL);
  if (!ctx) {
    ERR("dv_context_create() failed: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully created context: %s\n", dv_context_get_info_string(ctx));

  cmdlist = dv_cmdlist_create(ctx);
  if (!cmdlist) {
    ERR("dv_cmdlist_create() failed: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Created command list\n");

  cmdraw_max_version = dv_get_cmdraw_max_version();
  if (cmdraw_max_version < 0) {
    ERR("dv_get_cmdraw_max_version() returned %d\n", (int)cmdraw_max_version);
    goto L_EXIT;
  }
  LOG("Maximum supported version for raw command is %d\n", (int)cmdraw_max_version);

  dv_cmdraw_v0 cmd;
  memset(&cmd, 0, sizeof(cmd));
  cmd.size = sizeof(cmd);
  cmd.version = 0;
  cmd.w = 64;
  cmd.h = 32;
  cmd.c = 3;
  cmd.z = 1;
  cmd.tiles = 1;
  cmd.topo = 1;
  cmd.run[0].m = 32;
  cmd.run[0].conv_enable = 1;
  cmd.run[0].p = 3;
  cmd.run[0].conv_pad = 0x01010101;
  cmd.run[0].conv_stride = 0x0101;
  cmd.run[0].actfunc = 5;

  io_size = ((size_t)cmd.w * cmd.h * cmd.c + (size_t)cmd.w * cmd.h * cmd.run[0].m) * 2;
  io_mem = dv_mem_alloc(ctx, io_size);
  if (!io_mem) {
    ERR("dv_mem_alloc() failed for %zu bytes: %s\n", io_size, dv_get_last_error_message());
    goto L_EXIT;
  }
  cmd.input_buf.mem = io_mem;
  cmd.input_buf.offs = 0;
  cmd.output_buf.mem = io_mem;
  cmd.output_buf.offs = (size_t)cmd.w * cmd.h * cmd.c * 2;

  weights_size = 65536;  // TODO: put real size here.
  weights_mem = dv_mem_alloc(ctx, weights_size);
  if (!io_mem) {
    ERR("dv_mem_alloc() failed for %zu bytes: %s\n", weights_size, dv_get_last_error_message());
    goto L_EXIT;
  }

  cmd.run[0].weight_buf.mem = weights_mem;
  cmd.run[0].weight_buf.offs = 0;
  cmd.run[0].weight_fmt = 2;

  if (dv_cmdlist_add_raw(cmdlist, (dv_cmdraw*)&cmd)) {
    ERR("dv_cmdlist_add_raw() failed: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }

  if (dv_cmdlist_end(cmdlist)) {
    ERR("dv_cmdlist_end() failed: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Ended the command list");

  if (dv_cmdlist_exec(cmdlist)) {
    ERR("dv_cmdlist_exec() failed: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Scheduled command list for execution");

  if (dv_sync(ctx)) {
    ERR("dv_sync() failed: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Execution has completed");

  LOG("TODO: check the correctness of the result");

  result = 0;
  LOG("SUCCESS: test_cmdlist\n");

  L_EXIT:
  dv_cmdlist_destroy(cmdlist);
  dv_mem_free(weights_mem);
  dv_mem_free(io_mem);
  dv_context_destroy(ctx);

  LOG("EXIT: test_cmdlist\n");
  return result;
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
