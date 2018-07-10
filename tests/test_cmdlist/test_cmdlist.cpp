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
#include <dirent.h>

#include <stdio.h>
#include <string.h>

#include <memory>
#include <set>

#include "dv.h"
#include "dv_cmdraw_v0.h"


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)


typedef struct conv_config_impl {
  int width, height, n_channels, kx, ky, n_kernels, pad, stride, activation;

  bool operator <(const struct conv_config_impl& pt) const {
    if (width < pt.width) {
      return true;
    }
    if (width > pt.height) {
      return false;
    }
    if (height < pt.height) {
      return true;
    }
    if (height > pt.height) {
      return false;
    }
    if (n_channels < pt.n_channels) {
      return true;
    }
    if (n_channels > pt.n_channels) {
      return false;
    }
    if (kx < pt.kx) {
      return true;
    }
    if (kx > pt.kx) {
      return false;
    }
    if (ky < pt.ky) {
      return true;
    }
    if (ky > pt.ky) {
      return false;
    }
    if (n_kernels < pt.n_kernels) {
      return true;
    }
    if (n_kernels > pt.n_kernels) {
      return false;
    }
    if (pad < pt.pad) {
      return true;
    }
    if (pad > pt.pad) {
      return false;
    }
    if (stride < pt.stride) {
      return true;
    }
    if (stride > pt.stride) {
      return false;
    }
    if (activation < pt.activation) {
      return true;
    }
    return false;
  }
} conv_config;


int test_cmdlist(const conv_config& config) {
  char prefix[256];
  snprintf(prefix, sizeof(prefix), "data/%dx%dx%d_%dx%dx%d_%d_%d_%d",
           config.width, config.height, config.n_channels, config.kx, config.ky, config.n_kernels,
           config.pad, config.stride, config.activation);

  LOG("ENTER: test_cmdlist: %s\n", prefix);

  if (config.kx != config.ky) {
    ERR("Only square kernels are supported, got %d x %d\n", config.kx, config.ky);
    return -1;
  }

  char fnme[512];
  snprintf(fnme, sizeof(fnme), "%s.q.bin", prefix);
  uint16_t quant_map[256];
  FILE *fin = fopen(fnme, "rb");
  if (!fin) {
    ERR("fopen() failed for %s\n", fnme);
    return -1;
  }
  int n = fread(quant_map, 1, 256, fin);
  fclose(fin);
  if (n != 256) {
    ERR("fread() returned %d while expecting %d for %s\n", n, 256, fnme);
    return -1;
  }

  int result = -1;
  dv_context *ctx = NULL;
  dv_cmdlist *cmdlist = NULL;
  dv_mem *io_mem = NULL, *weights_mem = NULL;
  size_t io_size, weights_size;
  int32_t cmdraw_max_version;
  uint8_t *weights;
  __fp16 *io;

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
  cmd.w = config.width;
  cmd.h = config.height;
  cmd.c = config.n_channels;
  cmd.z = 1;
  cmd.topo = 1;
  cmd.run[0].m = config.n_kernels;
  cmd.run[0].conv_enable = 1;
  cmd.run[0].p = config.kx;
  {
    const uint32_t pad8 = config.pad;
    cmd.run[0].conv_pad = pad8 | (pad8 << 8) | (pad8 << 16) | (pad8 << 24);
  }
  {
    const uint16_t stride8 = config.stride;
    cmd.run[0].conv_stride = stride8 | (stride8 << 8);
  }
  cmd.run[0].actfunc = config.activation;

  io_size = ((size_t)cmd.w * cmd.h * cmd.c + (size_t)cmd.w * cmd.h * cmd.run[0].m) * 2;
  io_mem = dv_mem_alloc(ctx, io_size);
  if (!io_mem) {
    ERR("dv_mem_alloc() failed for %zu bytes: %s\n", io_size, dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Allocated %zu (%zu requested) bytes for input/output\n", dv_mem_get_size(io_mem), io_size);
  cmd.input_buf.mem = io_mem;
  cmd.input_buf.offs = 0;
  cmd.output_buf.mem = io_mem;
  cmd.output_buf.offs = (size_t)cmd.w * cmd.h * cmd.c * 2;

  weights_size = 0;
  if (pack_conv_weights(
        cmd.c, cmd.run[0].p, cmd.run[0].p, cmd.run[0].m,
        quant_map, NULL, NULL, NULL, &weights_size)) {
    ERR("pack_conv_weights() failed: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }

  weights_mem = dv_mem_alloc(ctx, weights_size);
  if (!weights_mem) {
    ERR("dv_mem_alloc() failed for %zu bytes: %s\n", weights_size, dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Allocated %zu (%zu requested) bytes for weights\n", dv_mem_get_size(weights_mem), weights_size);

  cmd.run[0].weight_buf.mem = weights_mem;
  cmd.run[0].weight_buf.offs = 0;
  cmd.run[0].weight_fmt = 2;

  weights = dv_mem_map(weights_mem);
  if (!weights) {
    ERR("dv_mem_map() failed for weights: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  if (dv_mem_sync_start(weights_mem, 0, 1)) {
    ERR("dv_mem_sync_start() failed for weights: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  // TODO: fill weights.
  if (dv_mem_sync_end(weights_mem)) {
    ERR("dv_mem_sync_end() failed for weights: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  dv_mem_unmap(weights_mem);

  io = (__fp16*)dv_mem_map(io_mem);
  if (!io) {
    ERR("dv_mem_map() failed for input/output: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  if (dv_mem_sync_start(io_mem, 0, 1)) {
    ERR("dv_mem_sync_start() failed for input/output: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  // TODO: fill input.
  if (dv_mem_sync_end(io_mem)) {
    ERR("dv_mem_sync_end() failed for input/output: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }

  if (dv_cmdlist_add_raw(cmdlist, (dv_cmdraw*)&cmd)) {
    ERR("dv_cmdlist_add_raw() failed: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }

  if (dv_cmdlist_end(cmdlist)) {
    ERR("dv_cmdlist_end() failed: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Ended the command list\n");

  if (dv_cmdlist_exec(cmdlist)) {
    ERR("dv_cmdlist_exec() failed: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Scheduled command list for execution\n");

  if (dv_sync(ctx)) {
    ERR("dv_sync() failed: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Execution has completed\n");

  if (dv_mem_sync_start(io_mem, 1, 0)) {
    ERR("dv_mem_sync_start() failed for input/output: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }
  // TODO: check the correctness of the result.
  if (dv_mem_sync_end(io_mem)) {
    ERR("dv_mem_sync_end() failed for input/output: %s\n", dv_get_last_error_message());
    goto L_EXIT;
  }

  result = 0;
  LOG("SUCCESS: test_cmdlist\n");

  L_EXIT:
  dv_cmdlist_destroy(cmdlist);
  dv_mem_free(weights_mem);
  dv_mem_free(io_mem);
  dv_context_destroy(ctx);

  LOG("EXIT: test_cmdlist: %s\n", prefix);
  return result;
}


int main(int argc, char **argv) {
  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  std::set<conv_config> configs;

  DIR *d;
  struct dirent *dir;
  d = opendir("data");
  if (!d) {
    ERR("Could not open \"data\" folder\n");
    return -1;
  }
  while ((dir = readdir(d))) {
    // 64x32x3_3x3x32_1_1_5.q.bin
    conv_config config;

    char fnme[512];
    for (int i = 0; ; ) {
      char c = dir->d_name[i];
      fnme[i] = ((c >= '0') && (c <= '9')) ? c : ' ';
      ++i;
      if (!dir->d_name[i]) {
        fnme[i] = 0;
        break;
      }
    }

    if (sscanf(fnme, "%d%d%d%d%d%d%d%d%d",
               &config.width, &config.height, &config.n_channels, &config.kx, &config.ky, &config.n_kernels,
               &config.pad, &config.stride, &config.activation) != 9) {
      continue;
    }

    configs.emplace(std::move(config));
  }
  closedir(d);

  for (auto it = configs.cbegin(); it != configs.cend(); ++it) {
    res = test_cmdlist(*it);
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
