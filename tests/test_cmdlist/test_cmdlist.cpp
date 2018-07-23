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
#include <vector>
#include <cmath>
#include <limits>

#include "dmp_dv.h"
#include "dmp_dv_cmdraw_v0.h"


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


/// @brief Returns width of the output based on kernel size, padding and stride.
int get_conv_out_width(int width, int kx, int pad, int stride) {
  return (pad + width + pad - kx) / stride + 1;
}


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

  // Load quantization map
  char fnme[512];
  snprintf(fnme, sizeof(fnme), "%s.q.bin", prefix);
  uint16_t quant_map[256];
  FILE *fin = fopen(fnme, "rb");
  if (!fin) {
    ERR("fopen() failed for %s\n", fnme);
    return -1;
  }
  int n = fread(quant_map, sizeof(quant_map[0]), sizeof(quant_map) / sizeof(quant_map[0]), fin);
  char c;
  bool fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
  fclose(fin);
  if (n != 256) {
    ERR("fread() returned %d while expecting %d for %s\n", n, 256, fnme);
    return -1;
  }
  if (!fend) {
    ERR("File is bigger than expected: %s\n", fnme);
    return -1;
  }

  // Load input
  snprintf(fnme, sizeof(fnme), "%s.i.bin", prefix);
  fin = fopen(fnme, "rb");
  if (!fin) {
    ERR("fopen() failed for %s\n", fnme);
    return -1;
  }
  std::vector<__fp16> caffe_input;
  caffe_input.resize(config.n_channels * config.height * config.width);
  n = fread(caffe_input.data(), sizeof(caffe_input[0]), caffe_input.size(), fin);
  fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
  fclose(fin);
  if (n != config.n_channels * config.height * config.width) {
    ERR("fread() returned %d while expecting %d for %s\n", n, config.n_channels * config.height * config.width, fnme);
    return -1;
  }
  if (!fend) {
    ERR("File is bigger than expected: %s\n", fnme);
    return -1;
  }

  // Load weights
  snprintf(fnme, sizeof(fnme), "%s.w.bin", prefix);
  fin = fopen(fnme, "rb");
  if (!fin) {
    ERR("fopen() failed for %s\n", fnme);
    return -1;
  }
  std::vector<uint8_t> caffe_weights;
  caffe_weights.resize(config.n_kernels * config.n_channels * config.kx * config.ky);
  n = fread(caffe_weights.data(), sizeof(caffe_weights[0]), caffe_weights.size(), fin);
  fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
  fclose(fin);
  if (n != config.n_kernels * config.n_channels * config.kx * config.ky) {
    ERR("fread() returned %d while expecting %d for %s\n", n, config.n_kernels * config.n_channels * config.kx * config.ky, fnme);
    return -1;
  }
  if (!fend) {
    ERR("File is bigger than expected: %s\n", fnme);
    return -1;
  }

  // Load bias
  snprintf(fnme, sizeof(fnme), "%s.b.bin", prefix);
  fin = fopen(fnme, "rb");
  if (!fin) {
    ERR("fopen() failed for %s\n", fnme);
    return -1;
  }
  std::vector<__fp16> caffe_bias;
  caffe_bias.resize(config.n_kernels);
  n = fread(caffe_bias.data(), sizeof(caffe_bias[0]), caffe_bias.size(), fin);
  fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
  fclose(fin);
  if (n != config.n_kernels) {
    ERR("fread() returned %d while expecting %d for %s\n", n, config.n_kernels, fnme);
    return -1;
  }
  if (!fend) {
    ERR("File is bigger than expected: %s\n", fnme);
    return -1;
  }

  // Load output
  snprintf(fnme, sizeof(fnme), "%s.o.bin", prefix);
  fin = fopen(fnme, "rb");
  if (!fin) {
    ERR("fopen() failed for %s\n", fnme);
    return -1;
  }
  const int out_width = get_conv_out_width(config.width, config.kx, config.pad, config.stride);
  const int out_height = get_conv_out_width(config.height, config.ky, config.pad, config.stride);
  std::vector<__fp16> caffe_output;
  caffe_output.resize(out_width * out_height * config.n_kernels);
  n = fread(caffe_output.data(), sizeof(caffe_output[0]), caffe_output.size(), fin);
  fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
  fclose(fin);
  if (n != out_width * out_height * config.n_kernels) {
    ERR("fread() returned %d while expecting %d for %s\n", n, out_width * out_height * config.n_kernels, fnme);
    return -1;
  }
  if (!fend) {
    ERR("File is bigger than expected: %s\n", fnme);
    return -1;
  }

  // Initialize DV context
  int result = -1;
  dmp_dv_context *ctx = NULL;
  dmp_dv_cmdlist *cmdlist = NULL;
  dmp_dv_mem *io_mem = NULL, *weights_mem = NULL;
  size_t io_size, weights_size;
  int32_t cmdraw_max_version;
  uint8_t *weights;
  __fp16 *io;
  float max_diff, max_diff_pt;
  float caffe_a = std::numeric_limits<float>::max(), caffe_b = std::numeric_limits<float>::min();
  float dv_a = std::numeric_limits<float>::max(), dv_b = std::numeric_limits<float>::min();

  LOG("dmp_dv_get_version_string(): %s\n", dmp_dv_get_version_string());

  ctx  = dmp_dv_context_create(NULL);
  if (!ctx) {
    ERR("dmp_dv_context_create() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully created context: %s\n", dmp_dv_context_get_info_string(ctx));

  cmdlist = dmp_dv_cmdlist_create(ctx);
  if (!cmdlist) {
    ERR("dmp_dv_cmdlist_create() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Created command list\n");

  cmdraw_max_version = dmp_dv_get_cmdraw_max_version();
  if (cmdraw_max_version < 0) {
    ERR("dmp_dv_get_cmdraw_max_version() returned %d\n", (int)cmdraw_max_version);
    goto L_EXIT;
  }
  LOG("Maximum supported version for raw command is %d\n", (int)cmdraw_max_version);

  dmp_dv_cmdraw_v0 cmd;
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

  io_size = (config.width * config.height * config.n_channels + out_width * out_height * config.n_kernels) * sizeof(__fp16);
  io_mem = dmp_dv_mem_alloc(ctx, io_size);
  if (!io_mem) {
    ERR("dmp_dv_mem_alloc() failed for %zu bytes: %s\n", io_size, dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Allocated %zu (%zu requested) bytes for input/output\n", dmp_dv_mem_get_size(io_mem), io_size);
  cmd.input_buf.mem = io_mem;
  cmd.input_buf.offs = 0;
  cmd.output_buf.mem = io_mem;
  cmd.output_buf.offs = config.width * config.height * config.n_channels * sizeof(__fp16);

  weights_size = 0;
  if (dmp_dv_pack_conv_weights(
        cmd.c, cmd.run[0].p, cmd.run[0].p, cmd.run[0].m,
        quant_map, NULL, NULL, NULL, &weights_size)) {
    ERR("dmp_dv_pack_conv_weights() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  weights_mem = dmp_dv_mem_alloc(ctx, weights_size);
  if (!weights_mem) {
    ERR("dmp_dv_mem_alloc() failed for %zu bytes: %s\n", weights_size, dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Allocated %zu (%zu requested) bytes for weights\n", dmp_dv_mem_get_size(weights_mem), weights_size);

  cmd.run[0].weight_buf.mem = weights_mem;
  cmd.run[0].weight_buf.offs = 0;
  cmd.run[0].weight_fmt = 2;

  weights = dmp_dv_mem_map(weights_mem);
  if (!weights) {
    ERR("dmp_dv_mem_map() failed for weights: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  if (dmp_dv_mem_sync_start(weights_mem, 0, 1)) {
    ERR("dmp_dv_mem_sync_start() failed for weights: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  // Fill weights
  if (dmp_dv_pack_conv_weights(
        cmd.c, cmd.run[0].p, cmd.run[0].p, cmd.run[0].m,
        quant_map, caffe_weights.data(), (const uint16_t*)caffe_bias.data(), weights, &weights_size)) {
    ERR("dmp_dv_pack_conv_weights() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  if (dmp_dv_mem_sync_end(weights_mem)) {
    ERR("dmp_dv_mem_sync_end() failed for weights: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  dmp_dv_mem_unmap(weights_mem);

  io = (__fp16*)dmp_dv_mem_map(io_mem);
  if (!io) {
    ERR("dmp_dv_mem_map() failed for input/output: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  if (dmp_dv_mem_sync_start(io_mem, 0, 1)) {
    ERR("dmp_dv_mem_sync_start() failed for input/output: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  // Caffe's input is stored as channel, height, width
  // DV input should be stored as chunks by max of 8 channels as width, height, channel
  for (int chan_group = 0, o_offs = 0; chan_group < config.n_channels; chan_group += 8) {
    const int last_chan = std::min(chan_group + 8, config.n_channels);
    for (int i = 0; i < config.width; ++i) {
      for (int j = 0; j < config.height; ++j) {
        for (int k = chan_group; k < last_chan; ++k, ++o_offs) {
          const int i_offs = k * config.width * config.height + j * config.width + i;
          const __fp16 vle = caffe_input[i_offs];
          io[o_offs] = vle;
        }
      }
    }
  }
  memset(io + config.width * config.height * config.n_channels, 0, out_width * out_height * config.n_kernels * sizeof(__fp16));
  if (dmp_dv_mem_sync_end(io_mem)) {
    ERR("dmp_dv_mem_sync_end() failed for input/output: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  if (dmp_dv_cmdlist_add_raw(cmdlist, (dmp_dv_cmdraw*)&cmd)) {
    ERR("dmp_dv_cmdlist_add_raw() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  if (dmp_dv_cmdlist_end(cmdlist)) {
    ERR("dmp_dv_cmdlist_end() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Ended the command list\n");

  if (dmp_dv_cmdlist_exec(cmdlist)) {
    ERR("dmp_dv_cmdlist_exec() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Scheduled command list for execution\n");

  LOG("Waiting for completion\n");
  if (dmp_dv_sync(ctx)) {
    ERR("dmp_dv_sync() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Execution has completed\n");

  if (dmp_dv_mem_sync_start(io_mem, 1, 0)) {
    ERR("dmp_dv_mem_sync_start() failed for input/output: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  // Caffe's output is stored as channels, height, width
  // DV output is stored as chunks by max of 8 channels as width, height, channel
  max_diff = 0;
  max_diff_pt = 0;
  for (int chan_group = 0, o_offs = config.width * config.height * config.n_channels;
       chan_group < config.n_kernels; chan_group += 8) {
    const int last_chan = std::min(chan_group + 8, config.n_kernels);
    for (int i = 0; i < out_width; ++i) {
      for (int j = 0; j < out_height; ++j) {
        for (int k = chan_group; k < last_chan; ++k, ++o_offs) {
          const int i_offs = k * out_width * out_height + j * out_width + i;
          const __fp16 vle = caffe_output[i_offs];
          const float y = (float)io[o_offs], t = (float)vle;
          caffe_a = std::min(caffe_a, t);
          caffe_b = std::max(caffe_b, t);
          dv_a = std::min(dv_a, y);
          dv_b = std::max(dv_b, y);
          const float diff = std::abs(y - t);
          const float mx = std::max(std::abs(y), std::abs(t));
          float diff_pt = std::abs((float)io[o_offs] - (float)vle);
          diff_pt *= 100.0f / std::max(mx, 1.0e-8f);
          max_diff = std::max(max_diff, diff);
          max_diff_pt = std::max(max_diff_pt, diff_pt);
        }
      }
    }
  }
  if (dmp_dv_mem_sync_end(io_mem)) {
    ERR("dmp_dv_mem_sync_end() failed for input/output: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("caffe: [%.6f, %.6f] dv: [%.6f, %.6f]\n", caffe_a, caffe_b, dv_a, dv_b);
  LOG("max_diff=%.6f max_diff_pt=%.1f%%\n", max_diff, max_diff_pt);
  if (max_diff_pt > 5.0f) {
    ERR("Difference is too large: max_diff_pt=%.1f%%\n", max_diff_pt);
    goto L_EXIT;
  }

  result = 0;
  LOG("SUCCESS: test_cmdlist\n");

  L_EXIT:
  dmp_dv_cmdlist_destroy(cmdlist);
  dmp_dv_mem_free(weights_mem);
  dmp_dv_mem_free(io_mem);
  dmp_dv_context_destroy(ctx);

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
