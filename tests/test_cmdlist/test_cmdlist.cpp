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
#include <tuple>

#include "dmp_dv.h"
#include "dmp_dv_cmdraw_v0.h"


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)


/// @brief Number of file descriptors for the process.
static int g_n_fd = -1;


typedef struct conv_config_impl {
  int width, height, n_channels, kx, ky, n_kernels, pad, stride, activation;

  bool operator <(const struct conv_config_impl& pt) const {
    return std::make_tuple(width, height, n_channels, kx, ky, n_kernels, pad, stride, activation) <
        std::make_tuple(pt.width, pt.height, pt.n_channels, pt.kx, pt.ky, pt.n_kernels, pt.pad, pt.stride, pt.activation);
  }
} conv_config;


/// @brief Returns width of the output based on kernel size, padding and stride.
int get_conv_out_width(int width, int kx, int pad, int stride) {
  return (pad + width + pad - kx) / stride + 1;
}


inline int divup(int a, int b) {
  int n = a / b;
  if (a % b) {
    ++n;
  }
  return n;
}


/// @brief Computes number of tiles for fpga job.
/// @returns > 0 on success, 0 when this configuration cannot be run on FPGA due to the lack of internal cache.
int get_conv_fpga_tiles(int width, int height, int n_channels, int kx, int ky, int n_kernels, int pad, int stride) {
  int t = 0;
  const int c_blocks = (n_channels >> 3) + ((n_channels & 7) ? 1 : 0);
  for (; t < width;) {
    ++t;
    const int tw = divup(width, t) + kx - 1;  // width of a tile
    const int ow = get_conv_out_width(tw, kx, pad, stride);
    const int oh = get_conv_out_width(height, ky, pad, stride);
    const int os = ow * oh * std::min(8, n_kernels);  // output buffer size
    const int ts_1c = tw * height;  // tile size for single channel
    const int ts_blk16 = ts_1c * std::min(8, n_channels);
    int ts_blk128 = (ts_blk16 >> 3) + ((ts_blk16 & 0x7) ? 1 : 0);
    // Ensure size modulo 16 = 2, this to ensure 8 blocks can be read in parallel from 16 cuts in 1x1 mode
    ts_blk128 += (2 - ts_blk128) & 0x0F;
    int ts_128 = ts_blk128 * c_blocks;
    // Ensure size modulo 16 = 0, this to ensure 8 blocks can be read in parallel from 16 cuts in 1x1 mode
    ts_128 += ((0 - ts_128) & 0x0F);
    const int ts = ts_128 << 3;  // input tile size in UBUF (in float16)
    const int uu = ts + os;  // unified buffer utilization
    if (uu * 2 <= 640 * 1024) {
      return t;
    }
  }
  return 0;
}



/// @brief Tests convolutional configurations for correctness using data from folder "data".
int test_cmdlist(const conv_config& config) {
  /*uint8_t *data = (uint8_t*)malloc(1024 * 1024);
  if (!data) {
    ERR("Failed to allocate %d bytes of memory\n", 1024 * 1024);
    return -1;
  }
  FILE *fwin = fopen("/lib/firmware/dmp/dv_program.bin", "rb");
  if (!fwin) {
    ERR("fopen() failed for /lib/firmware/dmp/dv_program.bin\n");
    free(data);
    return -1;
  }
  int nnn = fread(data, 1, 1024 * 1024, fwin);
  fclose(fwin);
  if ((nnn <= 0) || (nnn >= 1024 * 1024)) {
    ERR("fread() failed for /lib/firmware/dmp/dv_program.bin\n");
    free(data);
    return -1;
  }
  FILE *fwout = fopen("/sys/devices/platform/dmp_dv/firmware", "wb");
  if (!fwout) {
    ERR("fopen() failed for /sys/devices/platform/dmp_dv/firmware\n");
    free(data);
    return -1;
  }
  int nnnn = fwrite(data, 1, nnn, fwout);
  fclose(fwout);
  free(data);
  if (nnnn != nnn) {
    ERR("fwrite() failed for /sys/devices/platform/dmp_dv/firmware\n");
    return -1;
  }*/

  char prefix[256];
  snprintf(prefix, sizeof(prefix), "data/%dx%dx%d_%dx%dx%d_%d_%d_%d",
           config.width, config.height, config.n_channels, config.kx, config.ky, config.n_kernels,
           config.pad, config.stride, config.activation);

  const int tiles = get_conv_fpga_tiles(
      config.width, config.height, config.n_channels, config.kx | 1, config.ky | 1, config.n_kernels,
      config.pad + 1, 1);
  if (tiles != 1) {
    ERR("Unsupported tiles %d\n", tiles);
    _exit(-1);  // TODO: remove it when tiles will be fixed inside kernel module.
  }

  LOG("ENTER: test_cmdlist: %s\n", prefix);

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
  float max_diff = 0, max_diff_y = 0, max_diff_t = 0;
  float failed_diff = 0, failed_diff_y = 0, failed_diff_t = 0;
  int failed_x = -1, failed_y = -1, failed_c = -1;
  float caffe_a = std::numeric_limits<float>::max(), caffe_b = std::numeric_limits<float>::lowest();
  float dv_a = std::numeric_limits<float>::max(), dv_b = std::numeric_limits<float>::lowest();

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
  cmd.run[0].p = (uint16_t)config.kx | (((uint16_t)config.ky) << 8);
  cmd.run[0].pz = 1;
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
        config.n_channels, config.kx, config.ky, config.n_kernels,
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
  cmd.run[0].weight_fmt = 3;

  weights = dmp_dv_mem_map(weights_mem);
  if (!weights) {
    ERR("dmp_dv_mem_map() failed for weights: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  if (dmp_dv_mem_sync_start(weights_mem, 1, 1)) {
    ERR("dmp_dv_mem_sync_start() failed for weights: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  // Fill weights
  if (dmp_dv_pack_conv_weights(
      config.n_channels, config.kx, config.ky, config.n_kernels,
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
          if (diff > max_diff) {
            max_diff = diff;
            max_diff_y = y;
            max_diff_t = t;
          }
          const float ta = std::abs(t);
          if (((ta < 0.1f) && (diff > 0.03f)) ||
              ((ta >= 0.1f) && (diff > ta * 0.2f))) {
            if (diff > failed_diff) {
              failed_diff = diff;
              failed_diff_y = y;
              failed_diff_t = t;
              failed_x = i;
              failed_y = j;
              failed_c = k;
            }
          }
        }
      }
    }
  }
  if (dmp_dv_mem_sync_end(io_mem)) {
    ERR("dmp_dv_mem_sync_end() failed for input/output: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("caffe: [%.6f, %.6f] dv: [%.6f, %.6f]\n", caffe_a, caffe_b, dv_a, dv_b);
  LOG("max_diff=%.6f on y=%.6f and t=%.6f\n", max_diff, max_diff_y, max_diff_t);
  if (failed_diff > 0.0f) {
    ERR("FAILED: failed_diff=%.6f on y=%.6f and t=%.6f xy=(%d, %d) chan=%d\n", failed_diff, failed_diff_y, failed_diff_t,
        failed_x, failed_y, failed_c);
    goto L_EXIT;
  }

  result = 0;
  LOG("SUCCESS: test_cmdlist\n");

  L_EXIT:
  dmp_dv_cmdlist_destroy(cmdlist);
  dmp_dv_mem_free(weights_mem);
  dmp_dv_mem_free(io_mem);
  dmp_dv_context_destroy(ctx);

  int n_fd = 0;
  DIR *d;
  struct dirent *dir;
  d = opendir("/proc/self/fd");
  if (!d) {
    ERR("Could not open \"/proc/self/fd\" folder\n");
    return -1;
  }
  while ((dir = readdir(d))) {
    char *fnme = dir->d_name;
    bool num = true;
    for (; *fnme; ++fnme) {
      if ((*fnme >= '0') && (*fnme <= '9')) {
        continue;
      }
      num = false;
      break;
    }
    if (num) {
      ++n_fd;
    }
  }
  closedir(d);

  if (g_n_fd == -1) {
    g_n_fd = n_fd;
  }
  if (n_fd != g_n_fd) {
    ERR("Inconsistent file descriptor count detected, memory leak is probable");
    result = -1;
  }

  LOG("EXIT: test_cmdlist: %s: FD count: %d\n", prefix, n_fd);
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
    for (int k = 0; k < 2; ++k) {
      res = test_cmdlist(*it);
      if (res) {
        ++n_err;
      }
      else {
        ++n_ok;
      }
    }
  }

  LOG("Tests succeeded: %d\n", n_ok);
  LOG("Tests failed: %d\n", n_err);
  return n_err;
}
