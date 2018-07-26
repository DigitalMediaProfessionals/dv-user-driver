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


/// @brief File which will store logs.
static FILE *g_flog = NULL;


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout); fprintf(g_flog, __VA_ARGS__); fflush(g_flog)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr); fprintf(g_flog, __VA_ARGS__); fflush(g_flog)


/// @brief Number of file descriptors for the process.
static int g_n_fd = -1;


/// @brief Configuration description to be tested.
typedef struct conv_config_impl {
  int width, height, n_channels, kx, ky, n_kernels,
      pad_left, pad_top, pad_right, pad_bottom, stride_x, stride_y, activation;

  bool operator <(const struct conv_config_impl& pt) const {
    return std::make_tuple(width, height, n_channels, kx, ky, n_kernels,
                           pad_left, pad_top, pad_right, pad_bottom,
                           stride_x, stride_y, activation) <
        std::make_tuple(pt.width, pt.height, pt.n_channels, pt.kx, pt.ky, pt.n_kernels,
                        pt.pad_left, pt.pad_top, pt.pad_right, pt.pad_bottom,
                        pt.stride_x, pt.stride_y, pt.activation);
  }
} conv_config;


/// @brief Returns width of the output based on kernel size, padding and stride.
int get_conv_out_width(int width, int kx, int pad_left, int pad_right, int stride) {
  return (pad_left + width + pad_right - kx) / stride + 1;
}


/// @brief Prints command content for debugging.
void print_cmd(dmp_dv_cmdraw_v0& cmd) {
  LOG("topo = %u\nw = %u\nh = %u\nz = %u\nc = %u\ninput_circular_offset = %u\noutput_mode = %u\n",
      (uint32_t)cmd.topo, (uint32_t)cmd.w, (uint32_t)cmd.h, (uint32_t)cmd.z, (uint32_t)cmd.c,
      (uint32_t)cmd.input_circular_offset, (uint32_t)cmd.output_mode);
  LOG("conv_pad = 0x%08x\npool_pad = 0x%08x\nm = %u\nconv_enable = %u\np = 0x%04x\n"
      "pz = %u\nconv_stride = 0x%04x\nconv_dilation = %u\nweight_fmt = %u\n"
      "pool_enable = %u\npool_avg_param = %u\npool_size = 0x%04x\npool_stride = 0x%04x\n"
      "actfunc = %u\nactfunc_param = %u\nrectifi_en = %u\nlrn = %u\n",
      (uint32_t)cmd.run[0].conv_pad, (uint32_t)cmd.run[0].pool_pad, (uint32_t)cmd.run[0].m,
      (uint32_t)cmd.run[0].conv_enable, (uint32_t)cmd.run[0].p, (uint32_t)cmd.run[0].pz,
      (uint32_t)cmd.run[0].conv_stride, (uint32_t)cmd.run[0].conv_dilation,
      (uint32_t)cmd.run[0].weight_fmt, (uint32_t)cmd.run[0].pool_enable, (uint32_t)cmd.run[0].pool_avg_param,
      (uint32_t)cmd.run[0].pool_size, (uint32_t)cmd.run[0].pool_stride,
      (uint32_t)cmd.run[0].actfunc, (uint32_t)cmd.run[0].actfunc_param, (uint32_t)cmd.run[0].rectifi_en,
      (uint32_t)cmd.run[0].lrn);
}


/// @brief Tests convolutional configurations for correctness using data from folder "data".
int test_cmdlists(const std::vector<conv_config>& confs) {
  char prefix[256];
  LOG("ENTER: test_cmdlists: %d commands:", (int)confs.size());
  for (auto it = confs.cbegin(); it != confs.cend(); ++it) {
    snprintf(prefix, sizeof(prefix), "data/%dx%dx%d_%dx%dx%d_pad%dx%dx%dx%d_stride%dx%d_act%d",
             it->width, it->height, it->n_channels, it->kx, it->ky, it->n_kernels,
             it->pad_left, it->pad_top, it->pad_right, it->pad_bottom,
             it->stride_x, it->stride_y, it->activation);
    LOG(" %s", prefix);
  }
  LOG("\n");

  int result = -1;
  dmp_dv_context *ctx = NULL;
  dmp_dv_cmdlist *cmdlist = NULL;
  std::vector<dmp_dv_mem*> io_mems, weights_mems;
  dmp_dv_mem *io_mem, *weights_mem;
  size_t io_size, weights_size;
  int32_t cmdraw_max_version;
  uint8_t *weights;
  std::vector<__fp16*> io_ptrs;
  __fp16 *io_ptr;
  float max_diff = 0, max_diff_y = 0, max_diff_t = 0;
  float failed_diff = 0, failed_diff_y = 0, failed_diff_t = 0;
  int failed_x = -1, failed_y = -1, failed_c = -1;
  float caffe_a = std::numeric_limits<float>::max(), caffe_b = std::numeric_limits<float>::lowest();
  float dv_a = std::numeric_limits<float>::max(), dv_b = std::numeric_limits<float>::lowest();
  dmp_dv_cmdraw_v0 cmd;
  char fnme[512];
  uint16_t quant_map[256];
  FILE *fin;
  std::vector<__fp16> caffe_input;
  std::vector<uint8_t> caffe_weights;
  std::vector<__fp16> caffe_bias;
  std::vector<std::vector<__fp16> > caffe_outputs;
  int n;
  char c;
  int out_width, out_height;
  bool fend;
  int i_conf;

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

  // Outer loop by configurations to be packed in the single command list
  for (auto it = confs.cbegin(); it != confs.cend(); ++it) {
    snprintf(prefix, sizeof(prefix), "data/%dx%dx%d_%dx%dx%d_pad%dx%dx%dx%d_stride%dx%d_act%d",
             it->width, it->height, it->n_channels, it->kx, it->ky, it->n_kernels,
             it->pad_left, it->pad_top, it->pad_right, it->pad_bottom,
             it->stride_x, it->stride_y, it->activation);

    // Load quantization map
    snprintf(fnme, sizeof(fnme), "%s.q.bin", prefix);
    fin = fopen(fnme, "rb");
    if (!fin) {
      ERR("fopen() failed for %s\n", fnme);
      goto L_EXIT;
    }
    n = fread(quant_map, sizeof(quant_map[0]), sizeof(quant_map) / sizeof(quant_map[0]), fin);
    fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
    fclose(fin);
    if (n != 256) {
      ERR("fread() returned %d while expecting %d for %s\n", n, 256, fnme);
      goto L_EXIT;
    }
    if (!fend) {
      ERR("File is bigger than expected: %s\n", fnme);
      goto L_EXIT;
    }

    // Load input
    snprintf(fnme, sizeof(fnme), "%s.i.bin", prefix);
    fin = fopen(fnme, "rb");
    if (!fin) {
      ERR("fopen() failed for %s\n", fnme);
      goto L_EXIT;
    }
    caffe_input.resize(it->n_channels * it->height * it->width);
    n = fread(caffe_input.data(), sizeof(caffe_input[0]), caffe_input.size(), fin);
    fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
    fclose(fin);
    if (n != it->n_channels * it->height * it->width) {
      ERR("fread() returned %d while expecting %d for %s\n", n, it->n_channels * it->height * it->width, fnme);
      goto L_EXIT;
    }
    if (!fend) {
      ERR("File is bigger than expected: %s\n", fnme);
      goto L_EXIT;
    }

    // Load weights
    snprintf(fnme, sizeof(fnme), "%s.w.bin", prefix);
    fin = fopen(fnme, "rb");
    if (!fin) {
      ERR("fopen() failed for %s\n", fnme);
      goto L_EXIT;
    }
    caffe_weights.resize(it->n_kernels * it->n_channels * it->kx * it->ky);
    n = fread(caffe_weights.data(), sizeof(caffe_weights[0]), caffe_weights.size(), fin);
    fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
    fclose(fin);
    if (n != it->n_kernels * it->n_channels * it->kx * it->ky) {
      ERR("fread() returned %d while expecting %d for %s\n",
          n, it->n_kernels * it->n_channels * it->kx * it->ky, fnme);
      goto L_EXIT;
    }
    if (!fend) {
      ERR("File is bigger than expected: %s\n", fnme);
      goto L_EXIT;
    }

    // Load bias
    snprintf(fnme, sizeof(fnme), "%s.b.bin", prefix);
    fin = fopen(fnme, "rb");
    if (!fin) {
      ERR("fopen() failed for %s\n", fnme);
      goto L_EXIT;
    }
    caffe_bias.resize(it->n_kernels);
    n = fread(caffe_bias.data(), sizeof(caffe_bias[0]), caffe_bias.size(), fin);
    fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
    fclose(fin);
    if (n != it->n_kernels) {
      ERR("fread() returned %d while expecting %d for %s\n", n, it->n_kernels, fnme);
      goto L_EXIT;
    }
    if (!fend) {
      ERR("File is bigger than expected: %s\n", fnme);
      goto L_EXIT;
    }

    // Load output
    {
      std::vector<__fp16> caffe_output;
      snprintf(fnme, sizeof(fnme), "%s.o.bin", prefix);
      fin = fopen(fnme, "rb");
      if (!fin) {
        ERR("fopen() failed for %s\n", fnme);
        goto L_EXIT;
      }
      out_width = get_conv_out_width(it->width, it->kx, it->pad_left, it->pad_right, it->stride_x);
      out_height = get_conv_out_width(it->height, it->ky, it->pad_top, it->pad_bottom, it->stride_y);
      caffe_output.resize(out_width * out_height * it->n_kernels);
      n = fread(caffe_output.data(), sizeof(caffe_output[0]), caffe_output.size(), fin);
      fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
      fclose(fin);
      if (n != out_width * out_height * it->n_kernels) {
        ERR("fread() returned %d while expecting %d for %s\n", n, out_width * out_height * it->n_kernels, fnme);
        goto L_EXIT;
      }
      if (!fend) {
        ERR("File is bigger than expected: %s\n", fnme);
        goto L_EXIT;
      }
      caffe_outputs.push_back(std::move(caffe_output));
    }

    memset(&cmd, 0, sizeof(cmd));
    cmd.size = sizeof(cmd);
    cmd.version = 0;
    cmd.w = it->width;
    cmd.h = it->height;
    cmd.c = it->n_channels;
    cmd.z = 1;
    cmd.topo = 1;
    cmd.run[0].m = it->n_kernels;
    cmd.run[0].conv_enable = 1;
    cmd.run[0].p = (uint16_t)it->kx | (((uint16_t)it->ky) << 8);
    cmd.run[0].pz = 1;
    cmd.run[0].conv_pad = (uint32_t)it->pad_left | ((uint32_t)it->pad_right << 8) |
                          ((uint32_t)it->pad_top << 16) | ((uint32_t)it->pad_bottom << 24);
    cmd.run[0].conv_stride = (uint16_t)it->stride_x | ((uint16_t)it->stride_y << 8);
    cmd.run[0].actfunc = it->activation;

    io_size = (it->width * it->height * it->n_channels + out_width * out_height * it->n_kernels) * sizeof(__fp16);
    io_mem = dmp_dv_mem_alloc(ctx, io_size);
    if (!io_mem) {
      ERR("dmp_dv_mem_alloc() failed for %zu bytes: %s\n", io_size, dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    io_mems.push_back(io_mem);
    LOG("Allocated %zu (%zu requested) bytes for input/output\n", dmp_dv_mem_get_size(io_mem), io_size);
    cmd.input_buf.mem = io_mem;
    cmd.input_buf.offs = 0;
    cmd.output_buf.mem = io_mem;
    cmd.output_buf.offs = it->width * it->height * it->n_channels * sizeof(__fp16);

    weights_size = 0;
    if (dmp_dv_pack_conv_weights(
            it->n_channels, it->kx, it->ky, it->n_kernels,
            quant_map, NULL, NULL, NULL, &weights_size)) {
      ERR("dmp_dv_pack_conv_weights() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    weights_mem = dmp_dv_mem_alloc(ctx, weights_size);
    if (!weights_mem) {
      ERR("dmp_dv_mem_alloc() failed for %zu bytes: %s\n", weights_size, dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    weights_mems.push_back(weights_mem);
    LOG("Allocated %zu (%zu requested) bytes for weights\n", dmp_dv_mem_get_size(weights_mem), weights_size);
    cmd.run[0].weight_buf.mem = weights_mem;
    cmd.run[0].weight_buf.offs = 0;
    cmd.run[0].weight_fmt = 3;

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
          it->n_channels, it->kx, it->ky, it->n_kernels,
          quant_map, caffe_weights.data(), (const uint16_t*)caffe_bias.data(), weights, &weights_size)) {
      ERR("dmp_dv_pack_conv_weights() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    if (dmp_dv_mem_sync_end(weights_mem)) {
      ERR("dmp_dv_mem_sync_end() failed for weights: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    dmp_dv_mem_unmap(weights_mem);

    io_ptr = (__fp16*)dmp_dv_mem_map(io_mem);
    if (!io_ptr) {
      ERR("dmp_dv_mem_map() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    io_ptrs.push_back(io_ptr);
    if (dmp_dv_mem_sync_start(io_mem, 0, 1)) {
      ERR("dmp_dv_mem_sync_start() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    // Caffe's input is stored as channel, height, width
    // DV input should be stored as chunks by max of 8 channels as width, height, channel
    for (int chan_group = 0, o_offs = 0; chan_group < it->n_channels; chan_group += 8) {
      const int last_chan = std::min(chan_group + 8, it->n_channels);
      for (int i = 0; i < it->width; ++i) {
        for (int j = 0; j < it->height; ++j) {
          for (int k = chan_group; k < last_chan; ++k, ++o_offs) {
            const int i_offs = k * it->width * it->height + j * it->width + i;
            const __fp16 vle = caffe_input[i_offs];
            io_ptr[o_offs] = vle;
          }
        }
      }
    }
    memset(io_ptr + it->width * it->height * it->n_channels, 0, out_width * out_height * it->n_kernels * sizeof(__fp16));
    if (dmp_dv_mem_sync_end(io_mem)) {
      ERR("dmp_dv_mem_sync_end() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    //print_cmd(cmd);

    if (dmp_dv_cmdlist_add_raw(cmdlist, (dmp_dv_cmdraw*)&cmd)) {
      ERR("dmp_dv_cmdlist_add_raw() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
  }

  if (dmp_dv_cmdlist_end(cmdlist)) {
    ERR("dmp_dv_cmdlist_end() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Ended the command list\n");

  if (dmp_dv_cmdlist_exec(cmdlist) < 0) {
    ERR("dmp_dv_cmdlist_exec() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Scheduled command list for execution\n");

  LOG("Waiting for completion\n");
  if (dmp_dv_wait_all(ctx)) {
    ERR("dmp_dv_sync() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Execution has completed\n");

  i_conf = 0;
  for (auto it = confs.cbegin(); it != confs.cend(); ++it, ++i_conf) {
    if (dmp_dv_mem_sync_start(io_mems[i_conf], 1, 0)) {
      ERR("dmp_dv_mem_sync_start() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    // Caffe's output is stored as channels, height, width
    // DV output is stored as chunks by max of 8 channels as width, height, channel
    out_width = get_conv_out_width(it->width, it->kx, it->pad_left, it->pad_right, it->stride_x);
    out_height = get_conv_out_width(it->height, it->ky, it->pad_top, it->pad_bottom, it->stride_y);
    io_ptr = io_ptrs[i_conf];
    for (int chan_group = 0, o_offs = it->width * it->height * it->n_channels;
         chan_group < it->n_kernels; chan_group += 8) {
      const int last_chan = std::min(chan_group + 8, it->n_kernels);
      for (int i = 0; i < out_width; ++i) {
        for (int j = 0; j < out_height; ++j) {
          for (int k = chan_group; k < last_chan; ++k, ++o_offs) {
            const int i_offs = k * out_width * out_height + j * out_width + i;
            const __fp16 vle = caffe_outputs[i_conf][i_offs];
            const float y = (float)io_ptr[o_offs], t = (float)vle;
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
      ERR("FAILED: failed_diff=%.6f on y=%.6f and t=%.6f xy=(%d, %d) chan=%d %s\n", failed_diff, failed_diff_y, failed_diff_t,
          failed_x, failed_y, failed_c, prefix);
      goto L_EXIT;
    }
  }

  result = 0;
  LOG("SUCCESS: test_cmdlist\n");

  L_EXIT:
  dmp_dv_cmdlist_release(cmdlist);
  const int n_confs = (int)confs.size();
  for (int i = n_confs - 1; i >= 0; --i) {
    dmp_dv_mem_release(io_mems[i]);
    dmp_dv_mem_release(weights_mems[i]);
  }
  dmp_dv_context_release(ctx);

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

  LOG("EXIT: test_cmdlists: %d commands, %d FDs:", (int)confs.size(), n_fd);
  for (auto it = confs.cbegin(); it != confs.cend(); ++it) {
    snprintf(prefix, sizeof(prefix), "data/%dx%dx%d_%dx%dx%d_pad%dx%dx%dx%d_stride%dx%d_act%d",
             it->width, it->height, it->n_channels, it->kx, it->ky, it->n_kernels,
             it->pad_left, it->pad_top, it->pad_right, it->pad_bottom,
             it->stride_x, it->stride_y, it->activation);
    LOG(" %s", prefix);
  }
  LOG("\n");

  return result;
}


int main(int argc, char **argv) {
  g_flog = fopen(argc > 1 ? argv[1] : "/dev/null", "w");
  if (!g_flog) {
    fprintf(stderr, "fopen() failed for %s\n", argv[1]);
    fflush(stderr);
    return -1;
  }

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

    if (sscanf(fnme, "%d%d%d%d%d%d%d%d%d%d%d%d%d",
               &config.width, &config.height, &config.n_channels, &config.kx, &config.ky, &config.n_kernels,
               &config.pad_left, &config.pad_top, &config.pad_right, &config.pad_bottom,
               &config.stride_x, &config.stride_y, &config.activation) != 13) {
      continue;
    }

    configs.emplace(std::move(config));
  }
  closedir(d);

  const int n_configs = (int)configs.size();
  const size_t pack_sizes[4] = {1, 2, 20, 200};
  for (int i_pack = 0; i_pack < 4; ++i_pack) {
    std::vector<conv_config> confs;
    int i_config = 0;
    for (auto it = configs.cbegin(); it != configs.cend(); ++it, ++i_config) {
      confs.push_back(*it);
      if ((confs.size() < pack_sizes[i_pack]) && (i_config < n_configs - 1)) {
        continue;
      }
      res = test_cmdlists(confs);
      if (res) {
        ++n_err;
      }
      else {
        ++n_ok;
      }
      confs.clear();
    }
  }

  LOG("Tests succeeded: %d\n", n_ok);
  LOG("Tests failed: %d\n", n_err);

  fclose(g_flog);

  return n_err;
}
