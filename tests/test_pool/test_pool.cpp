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
 * @brief Tests Pooling layer.
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
#include <algorithm>
#include <random>
#include <chrono>

#include <openssl/sha.h>

#include "dmp_dv.h"
#include "dmp_dv_cmdraw_v0.h"


static int g_sleep = 0;


static inline uint16_t fp16_as_int(float x32) {
  __fp16 x = (__fp16)x32;
  __fp16 *px = &x;
  uint16_t *py = (uint16_t*)px;
  uint16_t y = *py;
  return y;
}


/// @brief Rounds up "a" to be the multiple of "n".
static inline int roundup(int a) {
  const int n = 64;
  int d = a % n;
  return d ? a + (n - d) : a;
}


/// @brief File which will store logs.
static FILE *g_flog = NULL;


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout); fprintf(g_flog, __VA_ARGS__); fflush(g_flog)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr); fprintf(g_flog, __VA_ARGS__); fflush(g_flog)


/// @brief Number of file descriptors for the process.
static int g_n_fd = -1;
#define N_T 6
static const float g_max_diff_bounds[N_T] = {0.1f, 0.5f, 1.0f, 5.0f, 10.0f, 1000000.0f};
static float g_max_diff[N_T] = {0, 0, 0, 0, 0, 0};
static float g_max_diff_y[N_T] = {0, 0, 0, 0, 0, 0};
static float g_max_diff_t[N_T] = {0, 0, 0, 0, 0, 0};


/// @brief Configuration description to be tested.
typedef struct conv_config_impl {
  int width, height, n_channels, pool_type;
  int pool_kx, pool_ky, pool_stride_x, pool_stride_y,
      pool_pad_left, pool_pad_right, pool_pad_top, pool_pad_bottom;
  bool hash_set;
  bool failed;
  int out_width, out_height;
  uint8_t hash[32];
  dmp_dv_mem io_mem;
  size_t io_size, io_offs, weights_offs;
  __fp16 *io_ptr;
  __fp16 *caffe_output;
  bool pure_ints;  // only integers were used - error check should be exact

  bool operator <(const struct conv_config_impl& pt) const {
    return std::make_tuple(width, height, n_channels, pool_type, pool_kx, pool_ky, pool_stride_x, pool_stride_y,
                           pool_pad_left, pool_pad_right, pool_pad_top, pool_pad_bottom) <
        std::make_tuple(pt.width, pt.height, pt.n_channels, pt.pool_type, pt.pool_kx, pt.pool_ky, pt.pool_stride_x, pt.pool_stride_y,
                        pt.pool_pad_left, pt.pool_pad_right, pt.pool_pad_top, pt.pool_pad_bottom);
  }
} conv_config;


/// @brief Prints command content for debugging.
void print_cmd(dmp_dv_cmdraw_conv_v0& cmd) {
  //return;
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


/// @brief Checks if error is acceptable.
/// @param y Output to check.
/// @param t Target.
/// @param conf Executed convolutional configuration.
/// @return 0 if error is acceptable, non-zero otherwise.
int check_err(float y, float t, conv_config *conf) {
  const float diff = std::abs(y - t);
  return diff < 0.1f ? 0 : -1;
}


/// @brief Checks if an array has a float.
bool has_float(__fp16 *x, int n) {
  for (int i = 0; i < n; ++i) {
    float vle = (float)x[i];
    if (vle != (float)(int)vle) {
      return true;
    }
  }
  return false;
}


static inline int get_conv_out_width(int width, int kx, int pad_left, int pad_right, int stride) {
  return (pad_left + width + pad_right - kx) / stride + 1;
}


/// @brief Tests convolutional configurations for correctness using data from folder "data".
int test_cmdlists(const std::vector<conv_config*>& confs) {
  char prefix[256];
  LOG("ENTER: test_cmdlists: %d commands:", (int)confs.size());
  for (auto it = confs.begin(); it != confs.end(); ++it) {
    conv_config *conf = *it;
    snprintf(prefix, sizeof(prefix), "data/%dx%dx%d_t%d/%dx%d_s%dx%d_p%dx%dx%dx%d",
             conf->width, conf->height, conf->n_channels, conf->pool_type, conf->pool_kx, conf->pool_ky,
             conf->pool_stride_x, conf->pool_stride_y, conf->pool_pad_left, conf->pool_pad_right,
             conf->pool_pad_top, conf->pool_pad_bottom);
    LOG(" %s", prefix);
  }
  LOG("\n");

  int result = -1;
  dmp_dv_context ctx = NULL;
  dmp_dv_cmdlist cmdlist = NULL;
  float failed_diff = 0, failed_diff_y = 0, failed_diff_t = 0;
  int failed_x = -1, failed_y = -1, failed_c = -1;
  float caffe_a = std::numeric_limits<float>::max(), caffe_b = std::numeric_limits<float>::lowest();
  float dv_a = std::numeric_limits<float>::max(), dv_b = std::numeric_limits<float>::lowest();
  struct dmp_dv_cmdraw_conv_v0 cmd;
  char fnme[512];
  FILE *fin;
  std::vector<__fp16> caffe_input;
  int n;
  char c;
  bool fend;
  int64_t exec_id;

  ctx  = dmp_dv_context_create();
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

  // Outer loop by configurations to be packed in the single command list
  for (auto it = confs.begin(); it != confs.end(); ++it) {
    conv_config *conf = *it;
    snprintf(prefix, sizeof(prefix), "data/%dx%dx%d_t%d/%dx%d_s%dx%d_p%dx%dx%dx%d",
             conf->width, conf->height, conf->n_channels, conf->pool_type, conf->pool_kx, conf->pool_ky,
             conf->pool_stride_x, conf->pool_stride_y, conf->pool_pad_left, conf->pool_pad_right,
             conf->pool_pad_top, conf->pool_pad_bottom);

    // Load input
    snprintf(fnme, sizeof(fnme), "%s.i.bin", prefix);
    fin = fopen(fnme, "rb");
    if (!fin) {
      ERR("fopen() failed for %s\n", fnme);
      goto L_EXIT;
    }
    caffe_input.resize(conf->n_channels * conf->height * conf->width);
    n = fread(caffe_input.data(), sizeof(caffe_input[0]), caffe_input.size(), fin);
    fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
    fclose(fin);
    if (n != conf->n_channels * conf->height * conf->width) {
      ERR("fread() returned %d while expecting %d for %s\n", n, conf->n_channels * conf->height * conf->width, fnme);
      goto L_EXIT;
    }
    if (!fend) {
      ERR("File is bigger than expected: %s\n", fnme);
      goto L_EXIT;
    }

    // Find if we are using only integers or not
    conf->pure_ints = (!has_float(caffe_input.data(), caffe_input.size()));

    // Load output
    conf->out_width = get_conv_out_width(conf->width, conf->pool_kx, conf->pool_pad_left, conf->pool_pad_right, conf->pool_stride_x);
    conf->out_height = get_conv_out_width(conf->height, conf->pool_ky, conf->pool_pad_top, conf->pool_pad_bottom, conf->pool_stride_y);
    if (!conf->hash_set) {
      snprintf(fnme, sizeof(fnme), "%s.o.bin", prefix);
      fin = fopen(fnme, "rb");
      if (!fin) {
        ERR("fopen() failed for %s\n", fnme);
        goto L_EXIT;
      }
      conf->caffe_output = (__fp16*)malloc(conf->n_channels * conf->out_height * conf->out_width * sizeof(__fp16));
      n = fread(conf->caffe_output, sizeof(__fp16), conf->n_channels * conf->out_height * conf->out_width, fin);
      fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
      fclose(fin);
      if (n != conf->n_channels * conf->out_height * conf->out_width) {
        ERR("fread() returned %d while expecting %d for %s\n", n, conf->n_channels * conf->out_height * conf->out_width, fnme);
        goto L_EXIT;
      }
      if (!fend) {
        ERR("File is bigger than expected: %s\n", fnme);
        goto L_EXIT;
      }
    }

    memset(&cmd, 0, sizeof(cmd));
    cmd.header.size = sizeof(cmd);
    cmd.header.device_type = DMP_DV_DEV_CONV;
    cmd.header.version = 0;
    cmd.w = conf->width;
    cmd.h = conf->height;
    cmd.c = conf->n_channels;
    cmd.z = 1;
    cmd.topo = 1;
    cmd.run[0].m = conf->n_channels;
    cmd.run[0].conv_enable = 0;
    cmd.run[0].p = 1;
    cmd.run[0].pz = 1;
    cmd.run[0].conv_pad = 0;
    cmd.run[0].conv_stride = 0x0101;
    cmd.run[0].actfunc = 0;
    cmd.run[0].lrn = 0;
    cmd.run[0].pool_enable = conf->pool_type;
    cmd.run[0].pool_size = ((uint16_t)(conf->pool_kx)) | (((uint16_t)(conf->pool_ky)) << 8);
    cmd.run[0].pool_stride = ((uint16_t)(conf->pool_stride_x)) | (((uint16_t)(conf->pool_stride_y)) << 8);
    cmd.run[0].pool_pad = ((uint32_t)(conf->pool_pad_left)) | (((uint32_t)(conf->pool_pad_right)) << 8) |
                          (((uint32_t)(conf->pool_pad_top)) << 16) | (((uint32_t)(conf->pool_pad_bottom)) << 24);
    cmd.run[0].pool_avg_param = (conf->pool_type == 2) ? fp16_as_int(1.0f / (conf->pool_kx * conf->pool_ky)) : 0;

    conf->io_offs = 4096;
    conf->io_size = (roundup(conf->width * conf->height * conf->n_channels) + roundup(conf->out_width * conf->out_height * conf->n_channels) + 2048) * sizeof(__fp16);
    conf->io_mem = dmp_dv_mem_alloc(ctx, conf->io_offs + conf->io_size);
    if (!conf->io_mem) {
      ERR("dmp_dv_mem_alloc() failed for %zu bytes: %s\n", conf->io_offs + conf->io_size, dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    LOG("Allocated %zu (%zu(+%zu random offset) requested) bytes for input/output\n", dmp_dv_mem_get_size(conf->io_mem), conf->io_size, conf->io_offs);
    cmd.input_buf.mem = conf->io_mem;
    cmd.input_buf.offs = conf->io_offs;
    cmd.output_buf.mem = conf->io_mem;
    cmd.output_buf.offs = conf->io_offs + roundup(conf->width * conf->height * conf->n_channels) * sizeof(__fp16);

    conf->io_ptr = (__fp16*)dmp_dv_mem_map(conf->io_mem);
    if (!conf->io_ptr) {
      ERR("dmp_dv_mem_map() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    if (dmp_dv_mem_sync_start(conf->io_mem, 1, 1)) {
      ERR("dmp_dv_mem_sync_start() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    for (int i = 0; i < (int)((conf->io_offs + conf->io_size) >> 1); ++i) {
      conf->io_ptr[i] = (__fp16)-65497.0f;
    }

    conf->io_ptr += conf->io_offs >> 1;

    // Caffe's input is stored as channel, height, width
    // DV input should be stored as chunks by max of 8 channels as width, height, channel
    for (int chan_group = 0, o_offs = 0; chan_group < conf->n_channels; chan_group += 8) {
      const int last_chan = std::min(chan_group + 8, conf->n_channels);
      for (int i = 0; i < conf->width; ++i) {
        for (int j = 0; j < conf->height; ++j) {
          for (int k = chan_group; k < last_chan; ++k, ++o_offs) {
            const int i_offs = k * conf->width * conf->height + j * conf->width + i;
            const __fp16 vle = caffe_input[i_offs];
            conf->io_ptr[o_offs] = vle;
          }
        }
      }
    }
    if (dmp_dv_mem_sync_end(conf->io_mem)) {
      ERR("dmp_dv_mem_sync_end() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    print_cmd(cmd);

    if (dmp_dv_cmdlist_add_raw(cmdlist, (dmp_dv_cmdraw*)&cmd)) {
      ERR("dmp_dv_cmdlist_add_raw() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
  }

  if (dmp_dv_cmdlist_commit(cmdlist)) {
    ERR("dmp_dv_cmdlist_commit() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Commited the command list\n");

  exec_id = dmp_dv_cmdlist_exec(cmdlist);
  if (exec_id < 0) {
    ERR("dmp_dv_cmdlist_exec() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Scheduled command list for execution\n");

  LOG("Waiting for completion\n");
  if (dmp_dv_cmdlist_wait(cmdlist, exec_id)) {
    ERR("dmp_dv_sync() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Execution has completed\n");
  if (g_sleep > 0) {
    usleep(g_sleep);
  }

  for (auto it = confs.begin(); it != confs.end(); ++it) {
    conv_config *conf = *it;
    __fp16 *base = conf->io_ptr - (conf->io_offs >> 1);

    if (dmp_dv_mem_sync_start(conf->io_mem, 1, 1)) {
      ERR("dmp_dv_mem_sync_start() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    for (int i = 0; i < (int)(conf->io_offs >> 1); ++i) {
      if (base[i] != (__fp16)-65497.0f) {
        ERR("Detected memory modification in area before input: %.6f\n", (float)base[i]);
        goto L_EXIT;
      }
    }
    for (int i = (conf->io_offs >> 1) + conf->width * conf->height * conf->n_channels;
         i < (int)(conf->io_offs >> 1) + roundup(conf->width * conf->height * conf->n_channels); ++i) {
      if (base[i] != (__fp16)-65497.0f) {
        ERR("Detected memory modification in area after input but before output: %.6f\n", (float)base[i]);
        goto L_EXIT;
      }
    }
    for (int i = (conf->io_offs >> 1) + roundup(conf->width * conf->height * conf->n_channels) + conf->width * conf->height * conf->n_channels;
         i < (int)(conf->io_size >> 1); ++i) {
      if (base[i] != (__fp16)-65497.0f) {
        ERR("Detected memory modification in area after output: %.6f\n", (float)base[i]);
        goto L_EXIT;
      }
    }
    LOG("No memory overwriting outside of the output detected\n");

    // Compare output with the gold one
    const int o_offs = roundup(conf->width * conf->height * conf->n_channels);
    if (conf->hash_set) {  // check hash
      uint8_t hash[32];
      SHA256_CTX sha256;
      SHA256_Init(&sha256);
      SHA256_Update(&sha256, conf->io_ptr + o_offs, conf->out_width * conf->out_height * conf->n_channels * sizeof(__fp16));
      SHA256_Final(hash, &sha256);
      if (memcmp(conf->hash, hash, 32)) {
        ERR("Hash differs\n");
        goto L_EXIT;
      }
    }
    else {  // approximately compare output
      // Caffe's output is stored as channels, height, width
      // DV output is stored as chunks by max of 8 channels as width, height, channel
      float max_diff[N_T], max_diff_y[N_T], max_diff_t[N_T];
      memset(max_diff, 0, sizeof(max_diff));
      memset(max_diff_y, 0, sizeof(max_diff_y));
      memset(max_diff_t, 0, sizeof(max_diff_t));
      int oo_offs = o_offs;
      std::vector<float> outp;
      outp.resize(conf->out_width * conf->out_height * conf->n_channels);
      std::set<int> offsets;
      for (int chan_group = 0; chan_group < conf->n_channels; chan_group += 8) {
        const int last_chan = std::min(chan_group + 8, conf->n_channels);
        for (int i = 0; i < conf->out_width; ++i) {
          for (int j = 0; j < conf->out_height; ++j) {
            for (int k = chan_group; k < last_chan; ++k, ++oo_offs) {
              const int i_offs = k * conf->out_width * conf->out_height + j * conf->out_width + i;
              const __fp16 vle = conf->caffe_output[i_offs];
              const float y = (float)conf->io_ptr[oo_offs], t = (float)vle;
              outp[i_offs] = y;
              offsets.insert(i_offs);
              caffe_a = std::min(caffe_a, t);
              caffe_b = std::max(caffe_b, t);
              dv_a = std::min(dv_a, y);
              dv_b = std::max(dv_b, y);
              const float diff = std::abs(y - t);
              if (check_err(y, t, conf)) {
                conf->failed = true;
                if (diff > failed_diff) {
                  failed_diff = diff;
                  failed_diff_y = y;
                  failed_diff_t = t;
                  failed_x = i;
                  failed_y = j;
                  failed_c = k;
                }
              }
              else {
                for (int ii = 0; ii < N_T; ++ii) {
                  if (std::abs(t) > g_max_diff_bounds[ii]) {
                    continue;
                  }
                  if (diff > max_diff[ii]) {
                    max_diff[ii] = diff;
                    max_diff_y[ii] = y;
                    max_diff_t[ii] = t;
                  }
                  break;
                }
              }
            }
          }
        }
      }
      LOG("caffe: [%.6f, %.6f] dv: [%.6f, %.6f]\n", caffe_a, caffe_b, dv_a, dv_b);
      for (int i = 0; i < N_T; ++i) {
        LOG("t <= %.1f: max_diff=%.6f on y=%.6f and t=%.6f\n",
            g_max_diff_bounds[i], max_diff[i], max_diff_y[i], max_diff_t[i]);
      }
      for (int i = 0; i < N_T; ++i) {
        if (max_diff[i] > g_max_diff[i]) {
          g_max_diff[i] = max_diff[i];
          g_max_diff_y[i] = max_diff_y[i];
          g_max_diff_t[i] = max_diff_t[i];
        }
      }
      snprintf(fnme, sizeof(fnme), "%s.y.bin", prefix);
      FILE *fout = fopen(fnme, "wb");
      if (fout) {
        fwrite(outp.data(), 4, outp.size(), fout);
        fclose(fout);
        LOG("Saved real output to %s\n", fnme);
        for (int i = 0; i < (int)outp.size(); ++i) {
          if (offsets.find(i) == offsets.end()) {
            ERR("Offset %d is missing\n", i);
            break;
          }
        }
      }
      else {
        ERR("fopen() failed for %s\n", fnme);
      }
      if (failed_diff > 0.0f) {
        ERR("FAILED: failed_diff=%.6f on y=%.6f and t=%.6f xy=(%d, %d) chan=%d %s\n", failed_diff, failed_diff_y, failed_diff_t,
            failed_x, failed_y, failed_c, prefix);
        goto L_EXIT;
      }
      if (!conf->failed) {  // compute hash
        const int output_size = conf->out_height * conf->out_width * conf->n_channels * sizeof(__fp16);
        if (o_offs * sizeof(__fp16) + output_size > conf->io_size) {
          ERR("Incorrect allocation size for input/output");
          _exit(-1);
        }
        SHA256_CTX sha256;
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, conf->io_ptr + o_offs, output_size);
        SHA256_Final(&conf->hash[0], &sha256);
        conf->hash_set = true;
      }
    }
    if (dmp_dv_mem_sync_end(conf->io_mem)) {
      ERR("dmp_dv_mem_sync_end() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
  }

  result = 0;
  LOG("SUCCESS: test_cmdlist\n");

  L_EXIT:
  dmp_dv_cmdlist_release(cmdlist);
  for (auto it = confs.rbegin(); it != confs.rend(); ++it) {
    conv_config *conf = *it;
    if (conf->caffe_output) {
      free(conf->caffe_output);
      conf->caffe_output = NULL;
    }
    dmp_dv_mem_release(conf->io_mem);
    conf->io_mem = NULL;
    conf->io_ptr = NULL;
    conf->io_size = 0;
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
    ERR("Inconsistent file descriptor count detected, memory leak is probable\n");
    result = -1;
  }

  LOG("EXIT: test_cmdlists: %d commands, %d FDs:", (int)confs.size(), n_fd);
  for (auto it = confs.begin(); it != confs.end(); ++it) {
    conv_config *conf = *it;
    snprintf(prefix, sizeof(prefix), "data/%dx%dx%d_t%d/%dx%d_s%dx%d_p%dx%dx%dx%d",
             conf->width, conf->height, conf->n_channels, conf->pool_type, conf->pool_kx, conf->pool_ky,
             conf->pool_stride_x, conf->pool_stride_y, conf->pool_pad_left, conf->pool_pad_right,
             conf->pool_pad_top, conf->pool_pad_bottom);
    LOG(" %s", prefix);
  }
  LOG("\n");

  return result;
}


/// @brief Entry point.
/// @details All messages will be logged to filename argv[1] if argc > 1.
int main(int argc, char **argv) {
  g_flog = fopen(argc > 1 ? argv[1] : "/dev/null", "w");
  if (!g_flog) {
    fprintf(stderr, "fopen() failed for %s\n", argv[1]);
    fflush(stderr);
    return -1;
  }

  if (argc > 2) {
    g_sleep = atoi(argv[2]);
  }
  else {
    g_sleep = 0;
  }
  LOG("Will add delay of %d usec\n", g_sleep);

  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  std::shared_ptr<std::set<conv_config> > config_set = std::make_shared<std::set<conv_config> >();

  DIR *d, *subd;
  struct dirent *dir, *subdir;
  d = opendir("data");
  if (!d) {
    ERR("Could not open \"data\" folder\n");
    return -1;
  }
  while ((dir = readdir(d))) {
    conv_config config;
    memset(&config, 0, sizeof(config));

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
    if (sscanf(fnme, "%d%d%d%d", &config.width, &config.height, &config.n_channels, &config.pool_type) != 4) {
      continue;
    }
    snprintf(fnme, sizeof(fnme), "data/%s", dir->d_name);
    subd = opendir(fnme);
    if (!subd) {
      continue;
    }

    while ((subdir = readdir(subd))) {
      for (int i = 0; ; ) {
        char c = subdir->d_name[i];
        fnme[i] = ((c >= '0') && (c <= '9')) ? c : ' ';
        ++i;
        if (!subdir->d_name[i]) {
          fnme[i] = 0;
          break;
        }
      }

      if (sscanf(fnme, "%d%d%d%d%d%d%d%d",
                 &config.pool_kx, &config.pool_ky, &config.pool_stride_x, &config.pool_stride_y,
                 &config.pool_pad_left, &config.pool_pad_right, &config.pool_pad_top, &config.pool_pad_bottom) != 8) {
        continue;
      }

      char prefix[256];
      snprintf(prefix, sizeof(prefix), "data/%dx%dx%d_t%d/%dx%d_s%dx%d_p%dx%dx%dx%d",
               config.width, config.height, config.n_channels, config.pool_type, config.pool_kx, config.pool_ky,
               config.pool_stride_x, config.pool_stride_y, config.pool_pad_left, config.pool_pad_right,
               config.pool_pad_top, config.pool_pad_bottom);

      const size_t prev_size = (int)config_set->size();
      config_set->emplace(config);
      const size_t this_size = (int)config_set->size();

      if ((this_size != prev_size) && (!(this_size % 1000))) {
        LOG("Loaded %zu configurations\n", this_size);
      }
    }
    closedir(subd);
  }
  closedir(d);
  LOG("Loaded %zu configurations\n", config_set->size());

  // Copy configurations from the set to obtain fixed memory pointers to elements
  std::vector<conv_config> config_data;
  for (auto it = config_set->cbegin(); it != config_set->cend(); ++it) {
    config_data.push_back(*it);
  }
  config_set->clear();
  config_set.reset();  // delete the set

  const int n_configs = (int)config_data.size();
  std::vector<conv_config*> configs;
  configs.resize(n_configs);
  for (int i = 0; i < n_configs; ++i) {
    configs[i] = config_data.data() + i;
  }

  const int seed = 1234;  // time(NULL);
  std::mt19937 mt_rand(seed);
  std::uniform_int_distribution<int> rnd_offs(0, 1024);

  const int n_passes = 1;
  for (int i_pass = 0; i_pass < n_passes; ++i_pass) {
    // Randomize configrations order
    std::shuffle(configs.begin(), configs.end(), mt_rand);

    // Execute configurations in different chunk sizes
    const size_t pack_sizes[2] = {1, 50};
    const int n_packs = 1;
    for (int i_pack = 0; i_pack < n_packs; ++i_pack) {
      std::vector<conv_config*> confs;
      int i_config = 0;
      for (auto it = configs.begin(); it != configs.end(); ++it, ++i_config) {
        conv_config *conf = *it;
        if (conf->failed) {
          continue;
        }
        conf->io_offs = rnd_offs(mt_rand) << 4;
        conf->weights_offs = rnd_offs(mt_rand) << 4;
        confs.push_back(conf);
        if (confs.size() < pack_sizes[i_pack]) {
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
      if (confs.size()) {
        res = test_cmdlists(confs);
        if (res) {
          ++n_err;
        }
        else {
          ++n_ok;
        }
      }
    }
  }

  for (int i = 0; i < N_T; ++i) {
    LOG("t <= %.1f: g_max_diff=%.6f on y=%.6f and t=%.6f\n",
        g_max_diff_bounds[i], g_max_diff[i], g_max_diff_y[i], g_max_diff_t[i]);
  }

  LOG("Tests succeeded: %d\n", n_ok);
  LOG("Tests failed: %d\n", n_err);
  LOG("Number of configurations tested: %d\n", (int)configs.size());

  fclose(g_flog);

  return n_err;
}
