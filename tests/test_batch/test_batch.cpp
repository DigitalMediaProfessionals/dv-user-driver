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
 * @brief Tests batch (matrix x matrix multiplication as batch of vector x matrix multiplication via 1x1 convolution).
 */
#include <unistd.h>
#include <sys/mman.h>
#include <dirent.h>
#include <time.h>

#include <stdio.h>
#include <string.h>
#include <math.h>

#include <algorithm>
#include <random>
#include <memory>

#include <dmp_dv.h>
#include <dmp_dv_cmdraw_v0.h>
#include <stats.h>

#ifdef __x86_64__
#include "half.h"
typedef half_float::half __fp16;
#endif


typedef __fp16 zia_float;


/// @brief Offset within NWHC8 layout.
#define NWHC8_OFFS(n, h, w, c, i_n, i_h, i_w, i_c) \
    (((i_n) * ((w) * (h) * (c))) + /* sample beginning */ \
     ((((i_c) >> 3) << 3) * ((w) * (h))) + /* 8-channel beginning */ \
     (((i_w) * (h) + (i_h)) * std::min((c) - (((i_c) >> 3) << 3), 8)) + /* pixel start */ \
     ((i_c) & 7))


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)


#define ALIGN64(n) (((n) & 63) ? (n) + (64 - ((n) & 63)) : (n))


int verbosity = 0;


/// @brief Prints command content for debugging.
void print_cmd(dmp_dv_cmdraw_conv_v0& cmd) {
  if (verbosity <= 0) {
    return;
  }
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


/// @brief Returns width of the output based on kernel size, padding and stride.
int get_conv_out_width(int width, int kx, int pad_left, int pad_right, int stride, bool is_deconv) {
  if (is_deconv) {
    return stride * (width - 1) + kx - pad_left - pad_right;
  } else {
    return (pad_left + width + pad_right - kx) / stride + 1;
  }
}


static inline uint16_t pack2(uint16_t a, uint16_t b) {
  return a | (b << 8);
}

static inline uint32_t pack4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  return a | (b << 8) | (c << 16) | (d << 24);
}


int get_used_cma_mb() {
  FILE *fin = fopen("/proc/meminfo", "r");
  if (!fin) {
    return -1;
  }

  int cma_total = 0, cma_free = 0;

  char s[512];

  while (!feof(fin)) {
    if (!fgets(s, sizeof(s), fin)) {
      break;
    }
    int i = 0;
    for (; s[i] && (s[i] != ':'); ++i) {
      // Empty by design
    }
    if (!s[i]) {
      continue;
    }
    s[i] = 0;
    if (!strcmp(s, "CmaTotal")) {
      if (sscanf(s + i + 1, "%d", &cma_total) == 1) {
        cma_total >>= 10;
      }
    }
    else if (!strcmp(s, "CmaFree")) {
      if (sscanf(s + i + 1, "%d", &cma_free) == 1) {
        cma_free >>= 10;
      }
    }
  }

  fclose(fin);

  return cma_total - cma_free;
}


int test_batch_mm(std::mt19937& prng) {
  LOG("ENTER: test_batch\n");

  int result = -1;
  dmp_dv_context ctx = dmp_dv_context_create();
  dmp_dv_mem input_mem = NULL, output_mem = NULL, weights_mem = NULL;
  size_t packed_weights_size = 0;
  zia_float *x = NULL, *y = NULL, *weights_ptr = NULL;
  dmp_dv_cmdlist cmdlist = NULL;
  const char *s_verbosity = getenv("VERBOSITY");
  verbosity = s_verbosity ? atoi(s_verbosity) : 0;
  const char *s_n = getenv("N");
  const char *s_h = getenv("H");
  const char *s_w = getenv("W");
  const char *s_c = getenv("C");
  const char *s_m = getenv("M");
  const char *s_kx = getenv("KX");
  const char *s_stride = getenv("STRIDE");
  const char *s_pad = getenv("PAD");
  const char *s_dil = getenv("DIL");
  const char *s_dw = getenv("DW");
  const char *s_deconv = getenv("DECONV");
  const int n = s_n ? atoi(s_n) : 1000;
  const int h = s_h ? atoi(s_h) : 1;
  const int w = s_w ? atoi(s_w) : 1;
  const int c = s_c ? atoi(s_c) : 8;
  const int m = s_m ? atoi(s_m) : 16;
  const int kx = s_kx ? atoi(s_kx) : 1;
  const int ky = kx;
  const int pad = s_pad ? atoi(s_pad) : 0;
  const int stride = s_stride ? atoi(s_stride) : 1;
  const int dil = s_dil ? atoi(s_dil) : 1;
  const bool dw = s_dw ? (atoi(s_dw) == 1) : false;
  const bool deconv = s_deconv ? (atoi(s_deconv) == 1) : false;
  const zia_float vals[8] = {-4, -3, -2, -1, 1, 2, 3, 4};
  std::uniform_int_distribution<int> dis(0, 7);
  double dt_unroll = 0, dt_batch = 0;

  const int ow = get_conv_out_width(w, kx, pad, pad, stride, deconv);
  const int oh = get_conv_out_width(h, ky, pad, pad, stride, deconv);

  std::unique_ptr<zia_float> t_ptr(new zia_float[n * oh * ow * m]);
  zia_float *t = t_ptr.get();
  if (!t) {
    ERR("Failed to allocate %d bytes of host memory\n", n * oh * ow * m);
    goto L_EXIT;
  }
  memset(t, 0, n * oh * ow * m * sizeof(zia_float));

  input_mem = dmp_dv_mem_alloc(ctx, n * h * w * c * sizeof(zia_float));
  if (!input_mem) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  x = (zia_float*)dmp_dv_mem_map(input_mem);
  if (!x) {
    ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  output_mem = dmp_dv_mem_alloc(ctx, n * oh * ow * m * sizeof(zia_float));
  if (!output_mem) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  y = (__fp16*)dmp_dv_mem_map(output_mem);
  if (!y) {
    ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  {
    const int n_channels = dw ? 1 : c;
    const int n_kernels = m;
    const uint16_t *quant_map = NULL;
    const void *weights = NULL;
    const uint16_t *prelu = NULL;
    uint8_t *packed_weights = NULL;
    const int n_weights = n_channels * n_kernels * kx * ky;
    std::unique_ptr<zia_float> ww_ptr(new zia_float[n_weights]);
    zia_float *ww = ww_ptr.get();
    std::unique_ptr<zia_float> bias_ptr(new zia_float[n_kernels]);
    zia_float *bias = bias_ptr.get();
    if (!ww) {
      ERR("Failed to allocate %d bytes of host memory\n", n_weights);
      goto L_EXIT;
    }
    if (dil <= 1) {
      dmp_dv_pack_conv_weights(
          n_channels, kx, ky, n_kernels,
          quant_map,
          weights, (uint16_t*)bias, prelu,
          packed_weights, &packed_weights_size);
    }
    else {
      dmp_dv_pack_dil_weights(
          n_channels, kx, ky, n_kernels,
          quant_map,
          weights, (uint16_t*)bias, prelu,
          packed_weights, &packed_weights_size);
    }
    weights_mem = dmp_dv_mem_alloc(ctx, packed_weights_size);
    if (!weights_mem) {
      ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    weights_ptr = (__fp16*)dmp_dv_mem_map(weights_mem);
    if (!weights_ptr) {
      ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    if (dmp_dv_mem_sync_start(weights_mem, 1, 1)) {
      ERR("dmp_dv_mem_sync_start() failed for weights: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    for (int i = 0; i < n_weights; ++i) {
      ww[i] = vals[dis(prng)];
    }
    for (int i = 0; i < n_kernels; ++i) {
      bias[i] = vals[dis(prng)];
    }

    if (dil <= 1) {
      dmp_dv_pack_conv_weights(
          n_channels, kx, ky, n_kernels,
          quant_map,
          ww, (uint16_t*)bias, prelu,
          (uint8_t*)weights_ptr, &packed_weights_size);
    }
    else {
      dmp_dv_pack_dil_weights(
          n_channels, kx, ky, n_kernels,
          quant_map,
          ww, (uint16_t*)bias, prelu,
          (uint8_t*)weights_ptr, &packed_weights_size);
    }

    if (dmp_dv_mem_sync_end(weights_mem)) {
      ERR("dmp_dv_mem_sync_end() failed for weights: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
  }

  if ((dmp_dv_mem_sync_start(input_mem, 1, 1)) ||
      (dmp_dv_mem_sync_start(output_mem, 1, 1))) {
    ERR("dmp_dv_mem_sync_start() failed for I/O: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  for (int i = 0; i < n * h * w * c; ++i) {
    x[i] = vals[dis(prng)];
  }
  for (int i = 0; i < n * oh * ow * m; ++i) {
    y[i] = 0;
  }
  if ((dmp_dv_mem_sync_end(input_mem)) ||
      (dmp_dv_mem_sync_end(output_mem))) {
    ERR("dmp_dv_mem_sync_end() failed for I/O: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  // Inference using single command for each sample in the batch
  {
    long max_mem_kb = 0;
    double utime = 0, stime = 0;
    get_exec_stats(&max_mem_kb, &utime, &stime);
    LOG("Will create Unrolled command list: RSS=%d Mb CMA=%d Mb\n",
        (int)(max_mem_kb >> 10), get_used_cma_mb());

    cmdlist = dmp_dv_cmdlist_create(ctx);
    if (!cmdlist) {
      ERR("dmp_dv_cmdlist_create() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    for (int i_sample = 0; i_sample < n; ++i_sample) {
      struct dmp_dv_cmdraw_conv_v0 conf;
      memset(&conf, 0, sizeof(conf));
      conf.header.size = sizeof(conf);
      conf.header.device_type = DMP_DV_DEV_CONV;
      conf.header.version = 0;
      conf.topo = 1;
      conf.w = w;
      conf.h = h;
      conf.z = 1;
      conf.c = c;
      conf.input_buf.mem = input_mem;
      conf.input_buf.offs = i_sample * w * h * c * sizeof(zia_float);
      conf.output_buf.mem = output_mem;
      conf.output_buf.offs = i_sample * ow * oh * m * sizeof(zia_float);
      if (deconv) {
        conf.run[0].conv_enable = dw ? 7 : 5;
      }
      else {
        conf.run[0].conv_enable = dw ? 3 : 1;
      }
      conf.run[0].conv_stride = pack2(stride, stride);
      conf.run[0].conv_pad = pack4(pad, pad, pad, pad);
      conf.run[0].weight_buf.mem = weights_mem;
      conf.run[0].weight_buf.offs = 0;
      conf.run[0].m = m;
      conf.run[0].p = pack2(kx, ky);
      conf.run[0].pz = 1;

      print_cmd(conf);

      if (dmp_dv_cmdlist_add_raw(cmdlist, (struct dmp_dv_cmdraw*)&conf)) {
        ERR("dmp_dv_cmdlist_add_raw() failed: %s\n", dmp_dv_get_last_error_message());
        goto L_EXIT;
      }
    }

    get_exec_stats(&max_mem_kb, &utime, &stime);
    LOG("Will commit Unrolled command list: RSS=%d Mb CMA=%d Mb\n",
        (int)(max_mem_kb >> 10), get_used_cma_mb());

    if (dmp_dv_cmdlist_commit(cmdlist)) {
      ERR("dmp_dv_cmdlist_commit() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    get_exec_stats(&max_mem_kb, &utime, &stime);
    LOG("Will execute Unrolled command list: RSS=%d Mb CMA=%d Mb\n",
        (int)(max_mem_kb >> 10), get_used_cma_mb());

    TimeInterval ti;
    int64_t exec_id = dmp_dv_cmdlist_exec(cmdlist);
    if (exec_id < 0) {
      ERR("dmp_dv_cmdlist_exec() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    if (dmp_dv_cmdlist_wait(cmdlist, exec_id)) {
      ERR("dmp_dv_cmdlist_wait() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    dt_unroll = ti.get_ms();
    LOG("Unrolled completed in %.3f msec\n", dt_unroll);

    dmp_dv_cmdlist_release(cmdlist);
    cmdlist = NULL;
  }

  if (dmp_dv_mem_sync_start(output_mem, 1, 1)) {
    ERR("dmp_dv_mem_sync_start() failed for output: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  memcpy(t, y, n * oh * ow * m * sizeof(zia_float));
  for (int i = 0; i < n * oh * ow * m; ++i) {
    y[i] = 0;
  }
  if (dmp_dv_mem_sync_end(output_mem)) {
    ERR("dmp_dv_mem_sync_end() failed for output: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  // Execute batched
  {
    long max_mem_kb = 0;
    double utime = 0, stime = 0;
    get_exec_stats(&max_mem_kb, &utime, &stime);
    LOG("Will create Batched command list: RSS=%d Mb CMA=%d Mb\n",
        (int)(max_mem_kb >> 10), get_used_cma_mb());

    cmdlist = dmp_dv_cmdlist_create(ctx);
    if (!cmdlist) {
      ERR("dmp_dv_cmdlist_create() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    struct dmp_dv_cmdraw_conv_v0 conf;
    memset(&conf, 0, sizeof(conf));
    conf.header.size = sizeof(conf);
    conf.header.device_type = DMP_DV_DEV_CONV;
    conf.header.version = 0;
    conf.topo = 1;
    conf.w = w;
    conf.h = h;
    conf.z = 1;
    conf.c = c;
    conf.input_buf.mem = input_mem;
    conf.output_buf.mem = output_mem;
    if (deconv) {
      conf.run[0].conv_enable = dw ? 7 : 5;
    }
    else {
      conf.run[0].conv_enable = dw ? 3 : 1;
    }
    conf.run[0].conv_stride = pack2(stride, stride);
    conf.run[0].conv_pad = pack4(pad, pad, pad, pad);
    conf.run[0].weight_buf.mem = weights_mem;
    conf.run[0].weight_buf.offs = 0;
    conf.run[0].m = m;
    conf.run[0].p = pack2(kx, ky);
    conf.run[0].pz = 1;
    conf.input_circular_offset = n;

    print_cmd(conf);

    if (dmp_dv_cmdlist_add_raw(cmdlist, (struct dmp_dv_cmdraw*)&conf)) {
      ERR("dmp_dv_cmdlist_add_raw() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    get_exec_stats(&max_mem_kb, &utime, &stime);
    LOG("Will commit Batched command list: RSS=%d Mb CMA=%d Mb\n",
        (int)(max_mem_kb >> 10), get_used_cma_mb());

    if (dmp_dv_cmdlist_commit(cmdlist)) {
      ERR("dmp_dv_cmdlist_commit() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    get_exec_stats(&max_mem_kb, &utime, &stime);
    LOG("Will execute Batched command list: RSS=%d Mb CMA=%d Mb\n",
        (int)(max_mem_kb >> 10), get_used_cma_mb());

    TimeInterval ti;
    int64_t exec_id = dmp_dv_cmdlist_exec(cmdlist);
    if (exec_id < 0) {
      ERR("dmp_dv_cmdlist_exec() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    if (dmp_dv_cmdlist_wait(cmdlist, exec_id)) {
      ERR("dmp_dv_cmdlist_wait() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    dt_batch = ti.get_ms();
    LOG("Batched completed in %.3f msec\n", dt_batch);

    dmp_dv_cmdlist_release(cmdlist);
    cmdlist = NULL;
  }

  if (dmp_dv_mem_sync_start(output_mem, 1, 0)) {
    ERR("dmp_dv_mem_sync_start() failed for output: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  {
    const int n_output = n * oh * ow * m;
    for (int i = 0; i < n_output; ++i) {
      if (y[i] != t[i]) {
        ERR("Batched != Unrolled at index %d (%.1f != %.1f)\n", i, y[i], t[i]);
        goto L_EXIT;
     }
    }
  }

  result = 0;

  L_EXIT:
  dmp_dv_cmdlist_release(cmdlist);
  dmp_dv_mem_release(weights_mem);
  dmp_dv_mem_release(output_mem);
  dmp_dv_mem_release(input_mem);
  dmp_dv_context_release(ctx);

  static int s_n_fd = -1;
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
    int num = 1;
    for (; *fnme; ++fnme) {
      if ((*fnme >= '0') && (*fnme <= '9')) {
        continue;
      }
      num = 0;
      break;
    }
    if (num) {
      ++n_fd;
    }
  }
  closedir(d);

  if (s_n_fd == -1) {
    s_n_fd = n_fd;
  }
  if (n_fd != s_n_fd) {
    ERR("Inconsistent file descriptor count detected, memory leak is probable\n");
    result = -1;
  }

  LOG("EXIT%s: test_batch: %d FDs\n", result ? "(FAILED)" : "", n_fd);
  return result;
}


int main(int argc, char **argv) {
  const char *s_seed = getenv("SEED");
  int seed = 0;
  if ((s_seed) && (s_seed[0])) {
    seed = atoi(s_seed);
  }
  if (seed == 0) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    seed = (int)ts.tv_nsec;
  }
  std::mt19937 prng(seed);

  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  LOG("Using seed: %d\n", seed);
  const char *s_repeat = getenv("REPEAT");
  int repeat = s_repeat ? atoi(s_repeat) : 1;
  for (int i = 0; i < repeat; ++i) {
    res = test_batch_mm(prng);
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
