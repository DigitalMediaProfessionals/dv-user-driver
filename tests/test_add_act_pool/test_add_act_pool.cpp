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
 * @brief Tests add+act+pool.
 */
#include <unistd.h>
#include <sys/mman.h>
#include <dirent.h>
#include <time.h>

#include <stdio.h>
#include <string.h>
#include <math.h>

#include <algorithm>

#include "dmp_dv.h"
#include "dmp_dv_cmdraw_v0.h"

#ifdef __x86_64__
#include "half.h"
typedef half_float::half __fp16;
#endif


/// @brief Offset within NWHC8 layout.
#define NWHC8_OFFS(n, h, w, c, i_n, i_h, i_w, i_c) \
    (((i_n) * ((w) * (h) * (c))) + /* sample beginning */ \
     ((((i_c) >> 3) << 3) * ((w) * (h))) + /* 8-channel beginning */ \
     (((i_w) * (h) + (i_h)) * std::min((c) - (((i_c) >> 3) << 3), 8)) + /* pixel start */ \
     ((i_c) & 7))


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)


#define ALIGN64(n) (((n) & 63) ? (n) + (64 - ((n) & 63)) : (n))


/* The state array must be initialized to not be all zero */
uint32_t xorshift128(uint32_t state[4]) {
    /* Algorithm "xor128" from p. 5 of Marsaglia, "Xorshift RNGs" */
    uint32_t s, t = state[3];
    t ^= t << 11;
    t ^= t >> 8;
    state[3] = state[2]; state[2] = state[1]; state[1] = s = state[0];
    t ^= s;
    t ^= s >> 19;
    state[0] = t;
    return t;
}


/// @brief Half floats used in test.
static const uint16_t valid_floats_u[256] = {
       51328, 48128, 17408, 17920, 17408, 16896, 50944, 15360, 18176,
       49152, 51328, 16896, 49664, 17408, 16896, 17408, 18560, 49664,
       17408, 17408, 49152, 50176, 18432, 17920, 49664, 49152, 50432,
       16384, 18176, 15360, 51328, 50176, 50432, 15360, 50176, 17920,
       18432, 18432, 50688, 49664, 50688, 15360, 49664, 50688, 48128,
       16384, 18176, 18560, 17408, 50688, 17664, 49664, 51328, 18176,
       50432, 18432, 16384, 50176, 49664, 17920, 18560, 50432, 17408,
       15360, 16896, 16384, 50688, 50688, 50176, 49152, 15360, 17408,
       16896, 16896, 17664, 16384, 15360, 17408, 18176, 51328, 17664,
       17664, 17408, 17664, 50944, 17408, 15360, 17664, 18560, 50944,
       50176, 17408, 51200, 17408, 48128, 16896, 18176, 50688, 50944,
       51328, 18560, 18560, 50176, 51200, 15360, 17664, 18560, 48128,
       17408, 50176, 18176, 48128, 49664, 17664, 49664, 16896, 16896,
       17408, 50944, 15360, 50176, 17920, 16896, 49152, 48128, 49152,
       50176, 18432, 18176, 16896, 49152, 50688, 48128, 17664, 16896,
       49664, 50944, 17920, 49664, 48128, 16384, 51200, 50944, 18432,
       17664, 50944, 51200, 48128, 18560, 50688, 17664, 49664, 50432,
       17920, 18176, 50176, 51200, 50944, 17920, 51200, 16896, 18432,
       16896, 18176, 50944, 17920, 50176, 51328, 51200, 15360, 51200,
       50432, 16384, 49152, 15360, 16896, 50944, 50432, 49152, 18176,
       50432, 49152, 49664, 49152, 17920, 16896, 18176, 50176, 49664,
       18560, 49152, 50432, 16896, 48128, 18176, 17408, 17408, 17664,
       50432, 16384, 51328, 16896, 17920, 17664, 16384, 50944, 48128,
       50432, 50944, 18560, 17920, 49152, 17920, 50176, 17920, 49152,
       16384, 18176, 15360, 17664, 18560, 50432, 16896, 18432, 50944,
       51200, 18432, 49152, 17408, 51328, 51328, 16896, 18560, 18560,
       51328, 50688, 16896, 17920, 50688, 50688, 17408, 50176, 49152,
       17920, 50944, 18560, 18176, 16384, 16896, 16384, 18176, 49152,
       17664, 49664, 18176, 17664
};  // [-9, ..., 9] except zero


volatile void *v_ptr = NULL;
int verbosity = 0;


/// @brief Prints command content for debugging.
void print_cmd(dmp_dv_cmdraw_conv_v0& cmd) {
  if (verbosity < 0) {
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


const float x0_preset[2][4][3] = {{{-1,  1, -1}, {-1,  1,  2}, {-2,  1,  2}, { 2,  3,  2}},
                                  {{ 1,  2,  1}, {-3, -1, -3}, { 2,  1, -2}, {-3,  3, -1}}};
const float x1_preset[2][4][3] = {{{-2,  1,  2}, {-1,  1,  3}, {-3,  2,  3}, {-3, -1, -2}},
                                  {{-1, -2,  3}, {-1,  3,  2}, {-2, -1,  1}, { 2,  3, -2}}};


int test_add_act_pool(uint32_t state[4]) {
  LOG("ENTER: test_add_act_pool\n");

  int result = -1;
  dmp_dv_context ctx = dmp_dv_context_create();
  dmp_dv_mem input_mem = NULL, output_mem = NULL, weights_mem = NULL;
  dmp_dv_cmdlist cmdlist = NULL;
  const char *s_w = getenv("W");
  const char *s_h = getenv("H");
  const char *s_c = getenv("C");
  const int w = s_w ? atoi(s_w) : 4;
  const int h = s_h ? atoi(s_h) : 2;
  const int c = s_c ? atoi(s_c) : 3;
  uint16_t *weights_ptr = NULL;
  __fp16 *x0, *x1, *y16;
  const char *s_no_add = getenv("NO_ADD");
  const char *s_no_relu = getenv("NO_RELU");
  const char *s_abs = getenv("ABS");
  const char *s_avg_pool = getenv("AVG_POOL");
  const char *s_conv = getenv("CONV");
  const char *s_no_pool = getenv("NO_POOL");
  const int do_add = s_no_add ? (atoi(s_no_add) > 0 ? 0 : 1) : 1;
  const int do_relu = s_no_relu ? (atoi(s_no_relu) > 0 ? 0 : 1) : 1;
  const int do_abs = s_abs ? (atoi(s_abs) > 0 ? 1 : 0) : 0;
  const int avg_pool = s_avg_pool ? (atoi(s_avg_pool) > 0 ? 1 : 0) : 0;
  const int do_conv = s_conv ? (atoi(s_conv) > 0 ? 1 : 0) : 0;
  const int do_pool = s_no_pool ? (atoi(s_no_pool) > 0 ? 0 : 1) : 1;
  const float threshold = 0.01f;
  size_t packed_weights_size = 0;
  const __fp16 *valid_floats = (const __fp16*)valid_floats_u;
  const char *s_verbosity = getenv("VERBOSITY");
  verbosity = s_verbosity ? atoi(s_verbosity) : 0;
  const char *s_batch = getenv("BATCH");
  int batch = s_batch ? atoi(s_batch) : 1;
  batch = batch < 1 ? 1 : batch;
  const int use_preset = ((!s_w) && (!s_h) && (!s_c) && (batch == 1)) ? 1 : 0;
  const char *s_out_offs = getenv("OUT_OFFS");
  int out_offs = s_out_offs ? atoi(s_out_offs) : 0;
  out_offs = out_offs < 0 ? 0 : out_offs;
  const int eltwise_base_offs = ALIGN64(batch * w * h * c * 2);
  const char *s_pool_size = getenv("POOL_SIZE");
  const int pool_size = s_pool_size ? atoi(s_pool_size) : 2;
  const char *s_pool_stride = getenv("POOL_STRIDE");
  const int pool_stride = s_pool_stride ? atoi(s_pool_stride) : 2;

  LOG("do_add=%d do_relu=%d do_abs=%d avg_pool=%d do_conv=%d do_pool=%d batch=%d out_offs=%d pool_size=%d pool_stride=%d\n",
      do_add, do_relu, do_abs, avg_pool, do_conv, do_pool, batch, out_offs, pool_size, pool_stride);

  if ((pool_size != 2) && (pool_size != 3)) {
    ERR("Only pool_size 2 and 3 is implemented\n");
    return 1;
  }

  if (!ctx) {
    ERR("dmp_dv_context_create() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully created context: %s\n", dmp_dv_context_get_info_string(ctx));

  input_mem = dmp_dv_mem_alloc(ctx, eltwise_base_offs * 2);
  if (!input_mem) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  output_mem = dmp_dv_mem_alloc(ctx, eltwise_base_offs + out_offs);
  if (!output_mem) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  if (do_conv) {
    const int n_channels = c;
    const int kx = 1;
    const int ky = 1;
    const int n_kernels = c;
    const uint16_t *quant_map = NULL;
    const void *weights = NULL;
    const uint16_t *bias = NULL;
    const uint16_t *prelu = NULL;
    uint8_t *packed_weights = NULL;
    dmp_dv_pack_conv_weights(
      n_channels, kx, ky, n_kernels,
      quant_map,
      weights, bias, prelu,
      packed_weights, &packed_weights_size);
    weights_mem = dmp_dv_mem_alloc(ctx, packed_weights_size);
    if (!weights_mem) {
      ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    weights_ptr = (uint16_t*)dmp_dv_mem_map(weights_mem);
    if (!weights_ptr) {
      ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
  }

  x0 = (__fp16*)dmp_dv_mem_map(input_mem);
  x1 = (__fp16*)(((uint8_t*)x0) + eltwise_base_offs);
  y16 = (__fp16*)(dmp_dv_mem_map(output_mem) + out_offs);
  if ((!x0) || (!x1) || ((size_t)y16 == (size_t)out_offs)) {
    ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  if ((dmp_dv_mem_sync_start(input_mem, 1, 1)) ||
      (dmp_dv_mem_sync_start(output_mem, 1, 1))) {
    ERR("dmp_dv_mem_sync_start() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  if (use_preset) {
    LOG("Filling input arrays with preset values\n");
    for (int c_start = 0; c_start < c; c_start += 8) {
      const int c_stop = c_start + 8 < c ? c_start + 8 : c;
      for (int i_w = 0, o_offs = 0; i_w < w; ++i_w) {
        for (int i_h = 0; i_h < h; ++i_h) {
          for (int i_c = c_start; i_c < c_stop; ++i_c, ++o_offs) {
            x0[o_offs] = (__fp16)x0_preset[i_h][i_w][i_c];
            x1[o_offs] = (__fp16)x1_preset[i_h][i_w][i_c];
          }
        }
      }
    }
    if (verbosity > 0) {
      LOG("x0(WHC8):");
      for (int i = 0; i < w * h * c; ++i) {
        LOG("%s%2.0f", (i ? ", " : " "), (float)x0[i]);
      }
      LOG("\n");
      LOG("x1(WHC8):");
      for (int i = 0; i < w * h * c; ++i) {
        LOG("%s%2.0f", (i ? ", " : " "), (float)x1[i]);
      }
      LOG("\n");
    }
  }
  else {
    for (int i = 0; i < (batch * w * h * c); ++i) {
      x0[i] = valid_floats[xorshift128(state) >> 24];
      x1[i] = valid_floats[xorshift128(state) >> 24];
      if (do_abs) {
        x0[i] = (__fp16)fabsf((float)x0[i]);
        x1[i] = (__fp16)fabsf((float)x1[i]);
      }
    }
  }
  if (!do_add) {
    memset(x1, 0, batch * w * h * c * 2);
  }
  memset(y16, 0xFF, batch * w * h * c * 2);  // set output to NaN

  v_ptr = x0;
  v_ptr = x1;
  v_ptr = y16;

  if ((dmp_dv_mem_sync_end(input_mem)) ||
      (dmp_dv_mem_sync_end(output_mem))) {
    ERR("dmp_dv_mem_sync_end() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  for (int i_conf = 0, confs_per_cmdlist = 1; i_conf < 100; i_conf += confs_per_cmdlist, confs_per_cmdlist = (xorshift128(state) & 15) + 1) {
    cmdlist = dmp_dv_cmdlist_create(ctx);
    if (!cmdlist) {
      ERR("dmp_dv_cmdlist_create() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    for (int j = 0; j < confs_per_cmdlist; ++j) {
      for (int i_batch = 0; i_batch < batch; ++i_batch) {
        if (do_conv) {
          const int n_channels = 1;
          const int kx = 1;
          const int ky = 1;
          const int n_kernels = c;
          const uint16_t *quant_map = NULL;
          uint16_t w16[c], b16[c];
          for (int i_c = 0; i_c < c; ++i_c) {
            w16[i_c] = 15360;
            b16[i_c] = 0;
          }
          const void *weights = w16;
          const uint16_t *bias = b16;
          const uint16_t *prelu = NULL;
          dmp_dv_pack_conv_weights(
            n_channels, kx, ky, n_kernels,
            quant_map,
            weights, bias, prelu,
            (uint8_t*)weights_ptr, &packed_weights_size);
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
        conf.input_buf.offs = i_batch * w * h * c * 2;
        conf.output_buf.mem = output_mem;
        conf.output_buf.offs = (do_pool ? i_batch * (w >> 1) * (h >> 1) * c * 2 : i_batch * w * h * c * 2) + out_offs;
        conf.eltwise_buf.mem = do_add ? input_mem : NULL;
        conf.eltwise_buf.offs = do_add ? eltwise_base_offs + conf.input_buf.offs : 0;
        conf.output_mode = do_add ? 1 : 0;
        if (do_conv) {
          conf.run[0].conv_enable = 3;  // depthwise dummy 1x1 conv
          conf.run[0].conv_stride = 0x0101;
          conf.run[0].weight_buf.mem = weights_mem;
          conf.run[0].weight_buf.offs = 0;
        }
        conf.run[0].m = c;
        conf.run[0].p = 0x0101;
        conf.run[0].pz = 1;
        if (do_pool) {
          conf.run[0].pool_enable = avg_pool ? 2 : 1;
          conf.run[0].pool_size = (uint16_t)pool_size | ((uint16_t)pool_size << 8);
          conf.run[0].pool_stride = (uint16_t)pool_stride | ((uint16_t)pool_stride << 8);
          conf.run[0].pool_avg_param = avg_pool ? 15360 : 0;
          const uint32_t pad_left = 0;
          const uint32_t w_apps = w / pool_stride + (w % pool_stride ? 1 : 0);
          const uint32_t pad_right = pool_stride * (w_apps - 1) + pool_size - w;
          const uint32_t pad_top = 0;
          const uint32_t h_apps = h / pool_stride + (h % pool_stride ? 1 : 0);
          const uint32_t pad_bottom = pool_stride * (h_apps - 1) + pool_size - h;
          conf.run[0].pool_pad = pad_left | (pad_right << 8) | (pad_top << 16) | (pad_bottom << 24);
        }
        conf.run[0].actfunc = do_relu ? 2 : 0;

        print_cmd(conf);

        if (dmp_dv_cmdlist_add_raw(cmdlist, (struct dmp_dv_cmdraw*)&conf)) {
          ERR("dmp_dv_cmdlist_add_raw() failed: %s\n", dmp_dv_get_last_error_message());
          goto L_EXIT;
        }
      }
      break;
    }

    if (dmp_dv_cmdlist_commit(cmdlist)) {
      ERR("dmp_dv_cmdlist_commit() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    int64_t exec_id = dmp_dv_cmdlist_exec(cmdlist);
    if (exec_id < 0) {
      ERR("dmp_dv_cmdlist_exec() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    if (dmp_dv_cmdlist_wait(cmdlist, exec_id)) {
      ERR("dmp_dv_cmdlist_wait() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    dmp_dv_cmdlist_release(cmdlist);
    cmdlist = NULL;

    // Check the last config for correctness
    if ((dmp_dv_mem_sync_start(input_mem, 1, 1)) ||
        (dmp_dv_mem_sync_start(output_mem, 1, 1))) {
      ERR("dmp_dv_mem_sync_start() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    float max_diff = 0;
    for (int i_batch = 0, i_offs = 0, o_offs = 0; i_batch < batch; ++i_batch) {

    for (int c_start = 0; c_start < c; c_start += 8) {
      const int c_stop = (c_start + 8 <= c) ? c_start + 8 : c;
      if (verbosity > 0) {
        LOG("\nc_start=%d c_stop=%d c=%d\n\n", c_start, c_stop, c);
      }
      if (do_pool) {
        const int w_out = (w / pool_stride) + (w % pool_stride ? 1 : 0);
        const int h_out = (h / pool_stride) + (h % pool_stride ? 1 : 0);
        for (int i_w = 0; i_w < w_out; ++i_w) {
          for (int i_h = 0; i_h < h_out; ++i_h) {
            for (int i_c = c_start; i_c < c_stop; ++i_c) {
              // Read pool window
              float window[2][16];
              bool window_valid[16];
              const float border_value = avg_pool ? 0 : -1.0e6f;
              float acc = border_value;
              for (int i_w_src = i_w * pool_stride, wnd_offs = 0; i_w_src < i_w * pool_stride + pool_size; ++i_w_src) {
                for (int i_h_src = i_h * pool_stride; i_h_src < i_h * pool_stride + pool_size; ++i_h_src, ++wnd_offs) {
                  const int offs = NWHC8_OFFS(batch, w, h, c, i_batch, i_w_src, i_h_src, i_c);
                  window[0][wnd_offs] = ((i_w_src < w) && (i_h_src < h)) ? x0[offs] : border_value;
                  window_valid[wnd_offs] = (i_w_src < w) && (i_h_src < h);
                  float vle = window[0][wnd_offs];
                  if (do_add) {
                    window[1][wnd_offs] = ((i_w_src < w) && (i_h_src < h)) ? x1[offs] : border_value;
                    vle += window[1][wnd_offs];
                  }
                  if (do_relu) {
                    vle = fmaxf(vle, 0.0f);
                  }
                  if (do_abs) {
                    vle = fabsf(vle);
                  }
                  if (avg_pool) {
                    acc += vle * (1.0f / (pool_size * pool_size));
                  }
                  else {
                    acc = fmaxf(acc, vle);
                  }
                }
              }
              const float t = acc;  // target
              const float y = y16[NWHC8_OFFS(batch, w_out, h_out, c, i_batch, i_w, i_h, i_c)];  // FPGA output

              const float diff = fabsf(y - t);
              max_diff = fmaxf(max_diff, isnanf(diff) ? 1000.0f : diff);
              if (verbosity > 0) {
                LOG("h=%d w=%d c=%2d y=%3.0f t=%3.0f d=%6.3f%s",
                    i_h, i_w, i_c, y, t, diff, (isnanf(diff) ? 1000.0f : diff) > threshold ? " ERR " : "  OK ");
                if (verbosity > 1) {
                  LOG(" x0=[");
                  for (int i = 0; i < pool_size * pool_size; ++i) {
                    if (window_valid[i]) {
                      LOG("%s%2.0f", i ? ", " : "", window[0][i]);
                    }
                    else {
                      LOG("%s ~", i ? ", " : "");
                    }
                  }
                  LOG("]");
                  if (do_add) {
                    LOG(" x1=[");
                    for (int i = 0; i < pool_size * pool_size; ++i) {
                      if (window_valid[i]) {
                        LOG("%s%2.0f", i ? ", " : "", window[1][i]);
                      }
                      else {
                        LOG("%s ~", i ? ", " : "");
                      }
                    }
                    LOG("]");
                  }
                }
                LOG("\n");
              }
            }
          }
        }
      }
      else {
        for (int i_w = 0; i_w < w; ++i_w) {
          for (int i_h = 0; i_h < h; ++i_h) {
            for (int i_c = c_start; i_c < c_stop; ++i_c, ++o_offs, ++i_offs) {
              const float vle = (float)y16[o_offs];
              float v0 = (float)x0[i_offs] + (float)x1[i_offs];

              float m0 = do_relu ? fmaxf(v0, 0) : v0;

              const float diff = fabsf(vle - m0);
              max_diff = fmaxf(max_diff, isnanf(diff) ? 1000.0f : diff);
              if (verbosity > 0) {
                LOG("h=%d w=%d c=%2d x0=%2.0f x1=%2.0f y=%.3f t=%.3f d=%.3f%s\n",
                    i_h, i_w, i_c, x0[i_offs], x1[i_offs], vle, m0, diff, (isnanf(diff) ? 1000.0f : diff) > threshold ? " ERR" : "");
              }
            }
          }
        }
      }
    }

    }  // batch loop

    if (max_diff > threshold) {
      goto L_EXIT;
    }

    memset(y16, 0xFF, eltwise_base_offs);  // set output to nan for the next test

    v_ptr = x0;
    v_ptr = x1;
    v_ptr = y16;

    if ((dmp_dv_mem_sync_end(input_mem)) ||
        (dmp_dv_mem_sync_end(output_mem))) {
      ERR("dmp_dv_mem_sync_end() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    break;
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

  LOG("EXIT%s: test_add_act_pool: %d FDs\n", result ? "(FAILED)" : "", n_fd);
  return result;
}


int main(int argc, char **argv) {
  const char *s_seed = getenv("SEED");
  uint64_t seed = 0;
  if ((s_seed) && (s_seed[0])) {
    seed = (uint64_t)atoll(s_seed);
  }
  if (seed == 0) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    seed = ((uint64_t)ts.tv_sec << 29) ^ (uint64_t)ts.tv_nsec;
  }

  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  uint32_t state[4] = {(uint32_t)(seed & 0xFFF), (uint32_t)((seed >> 12) & 0xFFF), (uint32_t)((seed >> 24) & 0xFFF), (uint32_t)((seed >> 36) & 0xFFF)};
  LOG("Using seed: [%u, %u, %u, %u]\n", (unsigned)state[0], (unsigned)state[1], (unsigned)state[2], (unsigned)state[3]);
  const char *s_repeat = getenv("REPEAT");
  int repeat = s_repeat ? atoi(s_repeat) : 1;
  for (int i = 0; i < repeat; ++i) {
    res = test_add_act_pool(state);
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
