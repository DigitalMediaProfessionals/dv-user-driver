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
 * @brief Test upsampling layer.
 */
#include <unistd.h>
#include <sys/mman.h>
#include <dirent.h>
#include <time.h>

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "dmp_dv.h"
#include "dmp_dv_cmdraw_v0.h"

#ifdef __x86_64__
#include "half.h"
typedef half_float::half __fp16;
#endif



#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)



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


/// @brief Half floats used in test (uniform in [-1, 1]).
static const uint16_t valid_floats_u[256] = {
       13330, 14745, 46058, 46868, 12012, 14870, 14660, 14730, 14361,
       15021, 44493, 15172,  8256, 13678, 47179, 11284, 47678, 12933,
       14245, 47536, 15318, 45116, 48088, 13020, 47217, 14955, 13132,
       13783, 12044, 46757, 46384, 46793, 45467, 14077, 12699, 47258,
       14891, 47287, 12427, 13731, 44173, 47793, 15118, 14612, 48098,
       12998, 12498, 46399, 15218, 47490, 15169, 47814, 47462, 47954,
       14536, 43071, 47329, 45182, 13498, 13420, 47989, 46681, 47569,
       14774, 46235, 44882, 47731, 47989, 14230, 14925, 14472, 46064,
       46377, 13604, 15252, 45499, 12971, 46490, 13791, 13545, 47893,
       13681, 14655, 11266, 12543, 11596, 47521, 44529, 46757, 46597,
       47859, 47155, 15118, 45743, 15104, 14534, 47610, 14963, 46366,
       15080, 14566, 13670, 15073, 12528, 45529, 45303, 46647, 15124,
       13896, 15328, 47098, 46274, 14317, 46743, 45261, 13458, 12603,
       15128, 15009, 15332, 47314, 11199, 14354, 46222, 46760, 13787,
       47761, 42951, 12790, 47618, 15233, 47493, 41865, 12192, 14982,
       45123, 45378, 15178, 14235, 47393, 15161, 14906, 13749, 47215,
       15283, 14391, 46227, 14958, 15271, 10967, 14451, 13291, 13368,
       15051, 48115, 48050, 15309, 15226, 12757, 12701, 46341, 14426,
       14511, 14907, 47390, 12790, 14899, 14194, 14812, 14823, 14478,
       13289, 11131, 12555, 14087, 14585, 14933, 47289, 15171, 14168,
       47831, 47700, 46176, 47406, 14994, 44573, 46676, 15172, 13628,
       47867, 47277, 14793, 15266, 47220, 14923, 14977, 14997, 13464,
       47545, 47495, 47276, 47645, 46115, 12382, 47691, 14565, 46783,
       14529, 14569, 47367, 43776, 47795, 47253, 47471, 47431, 48069,
       42683, 15289, 46553, 47658, 44959, 47156, 48106, 14811, 47181,
       13879, 14300, 47923, 48009, 14225, 15052, 15199, 47460, 46476,
       13511, 45752, 14172, 46667, 47255, 46515, 47804, 47194, 13758,
       13393, 45545, 14531, 47694, 15356, 14966, 47259, 14167, 46319,
       44035, 42651, 47635, 12978
};


volatile void *v_ptr = NULL;


/// @brief Prints command content for debugging.
void print_cmd(dmp_dv_cmdraw_conv_v0& cmd) {
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


int test_add_act_pool(uint32_t state[4]) {
  LOG("ENTER: test_add_act_pool\n");

  int result = -1;
  dmp_dv_context ctx = dmp_dv_context_create();
  dmp_dv_mem input_mem = NULL, eltwise_mem = NULL, output_mem = NULL, weights_mem = NULL;
  dmp_dv_cmdlist cmdlist = NULL;
  const int w = 8, h = 8, c = 1;
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
  const int max_input_bytes = w * h * c * 2;
  const __fp16 *valid_floats = (const __fp16*)valid_floats_u;

  LOG("do_add=%d do_relu=%d do_abs=%d avg_pool=%d do_conv=%d do_pool=%d\n",
      do_add, do_relu, do_abs, avg_pool, do_conv, do_pool);

  if (!ctx) {
    ERR("dmp_dv_context_create() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully created context: %s\n", dmp_dv_context_get_info_string(ctx));

  input_mem = dmp_dv_mem_alloc(ctx, max_input_bytes);
  if (!input_mem) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  eltwise_mem = dmp_dv_mem_alloc(ctx, max_input_bytes);
  if (!eltwise_mem) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  output_mem = dmp_dv_mem_alloc(ctx, max_input_bytes);
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
  x1 = (__fp16*)dmp_dv_mem_map(eltwise_mem);
  y16 = (__fp16*)dmp_dv_mem_map(output_mem);
  if ((!x0) || (!x1) || (!y16)) {
    ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  if ((dmp_dv_mem_sync_start(input_mem, 1, 1)) ||
      (dmp_dv_mem_sync_start(eltwise_mem, 1, 1)) ||
      (dmp_dv_mem_sync_start(output_mem, 1, 1))) {
    ERR("dmp_dv_mem_sync_start() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  for (int i = 0; i < (max_input_bytes >> 1); ++i) {
    x0[i] = valid_floats[xorshift128(state) >> 24];
    x1[i] = valid_floats[xorshift128(state) >> 24];
    if (do_abs) {
      x0[i] = (__fp16)fabsf((float)x0[i]);
      x1[i] = (__fp16)fabsf((float)x1[i]);
    }
  }
  if (!do_add) {
    memset(x1, 0, max_input_bytes);
  }
  memset(y16, 0xFF, max_input_bytes);  // set output to nan
  
  v_ptr = x0;
  v_ptr = x1;
  v_ptr = y16;
  
  if ((dmp_dv_mem_sync_end(input_mem)) ||
      (dmp_dv_mem_sync_end(eltwise_mem)) ||
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
      conf.input_buf.offs = 0;
      conf.output_buf.mem = output_mem;
      conf.output_buf.offs = 0;
      conf.eltwise_buf.mem = do_add ? eltwise_mem : NULL;
      conf.eltwise_buf.offs = 0;
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
        conf.run[0].pool_size = 0x0202;
        conf.run[0].pool_stride = 0x0202;
        conf.run[0].pool_avg_param = avg_pool ? 15360 : 0;
      }
      conf.run[0].actfunc = do_relu ? 2 : 0;

      print_cmd(conf);

      if (dmp_dv_cmdlist_add_raw(cmdlist, (struct dmp_dv_cmdraw*)&conf)) {
        ERR("dmp_dv_cmdlist_add_raw() failed: %s\n", dmp_dv_get_last_error_message());
        goto L_EXIT;
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
        (dmp_dv_mem_sync_start(eltwise_mem, 1, 1)) ||
        (dmp_dv_mem_sync_start(output_mem, 1, 1))) {
      ERR("dmp_dv_mem_sync_start() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    float max_diff = 0;
    for (int c_start = 0, o_offs = 0, i_offs = 0; c_start < c; c_start += 8) {
      const int c_stop = (c_start + 8 <= c) ? c_start + 8 : c;
      if (do_pool) {
        for (int i_w = 0; i_w < (w >> 1); ++i_w, i_offs += (c_stop - c_start) * h) {
          for (int i_h = 0; i_h < (h >> 1); ++i_h, i_offs += (c_stop - c_start)) {
            for (int i_c = c_start; i_c < c_stop; ++i_c, ++o_offs, ++i_offs) {
              const float vle = (float)y16[o_offs];
              float v0, v1, v2, v3;
              float x00[4], x11[4];
              
              int offs = i_offs;
              x00[0] = (float)x0[offs]; x11[0] = (float)x1[offs];
              v0 = avg_pool ? x00[0] : x00[0] + x11[0];
              
              offs = i_offs + (c_stop - c_start);
              x00[1] = (float)x0[offs]; x11[1] = (float)x1[offs];
              v1 = avg_pool ? x00[1] : x00[1] + x11[1];

              offs = i_offs + h * (c_stop - c_start);
              x00[2] = (float)x0[offs]; x11[2] = (float)x1[offs];
              v2 = avg_pool ? x00[2] : x00[2] + x11[2];
              
              offs = i_offs + h * (c_stop - c_start) + (c_stop - c_start);
              x00[3] = (float)x0[offs]; x11[3] = (float)x1[offs];
              v3 = avg_pool ? x00[3] : x00[3] + x11[3];
              
              float m0;
              if (avg_pool) {
                v0 = (v0 + v1 + v2 + v3) + (float)x1[o_offs];
                m0 = do_relu ? fmaxf(v0, 0) : v0;
              }
              else {
                float v0a = do_relu ? fmaxf(v0, 0) : v0,
                      v1a = do_relu ? fmaxf(v1, 0) : v1,
                      v2a = do_relu ? fmaxf(v2, 0) : v2,
                      v3a = do_relu ? fmaxf(v3, 0) : v3;
              
                 m0 = fmaxf(fmaxf(v0a, v1a), fmaxf(v2a, v3a));
              }
              
              const float diff = fabsf(vle - m0);
              max_diff = fmaxf(max_diff, isnanf(diff) ? 1000.0f : diff);
              LOG("h=%d w=%d c=%d i_offs=%d x0=[%.3f, %.3f, %.3f, %.3f] x1=[%.3f, %.3f, %.3f, %.3f] y=%.3f t=%.3f d=%.3f%s\n",
                  i_h, i_w, i_c, i_offs, x00[0], x00[1], x00[2], x00[3], x11[0], x11[1], x11[2], x11[3], vle, m0, diff, diff > threshold ? " ERR" : "");
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
              LOG("h=%d w=%d c=%d x0=%.3f x1=%.3f y=%.3f t=%.3f d=%.3f%s\n",
                  i_h, i_w, i_c, x0[i_offs], x1[i_offs], vle, m0, diff, diff > threshold ? " ERR" : "");
            }
          }
        }
      }
      if (max_diff > threshold) {
        goto L_EXIT;
      }
    }

    memset(y16, 0xFF, max_input_bytes);  // set output to nan for the next test

    v_ptr = x0;
    v_ptr = x1;
    v_ptr = y16;

    if ((dmp_dv_mem_sync_end(input_mem)) ||
        (dmp_dv_mem_sync_end(eltwise_mem)) ||
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
  dmp_dv_mem_release(eltwise_mem);
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
  uint64_t seed;
  if (s_seed) {
    seed = (uint64_t)atoll(s_seed);
  }
  else {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    seed = ((uint64_t)ts.tv_sec << 29) ^ (uint64_t)ts.tv_nsec;
  }

  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  uint32_t state[4] = {(uint32_t)(seed & 0xFFF), (uint32_t)((seed >> 12) & 0xFFF), (uint32_t)((seed >> 24) & 0xFFF), (uint32_t)((seed >> 36) & 0xFFF)};
  LOG("Using seed: [%u, %u, %u, %u]\n", (unsigned)state[0], (unsigned)state[1], (unsigned)state[2], (unsigned)state[3]);
  for (int i = 0; i < 1; ++i) {
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
