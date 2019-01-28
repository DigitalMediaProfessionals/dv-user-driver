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

#include <stdio.h>
#include <string.h>

#include "dmp_dv.h"
#include "dmp_dv_cmdraw_v0.h"


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
static const uint16_t valid_floats[256] = {
    0, 14249, 13806, 47192, 14461, 12825, 14256, 15260, 47742,
    14349, 14862, 14781, 11943, 48047, 44506, 10491, 12801, 44023,
    15000, 11521, 37940, 47775, 47844, 13322, 12841, 48012, 46678,
    47158, 10691, 15296, 45887, 44346, 46028, 43918, 47876, 45657,
    15294, 15265, 14684, 15337, 44426, 47338, 47941, 41546, 47891,
    15086, 13759, 47929, 15331, 47152, 47067, 14598, 46890,  9515,
    14989, 15181, 47345, 47567, 14310, 14702, 46163, 47710, 15177,
    14769, 44121, 10401, 45249, 14446, 15149, 15338, 12361, 47419,
    46509, 15317, 14530, 14534, 13729, 44317, 14663, 15354, 47400,
    44544, 48004, 46658, 46946, 15129, 44006, 14257, 10093, 47363,
    48075, 47713, 12068, 13237, 47512, 15215, 45544, 47685, 12603,
    14876, 42069, 47286, 47629, 46211, 14600, 46347, 14621, 14570,
    46489, 12440, 13645, 14558, 13349, 13619, 47359, 15318, 47981,
    44117, 47162, 13673, 44761, 47630, 47743, 15007, 47686, 47755,
    44436, 47909, 13723, 14103, 14321, 46936, 45528, 14375, 14377,
    12445, 47132, 42341, 14693, 46193, 14717, 14547, 47847, 46309,
    45088, 15270, 42764, 47601, 48063, 46709, 11819, 44506, 47612,
    14047, 47579, 10633, 14996, 13390, 47361, 14479, 14233, 47148,
    14372, 47875, 47505, 47532, 15166, 14597, 46819, 47288, 10735,
    13007, 40891, 37194, 13637, 48072, 47204, 47983, 47299, 13286,
    47590, 47761, 46093, 46572, 47246, 47480, 14362, 47181, 47687,
    12599, 15036, 47269, 46527, 13677, 48112, 11607, 13685, 47200,
    44771, 46303, 15176, 46612, 15269, 45363, 15155, 47039, 46750,
    13870, 14534, 15087, 14966, 12323, 47154, 14496, 47561, 47308,
    45809, 47602, 15096, 14784, 15024, 14515, 13411, 12563, 46854,
    48021, 13754, 45794, 47789, 13626, 47205, 14117, 14300, 45514,
    46410, 47210, 12741, 47218, 46168,  6839, 11508, 46528, 14784,
    47346, 46640, 14373, 47607, 13478, 13922, 45830, 13773, 13734,
    12359, 13764, 14442, 13234
};


int test_upsampling(uint32_t state[4]) {
  LOG("ENTER: test_upsampling\n");

  int result = -1;
  dmp_dv_context ctx = dmp_dv_context_create();
  dmp_dv_mem input = NULL, output = NULL;
  dmp_dv_cmdlist cmdlist = NULL;
  int w, h, c, i_offs, o_offs;
  uint16_t *input_ptr, *output_ptr;

  if (!ctx) {
    ERR("dmp_dv_context_create() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully created context: %s\n", dmp_dv_context_get_info_string(ctx));

  input = dmp_dv_mem_alloc(ctx, 128 * 128 * 512 * 2 + 4096);
  if (!input) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  output = dmp_dv_mem_alloc(ctx, 256 * 256 * 512 * 2 + 4096);
  if (!output) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  input_ptr = (uint16_t*)dmp_dv_mem_map(input);
  output_ptr = (uint16_t*)dmp_dv_mem_map(output);
  if ((!input_ptr) || (!output_ptr)) {
    ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  if (dmp_dv_mem_sync_start(input, 0, 1)) {
    ERR("dmp_dv_mem_sync_start() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  int n = dmp_dv_mem_get_size(input) >> 1;
  for (int i = 0; i < n; ++i) {
    input_ptr[i] = valid_floats[xorshift128(state) >> 24];
  }
  if (dmp_dv_mem_sync_end(input)) {
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
      w = (xorshift128(state) & 127) + 1;
      h = (xorshift128(state) & 127) + 1;
      c = (xorshift128(state) & 255) + 1;
      i_offs = 0;//((xorshift128(state) & 4095) >> 4) << 4;
      o_offs = 0;//((xorshift128(state) & 4095) >> 4) << 4;

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
      conf.input_buf.mem = input;
      conf.input_buf.offs = i_offs;
      conf.output_buf.mem = output;
      conf.output_buf.offs = o_offs;
      conf.run[0].m = c;
      conf.run[0].p = 0x0101;
      conf.run[0].pz = 1;
      conf.run[0].conv_stride = 0x0101;
      conf.run[0].pool_enable = 4;
      conf.run[0].pool_size = 0x0202;
      conf.run[0].pool_stride = 0x0101;

      if (dmp_dv_cmdlist_add_raw(cmdlist, (struct dmp_dv_cmdraw*)&conf)) {
        ERR("dmp_dv_cmdlist_add_raw() failed: %s\n", dmp_dv_get_last_error_message());
        goto L_EXIT;
      }
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

    if (dmp_dv_mem_sync_start(output, 1, 1)) {
      ERR("dmp_dv_mem_sync_start() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    for (int c_start = 0, i_output = o_offs; c_start < c; c_start += 8) {
      int chan_chunk_size = c - c_start;
      if (chan_chunk_size > 8) {
        chan_chunk_size = 8;
      }
      for (int x = 0; x < w * 2; ++x) {
        for (int y = 0; y < h * 2; ++y) {
          for (int chan = 0; chan < chan_chunk_size; ++chan, ++i_output) {
            const uint16_t o_vle = output_ptr[i_output];
            const uint16_t i_vle = input_ptr[i_offs + w * h * c_start + (x >> 1) * h * chan_chunk_size + (y >> 1) * chan_chunk_size + chan];
            if (i_vle != o_vle) {
              ERR("%hu != %hu for output (x, y, c) = (%d, %d, %d)\n", i_vle, o_vle, x, y, c_start + chan);
              goto L_EXIT;
            }
          }
        }
      }
    }

    memset(output_ptr, 0, dmp_dv_mem_get_size(output));  // clear output for the next test

    if (dmp_dv_mem_sync_end(output)) {
      ERR("dmp_dv_mem_sync_end() failed: %s\n", dmp_dv_get_last_error_message());
      return -1;
    }
  }

  result = 0;

  L_EXIT:
  dmp_dv_cmdlist_release(cmdlist);

  dmp_dv_mem_release(output);
  dmp_dv_mem_release(input);
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

  LOG("EXIT%s: test_upsampling: %d FDs\n", result ? "(FAILED)" : "", n_fd);
  return result;
}


int main(int argc, char **argv) {
  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  uint32_t state[4] = {1, 2, 3, 4};
  for (int i = 0; i < 3; ++i) {
    res = test_upsampling(state);
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
