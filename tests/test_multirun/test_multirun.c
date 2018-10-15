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
 * @brief Test that multirun configuration does not hang.
 */
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
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


static int fill_mem(dmp_dv_mem mem, uint32_t state[4]) {
  uint16_t *ptr = (uint16_t*)dmp_dv_mem_map(mem);
  if (!ptr) {
    ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
    return -1;
  }

  if (dmp_dv_mem_sync_start(mem, 0, 1)) {
    ERR("dmp_dv_mem_sync_start() failed: %s\n", dmp_dv_get_last_error_message());
    return -1;
  }

  int n = dmp_dv_mem_get_size(mem) >> 1;
  for (int i = 0; i < n; ++i) {
    ptr[i] = valid_floats[xorshift128(state) >> 24];
  }

  if (dmp_dv_mem_sync_end(mem)) {
    ERR("dmp_dv_mem_sync_end() failed: %s\n", dmp_dv_get_last_error_message());
    return -1;
  }

  return 0;
}


int test_multirun() {
  LOG("ENTER: test_multirun\n");

  int result = -1;
  dmp_dv_context ctx = dmp_dv_context_create();
  dmp_dv_mem weights_mem = NULL, io_mem = NULL;
  dmp_dv_cmdlist cmdlist = NULL;
  struct dmp_dv_cmdraw_conv_v0 conf;
  int64_t exec_id;
  uint32_t state[4] = {1, 2, 3, 4};

  if (!ctx) {
    ERR("dmp_dv_context_create() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully created context: %s\n", dmp_dv_context_get_info_string(ctx));

  weights_mem = dmp_dv_mem_alloc(ctx, 8 * 1024 * 1024);
  io_mem = dmp_dv_mem_alloc(ctx, 8 * 1024 * 1024);
  if ((!weights_mem) || (!io_mem)) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  if ((fill_mem(weights_mem, state)) || (fill_mem(io_mem, state))) {
    goto L_EXIT;
  }

  memset(&conf, 0, sizeof(conf));
  conf.header.size = sizeof(conf);
  conf.header.device_type = DMP_DV_DEV_CONV;
  conf.header.version = 0;
  // Topo: 00000000000000000000000001010101
  conf.topo = 0x55;  // [31:0] Output Destination of each run, 0 = UBUF, 1 = EXTMEM

  // Input Configuration:
  conf.w = 28;  // Input Width
  conf.h = 28;  // Input Height
  conf.z = 1;  // Input Depth
  conf.c = 256;  // Input Channels
  conf.input_buf.mem = io_mem;
  conf.input_buf.offs = 0;

  // Output Configuration:
  conf.output_buf.mem = io_mem;
  conf.output_buf.offs = 0;

  conf.eltwise_buf.mem = NULL;
  conf.eltwise_buf.offs = 0;  // Input byte address for elementwise add (0 = UBUF Input Buffer)
  conf.output_mode = 0;  // 0 = concat, 1 = eltwise add

  // Runs Configuration:
  // ->7 run(s)
  //--------------------------------------------------
  //RUN : 0
  //--------------------------------------------------
  //->: inception_3b/1x1
  //->: inception_3b/relu_1x1
  //->: pool3/3x3_s2
  conf.run[0].m = 128;  // Output Channels
  conf.run[0].conv_enable = 1;  // 1 = Enabled, 0 = Disabled
  conf.run[0].p = 1;  // Filter Width and Height
  conf.run[0].pz = 1;  // Filter Depth
  conf.run[0].weight_buf.mem = weights_mem;
  conf.run[0].weight_buf.offs = 0;
  conf.run[0].weight_fmt = 3;  // Weight format (0 = random access blocks, 1 = compact stream, 3 = 8-bit qunatized stream)
  conf.run[0].conv_pad = 0x0;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[0].conv_stride = 0x101;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[0].conv_dilation = 0x0;  // bits [7:0] = X dilation, bits [15:8] = Y dilation
  conf.run[0].pool_enable = 1;  // 0 = disabled, 1 = max pooling, 2 = average pooling
  conf.run[0].pool_size = 0x303;  // bits [7:0] = width, bits [15:8] = height
  conf.run[0].pool_stride = 0x202;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[0].pool_pad = 0x1010101;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[0].pool_avg_param = 0x0;  // Must be set to 1/pool_size^2 in FP16 format when using average pooling (average pooling assumes square size)
  conf.run[0].actfunc = 2;  // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6
  conf.run[0].actfunc_param = 0x0;  // Leaky ReLU parameter (NOTE: 0x2E66 is 0.1 in FP16)
  conf.run[0].rectifi_en = 0;  // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)
  conf.run[0].lrn = 0x0;  // [0] : 1 = LRN enable, 0 = LRN disable, [1] : 1 = incl. power func, 0 = excl., [8:11] = x^2 scale factor log2
  //--------------------------------------------------
  //RUN : 1
  //--------------------------------------------------
  //->: inception_3b/3x3_reduce
  //->: inception_3b/relu_3x3_reduce
  conf.run[1].m = 128;  // Output Channels
  conf.run[1].conv_enable = 1;  // 1 = Enabled, 0 = Disabled
  conf.run[1].p = 1;  // Filter Width and Height
  conf.run[1].pz = 1;  // Filter Depth
  conf.run[1].weight_buf.mem = weights_mem;
  conf.run[1].weight_buf.offs = 0;
  conf.run[1].weight_fmt = 3;  // Weight format (0 = random access blocks, 1 = compact stream, 3 = 8-bit qunatized stream)
  conf.run[1].conv_pad = 0x0;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[1].conv_stride = 0x101;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[1].conv_dilation = 0x0;  // bits [7:0] = X dilation, bits [15:8] = Y dilation
  conf.run[1].pool_enable = 0;  // 0 = disabled, 1 = max pooling, 2 = average pooling
  conf.run[1].pool_size = 0x0;  // bits [7:0] = width, bits [15:8] = height
  conf.run[1].pool_stride = 0x101;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[1].pool_pad = 0x0;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[1].pool_avg_param = 0x0;  // Must be set to 1/pool_size^2 in FP16 format when using average pooling (average pooling assumes square size)
  conf.run[1].actfunc = 2;  // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6
  conf.run[1].actfunc_param = 0x0;  // Leaky ReLU parameter (NOTE: 0x2E66 is 0.1 in FP16)
  conf.run[1].rectifi_en = 0;  // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)
  conf.run[1].lrn = 0x0;  // [0] : 1 = LRN enable, 0 = LRN disable, [1] : 1 = incl. power func, 0 = excl., [8:11] = x^2 scale factor log2
  //--------------------------------------------------
  //RUN : 2
  //--------------------------------------------------
  //->: inception_3b/3x3
  //->: inception_3b/relu_3x3
  //->: pool3/3x3_s2
  conf.run[2].m = 192;  // Output Channels
  conf.run[2].conv_enable = 1;  // 1 = Enabled, 0 = Disabled
  conf.run[2].p = 3;  // Filter Width and Height
  conf.run[2].pz = 1;  // Filter Depth
  conf.run[2].weight_buf.mem = weights_mem;
  conf.run[2].weight_buf.offs = 0;
  conf.run[2].weight_fmt = 3;  // Weight format (0 = random access blocks, 1 = compact stream, 3 = 8-bit qunatized stream)
  conf.run[2].conv_pad = 0x1010101;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[2].conv_stride = 0x101;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[2].conv_dilation = 0x0;  // bits [7:0] = X dilation, bits [15:8] = Y dilation
  conf.run[2].pool_enable = 1;  // 0 = disabled, 1 = max pooling, 2 = average pooling
  conf.run[2].pool_size = 0x303;  // bits [7:0] = width, bits [15:8] = height
  conf.run[2].pool_stride = 0x202;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[2].pool_pad = 0x1010101;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[2].pool_avg_param = 0x0;  // Must be set to 1/pool_size^2 in FP16 format when using average pooling (average pooling assumes square size)
  conf.run[2].actfunc = 2;  // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6
  conf.run[2].actfunc_param = 0x0;  // Leaky ReLU parameter (NOTE: 0x2E66 is 0.1 in FP16)
  conf.run[2].rectifi_en = 0;  // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)
  conf.run[2].lrn = 0x0;  // [0] : 1 = LRN enable, 0 = LRN disable, [1] : 1 = incl. power func, 0 = excl., [8:11] = x^2 scale factor log2
  //--------------------------------------------------
  //RUN : 3
  //--------------------------------------------------
  //->: inception_3b/5x5_reduce
  //->: inception_3b/relu_5x5_reduce
  conf.run[3].m = 32;  // Output Channels
  conf.run[3].conv_enable = 1;  // 1 = Enabled, 0 = Disabled
  conf.run[3].p = 1;  // Filter Width and Height
  conf.run[3].pz = 1;  // Filter Depth
  conf.run[3].weight_buf.mem = weights_mem;
  conf.run[3].weight_buf.offs = 0;
  conf.run[3].weight_fmt = 3;  // Weight format (0 = random access blocks, 1 = compact stream, 3 = 8-bit qunatized stream)
  conf.run[3].conv_pad = 0x0;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[3].conv_stride = 0x101;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[3].conv_dilation = 0x0;  // bits [7:0] = X dilation, bits [15:8] = Y dilation
  conf.run[3].pool_enable = 0;  // 0 = disabled, 1 = max pooling, 2 = average pooling
  conf.run[3].pool_size = 0x0;  // bits [7:0] = width, bits [15:8] = height
  conf.run[3].pool_stride = 0x101;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[3].pool_pad = 0x0;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[3].pool_avg_param = 0x0;  // Must be set to 1/pool_size^2 in FP16 format when using average pooling (average pooling assumes square size)
  conf.run[3].actfunc = 2;  // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6
  conf.run[3].actfunc_param = 0x0;  // Leaky ReLU parameter (NOTE: 0x2E66 is 0.1 in FP16)
  conf.run[3].rectifi_en = 0;  // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)
  conf.run[3].lrn = 0x0;  // [0] : 1 = LRN enable, 0 = LRN disable, [1] : 1 = incl. power func, 0 = excl., [8:11] = x^2 scale factor log2
  //--------------------------------------------------
  //RUN : 4
  //--------------------------------------------------
  //->: inception_3b/5x5
  //->: inception_3b/relu_5x5
  //->: pool3/3x3_s2
  conf.run[4].m = 96;  // Output Channels
  conf.run[4].conv_enable = 1;  // 1 = Enabled, 0 = Disabled
  conf.run[4].p = 5;  // Filter Width and Height
  conf.run[4].pz = 1;  // Filter Depth
  conf.run[4].weight_buf.mem = weights_mem;
  conf.run[4].weight_buf.offs = 0;
  conf.run[4].weight_fmt = 3;  // Weight format (0 = random access blocks, 1 = compact stream, 3 = 8-bit qunatized stream)
  conf.run[4].conv_pad = 0x2020202;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[4].conv_stride = 0x101;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[4].conv_dilation = 0x0;  // bits [7:0] = X dilation, bits [15:8] = Y dilation
  conf.run[4].pool_enable = 1;  // 0 = disabled, 1 = max pooling, 2 = average pooling
  conf.run[4].pool_size = 0x303;  // bits [7:0] = width, bits [15:8] = height
  conf.run[4].pool_stride = 0x202;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[4].pool_pad = 0x1010101;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[4].pool_avg_param = 0x0;  // Must be set to 1/pool_size^2 in FP16 format when using average pooling (average pooling assumes square size)
  conf.run[4].actfunc = 2;  // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6
  conf.run[4].actfunc_param = 0x0;  // Leaky ReLU parameter (NOTE: 0x2E66 is 0.1 in FP16)
  conf.run[4].rectifi_en = 0;  // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)
  conf.run[4].lrn = 0x0;  // [0] : 1 = LRN enable, 0 = LRN disable, [1] : 1 = incl. power func, 0 = excl., [8:11] = x^2 scale factor log2
  //--------------------------------------------------
  //RUN : 5
  //--------------------------------------------------
  //->: inception_3b/pool
  conf.run[5].m = 256;  // Output Channels
  conf.run[5].conv_enable = 0;  // 1 = Enabled, 0 = Disabled
  conf.run[5].p = 1;  // Filter Width and Height
  conf.run[5].pz = 1;  // Filter Depth
  conf.run[5].weight_buf.mem = weights_mem;
  conf.run[5].weight_buf.offs = 0;
  conf.run[5].weight_fmt = 0;  // Weight format (0 = random access blocks, 1 = compact stream, 3 = 8-bit qunatized stream)
  conf.run[5].conv_pad = 0x0;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[5].conv_stride = 0x101;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[5].conv_dilation = 0x0;  // bits [7:0] = X dilation, bits [15:8] = Y dilation
  conf.run[5].pool_enable = 1;  // 0 = disabled, 1 = max pooling, 2 = average pooling
  conf.run[5].pool_size = 0x303;  // bits [7:0] = width, bits [15:8] = height
  conf.run[5].pool_stride = 0x101;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[5].pool_pad = 0x1010101;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[5].pool_avg_param = 0x0;  // Must be set to 1/pool_size^2 in FP16 format when using average pooling (average pooling assumes square size)
  conf.run[5].actfunc = 0;  // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6
  conf.run[5].actfunc_param = 0x0;  // Leaky ReLU parameter (NOTE: 0x2E66 is 0.1 in FP16)
  conf.run[5].rectifi_en = 0;  // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)
  conf.run[5].lrn = 0x0;  // [0] : 1 = LRN enable, 0 = LRN disable, [1] : 1 = incl. power func, 0 = excl., [8:11] = x^2 scale factor log2
  //--------------------------------------------------
  //RUN : 6
  //--------------------------------------------------
  //->: inception_3b/pool_proj
  //->: inception_3b/relu_pool_proj
  //->: pool3/3x3_s2
  conf.run[6].m = 64;  // Output Channels
  conf.run[6].conv_enable = 1;  // 1 = Enabled, 0 = Disabled
  conf.run[6].p = 1;  // Filter Width and Height
  conf.run[6].pz = 1;  // Filter Depth
  conf.run[6].weight_buf.mem = weights_mem;
  conf.run[6].weight_buf.offs = 0;
  conf.run[6].weight_fmt = 3;  // Weight format (0 = random access blocks, 1 = compact stream, 3 = 8-bit qunatized stream)
  conf.run[6].conv_pad = 0x0;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[6].conv_stride = 0x101;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[6].conv_dilation = 0x0;  // bits [7:0] = X dilation, bits [15:8] = Y dilation
  conf.run[6].pool_enable = 1;  // 0 = disabled, 1 = max pooling, 2 = average pooling
  conf.run[6].pool_size = 0x303;  // bits [7:0] = width, bits [15:8] = height
  conf.run[6].pool_stride = 0x202;  // bits [7:0] = X stride, bits [15:8] = Y stride
  conf.run[6].pool_pad = 0x1010101;  // bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  conf.run[6].pool_avg_param = 0x0;  // Must be set to 1/pool_size^2 in FP16 format when using average pooling (average pooling assumes square size)
  conf.run[6].actfunc = 2;  // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6
  conf.run[6].actfunc_param = 0x0;  // Leaky ReLU parameter (NOTE: 0x2E66 is 0.1 in FP16)
  conf.run[6].rectifi_en = 0;  // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)
  conf.run[6].lrn = 0x0;  // [0] : 1 = LRN enable, 0 = LRN disable, [1] : 1 = incl. power func, 0 = excl., [8:11] = x^2 scale factor log2

  cmdlist = dmp_dv_cmdlist_create(ctx);
  if (!cmdlist) {
    ERR("dmp_dv_cmdlist_create() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  if (dmp_dv_cmdlist_add_raw(cmdlist, (struct dmp_dv_cmdraw*)&conf)) {
    ERR("dmp_dv_cmdlist_add_raw() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  if (dmp_dv_cmdlist_commit(cmdlist)) {
    ERR("dmp_dv_cmdlist_commit() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  exec_id = dmp_dv_cmdlist_exec(cmdlist);
  if (exec_id < 0) {
    ERR("dmp_dv_cmdlist_exec() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  if (dmp_dv_cmdlist_wait(cmdlist, exec_id)) {
    ERR("dmp_dv_cmdlist_wait() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  LOG("SUCCEEDED\n");

  result = 0;

  L_EXIT:

  dmp_dv_cmdlist_release(cmdlist);
  dmp_dv_mem_release(io_mem);
  dmp_dv_mem_release(weights_mem);
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

  LOG("EXIT%s: test_context: %d FDs\n", result ? "(FAILED)" : "", n_fd);
  return result;
}


int main(int argc, char **argv) {
  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  for (int i = 0; i < 3; ++i) {
    res = test_multirun();
    if (res) {
      ++n_err;
      break;  // we are testing FPGA hanging, so exit on first failure
    }
    else {
      ++n_ok;
    }
  }

  LOG("Tests succeeded: %d\n", n_ok);
  LOG("Tests failed: %d\n", n_err);
  return n_err;
}
