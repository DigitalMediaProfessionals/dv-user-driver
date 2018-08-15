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
 * @brief User-space definition of dv_cmdraw structure version v0.
 */
#pragma once

#include "dmp_dv.h"


/// @brief Convolution layer runs.
/// @details Members within structure are rearranged by size to avoid requirements for 64-bits padding in the middle.
typedef struct dmp_dv_cmdraw_conv_v0_run_impl {
  dmp_dv_buf weight_buf;    // Buffer with packed weights
  uint32_t conv_pad;        // Bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  uint32_t pool_pad;        // Bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  uint16_t m;               // Number of Output Channels
  uint16_t conv_enable;     // 1 = Enabled, 0 = Disabled, 3 = Depthwise
  uint16_t p;               // Filter Size (bits[7:0] = width, bits[15:8] = height)
  uint16_t pz;              // Filter Depth (1 in case of 2D convolution)
  uint16_t conv_stride;     // Bits [7:0] = X stride, bits [15:8] = Y stride
  uint16_t conv_dilation;   // Bits [7:0] = X dilation, bits [15:8] = Y dilation
  uint16_t weight_fmt;      // Weights format (0 = random access blocks, 1 = compact stream, 3 = 8-bit quantized stream)
  uint16_t pool_enable;     // 0 = disabled, 1 = max pooling, 2 = average pooling, 4 - upsampling
  uint16_t pool_avg_param;  // Usually be set to 1/pool_size^2 in FP16 when using average pooling (average pooling assumes square size)
  uint16_t pool_size;       // Bits [7:0] = width, bits [15:8] = height
  uint16_t pool_stride;     // Bits [7:0] = X stride, bits [15:8] = Y stride
  uint16_t actfunc;         // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6
  uint16_t actfunc_param;   // Leaky ReLU parameter in FP16
  uint16_t rectifi_en;      // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)
  uint16_t lrn;             // Bits [0]: 1 = LRN enable, 0 = LRN disable, [1]: 1 = incl. power func, 0 = excl., [8:11]: x^2 scale factor log2
  uint16_t rsvd;            // padding to 64-bit size
} dmp_dv_cmdraw_conv_v0_run;


/// @brief Raw command for convolutional block version 0.
typedef struct dmp_dv_cmdraw_conv_v0_impl {
  dmp_dv_cmdraw header;               // General structure information

  dmp_dv_buf input_buf;               // Input buffer
  dmp_dv_buf output_buf;              // Output buffer
  dmp_dv_buf eltwise_buf;             // Buffer for elementwise add (0 = UBUF Input Buffer)
  uint32_t topo;                      // [31:0] Output Destination of each run, 0 = UBUF, 1 = EXTMEM
  uint16_t w;                         // Input Width
  uint16_t h;                         // Input Height
  uint16_t z;                         // Input Depth
  uint16_t c;                         // Input Channels
  uint16_t input_circular_offset;     // Input Depth circular offset
  uint16_t output_mode;               // 0 = concat, 1 = elementwise add
  dmp_dv_cmdraw_conv_v0_run run[32];  // description of each run
} dmp_dv_cmdraw_conv_v0;


/// @brief Raw command for fully connected block version 0.
typedef struct dmp_dv_cmdraw_fc_v0_impl {
  dmp_dv_cmdraw header;    // General structure information

  dmp_dv_buf weight_buf;   // Buffer with packed weights
  dmp_dv_buf input_buf;    // Input buffer
  dmp_dv_buf output_buf;   // Output buffer

  uint16_t input_size;     // Size of the input in elements
  uint16_t output_size;    // Size of the output in elements

  uint16_t weight_fmt;     // Weights format: 0 = half-float unquantized, 1 = 8-bit quantized

  uint16_t actfunc;        // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6
  uint16_t actfunc_param;  // Leaky ReLU parameter (in FP16 format), 0 = non-leaky
  uint16_t rsvd[3];        // padding to 64-bit size
} dmp_dv_cmdraw_fc_v0;
