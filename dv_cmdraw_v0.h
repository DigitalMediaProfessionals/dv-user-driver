/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief User-space definition of dv_cmdraw structure version v0.
 */
#pragma once

#include "dv.h"


/// @brief Convolution layer runs.
typedef struct dv_cmdraw_v0_conv_run_impl {
  uint16_t m;  // Number of Output Channels

  uint16_t conv_enable;    // 1 = Enabled, 0 = Disabled, 3 = LRN
  uint16_t p;              // Filter Size (width = height)
  uint16_t pz;             // Filter Depth (1 in case of 2D convolution)
  uint32_t conv_pad;       // Bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  uint16_t conv_stride;    // Bits [7:0] = X stride, bits [15:8] = Y stride
  uint16_t conv_dilation;  // Bits [7:0] = X dilation, bits [15:8] = Y dilation
  dv_buf weight_buf;       // Buffer with packed weights
  uint16_t weight_fmt;     // Weight format (0 = random access blocks, 1 = compact stream, 2 = 8-bit qunatized stream)

  uint16_t pool_enable;     // 0 = disabled, 1 = max pooling, 2 = average pooling
  uint16_t pool_avg_param;  // Usually be set to 1/pool_size^2 in FP16 when using average pooling (average pooling assumes square size)
  uint16_t pool_size;       // Bits [7:0] = width, bits [15:8] = height
  uint16_t pool_stride;     // Bits [7:0] = X stride, bits [15:8] = Y stride
  uint32_t pool_pad;        // Bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding

  uint16_t actfunc;        // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU
  uint16_t actfunc_param;  // Leaky ReLU parameter in FP16
  uint16_t rectifi_en;     // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)
  uint16_t lrn;            // Bits [0]: 1 = LRN enable, 0 = LRN disable, [1]: 1 = incl. power func, 0 = excl., [8:11]: x^2 scale factor log2
} __attribute__((packed)) dv_cmdraw_v0_conv_run;


/// @brief Raw command for execution.
typedef struct dv_cmdraw_v0_impl {
  int32_t size;     // size of this structure
  int32_t version;  // version of this structure

  uint32_t topo;                   // [31:0] Output Destination of each run, 0 = UBUF, 1 = EXTMEM
  uint16_t w;                      // Input Width
  uint16_t h;                      // Input Height
  uint16_t z;                      // Input Depth
  uint16_t c;                      // Input Channels
  dv_buf   input_buf;              // Input buffer
  uint16_t input_circular_offset;  // Input Depth circular offset
  uint16_t tiles;                  // Number of horizontal tiles  // TODO: calculate in kernel module.

  dv_buf output_buf;     // Output buffer
  dv_buf eltwise_buf;    // Buffer for elementwise add (0 = UBUF Input Buffer)
  uint16_t output_mode;  // 0 = concat, 1 = elementwise add

  dv_cmdraw_v0_conv_run run[32];  // description of each run
} __attribute__((packed)) dv_cmdraw_v0;
