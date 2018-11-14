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
struct dmp_dv_cmdraw_conv_v0_run {
  struct dmp_dv_buf weight_buf;  // Buffer with packed weights

  uint32_t conv_pad;        // Bits [6:0] = left padding, bits [15:8] = right padding, bits [22:16] = top padding, bits [31:24] = bottom padding
  uint32_t pool_pad;        // Bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
  uint16_t m;               // Number of Output Channels
  uint16_t conv_enable;     // 1 = Enabled, 0 = Disabled, 3 = Depthwise
  uint16_t p;               // Filter Size (bits[7:0] = width, bits[15:8] = height)
  uint16_t pz;              // Filter Depth (1 in case of 2D convolution)
  uint16_t conv_stride;     // Bits [7:0] = X stride, bits [15:8] = Y stride
  uint16_t conv_dilation;   // Bits [7:0] = X dilation, bits [15:8] = Y dilation
  uint16_t weight_fmt;      // Weights format (0 = random access blocks, 1 = FP16, 3 = 8-bit quantized)
  uint16_t pool_enable;     // 0 = disabled, 1 = max pooling, 2 = average pooling, 4 - upsampling
  uint16_t pool_avg_param;  // Usually be set to 1/pool_size^2 in FP16 when using average pooling (average pooling assumes square size)
  uint16_t pool_size;       // Bits [7:0] = width, bits [15:8] = height
  uint16_t pool_stride;     // Bits [7:0] = X stride, bits [15:8] = Y stride
  uint16_t actfunc;         // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU, 3 = Sigmoid, 4 = PReLU, 5 = ELU, 6 = ReLU6
  uint16_t actfunc_param;   // Leaky ReLU parameter in FP16
  uint16_t rectifi_en;      // Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)
  uint16_t lrn;             // Bits [0]: 1 = LRN enable, 0 = LRN disable, [1]: 1 = incl. power func, 0 = excl., [8:11]: x^2 scale factor log2
  uint16_t rsvd;            // padding to 64-bit size
};


/// @brief Raw command for convolutional block version 0.
struct dmp_dv_cmdraw_conv_v0 {
  struct dmp_dv_cmdraw header;  // General structure information

  struct dmp_dv_buf input_buf;    // Input buffer
  struct dmp_dv_buf output_buf;   // Output buffer
  struct dmp_dv_buf eltwise_buf;  // Buffer for elementwise add (0 = UBUF Input Buffer)

  uint32_t topo;                   // [31:0] Output Destination of each run, 0 = UBUF, 1 = EXTMEM
  uint16_t w;                      // Input Width
  uint16_t h;                      // Input Height
  uint16_t z;                      // Input Depth
  uint16_t c;                      // Input Channels
  uint16_t input_circular_offset;  // Input Depth circular offset
  uint16_t output_mode;            // 0 = concat, 1 = elementwise add

  struct dmp_dv_cmdraw_conv_v0_run run[32];  // description of each run
};


/// @brief Raw command for fully connected block version 0.
struct dmp_dv_cmdraw_fc_v0 {
  struct dmp_dv_cmdraw header;   // General structure information

  struct dmp_dv_buf weight_buf;  // Buffer with packed weights
  struct dmp_dv_buf input_buf;   // Input buffer
  struct dmp_dv_buf output_buf;  // Output buffer

  uint16_t input_size;     // Size of the input in elements
  uint16_t output_size;    // Size of the output in elements

  uint16_t weight_fmt;     // Weights format: 0 = half-float unquantized, 1 = 8-bit quantized

  uint16_t actfunc;        // Activation Function: 0 = None, 1 = ReLU, 2 = Tanh, 3 = Leaky ReLU, 4 = Sigmoid, 5 = PReLU (PReLU must be used with POST-OP=1)
  uint16_t actfunc_param;  // Leaky ReLU parameter (in FP16 format), 0 = non-leaky
  uint16_t rsvd[3];        // padding to 64-bit size
};


/// @brief Raw command for image processing unit version 0.
struct dmp_dv_cmdraw_ipu_v0 {
  struct dmp_dv_cmdraw header; 	// general structure information

  /* image buffer */
  struct dmp_dv_buf tex;  // texture buffer
  struct dmp_dv_buf rd;   // read buffer
  struct dmp_dv_buf wr;   // write buffer

  /* image format */
  uint8_t fmt_tex;  	    // format of texture buffer. This must be DMP_DV_RGBA8888, DMP_DV_RGB888 or DMP_DV_LUT.
  uint8_t fmt_rd;   	    // format of read buffer. This must be DMP_DV_RGBA8888 or DMP_DV_RGB888.
  uint8_t fmt_wr;   	    // format of write buffer. This must be DMP_DV_RGBA8888, DMP_DV_RGB888 or DMP_DV_RGBFP16.
  uint8_t rsvd1[1];			  // padding to 16-bit size

  /* dimension */
  uint16_t tex_width;	    // width of texture
  uint16_t tex_height;		// height of texture
  uint16_t rect_width;		// width of rendering rectangle
  uint16_t rect_height;		// height of rendering rectangle

  /* stride */
  int32_t stride_rd;  	 	// stride for read buffer
  int32_t stride_wr;  	 	// stride for write buffer

  uint32_t lut[32];		  	// look up table for texture of DMP_DV_LUT.
  uint8_t ncolor_lut;	  	// number of color in lut.  If 0, the look up table used at the last time is used.

  uint8_t alpha;  	      // alpha value for blending

  /* operation flags */
  uint8_t transpose;  	  	// exchange x-y axis of texture
  uint8_t use_const_alpha;	// use alpha in this structure for blending
  uint8_t use_tex;  		    // use texture in this structure
  uint8_t use_rd;  	      	// use rd in this structure
  uint8_t blf;  		        // apply bilinear filter

  /** swizzle
   * Specify an order of RGBA in texture buffer
   * aidx can be -1 if the texture not contain alpha channel.
   *   - If aidx is -1 ridx, gidx and bidx must be in {0, 1, 2} without overlap.
   *   - Otherwize, ridx, gidx, bidx and aidx must be in {0, 1, 2, 3} without overlap.
   */
  int8_t ridx;  // index of red channel
  int8_t gidx;  // index of green channel
  int8_t bidx;  // index of blue channel
  int8_t aidx;  // index of alpha channel

  /*! cnversion to fp16
   * Each pixels in type of uint8_t is converted to a fp16 value as below. 
   *   - For DMP_DV_CNV_FP16_SUB, R_F16 = F16(R_8 - param[0]), G_F16 = F16(G_8 - param[1]), B_F16 = F16(B_8 - param[2])
   *   - For DMP_DV_CNV_FP16_DIV_255, R_F16 = F16(R_8/255.0), G_F16 = F16(G_8/255.0), B_F16 = F16(B_8/255.0). cnv_param can be NULL.
   * (*_8 means uint8_t values of the channel, *_F16 means fp16 values of the channel, F16() represents cast function to fp16)
   */
  uint8_t cnv_type;       // conversion type
  uint8_t cnv_param[3];   // conversion parameter
  uint8_t rsvd2[5];		    // padding to 64-bit size
};

