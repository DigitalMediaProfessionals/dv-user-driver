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
/// @file
/// @brief Weights-packing helper functions for fully connected layer.

#include "common.h"


/// @brief Packs fully connected layer weights and biases into output array possibly rearranging them to match input and output shapes.
/// @param c_input Number of input channels.
/// @param h_input Input height (set to 1 for 1D input).
/// @param w_input Input width (set to 1 for 1D input).
/// @param c_output Number of output channels.
/// @param h_output Output height (set to 1 for 1D output).
/// @param w_output Output width (set to 1 for 1D output).
/// @param quant_map Quantization table for weights (but not bias), 256 elements, can be NULL.
/// @param weights If quant_map is NULL, array of half precision floating point weights in NCHW format (N=output_size), else array of 1-byte indices.
/// @param bias Array of half precision floating point biases of size output_size.
/// @param packed_weights Output buffer for packed weights information (can be NULL if packed_weights_size is 0).
/// @param packed_weights_size On input, contains the size of the packed_weights buffer in bytes (can be 0, in such case it will be filled with the required buffer size), on output will contain the required buffer size.
/// @return 0 on success, non-zero otherwise.
/// @details The function packs weights in NCHW format to the DV input format WHC8 (n_channels / 8, width, height, 8 channels)
///          with rearranging to produce output in DV format WHC8.
///          It is thread-safe.
int dmp_dv_pack_fc_weights(
    int c_input, int h_input, int w_input,
    int c_output, int h_output, int w_output,
    const uint16_t quant_map[256],
    const void *weights, const uint16_t *bias,
    uint8_t *packed_weights, size_t *packed_weights_size) {

  if ((c_input <= 0) || (h_input <= 0) || (w_input <= 0) ||
      (c_output <= 0) || (h_output <= 0) || (w_output <= 0)) {
    SET_ERR("Input/output dimensions must be positive");
    return EINVAL;
  }
  if (!packed_weights_size) {
    SET_ERR("packed_weights_size must not be NULL");
    return EINVAL;
  }
  if ((!packed_weights) && (*packed_weights_size)) {
    SET_ERR("packed_weights is NULL but *packed_weights_size is non-zero");
    return EINVAL;
  }
  if ((packed_weights) && (((size_t)packed_weights) & 15)) {
    SET_ERR("packed_weights must be 16-bytes aligned");
    return EINVAL;
  }

  size_t out_offs = 0;
  if (quant_map) {
    if (out_offs + 512 <= *packed_weights_size) {
      memcpy(packed_weights + out_offs, quant_map, 512);
    }
    out_offs += 512;
  }

  size_t output_size;
  if ((h_input == 1) && (w_input == 1) &&
      (h_output == 1) && (w_output == 1)) {  // 1D input and 1D output
    output_size = c_output;
    size_t weights_size = (size_t)c_input * output_size;
    if (!quant_map) {
      weights_size <<= 1;
    }
    if (out_offs + weights_size <= *packed_weights_size) {
      memcpy(packed_weights + out_offs, weights, weights_size);
    }
    out_offs += weights_size;
  }
  else {
    const int s1 = h_input * w_input;
    const int s2 = c_input * s1;
    const int s3 = w_output * s2;
    const int s4 = h_output * s3;
    output_size = c_output * h_output * w_output;
    const size_t weights_size = (size_t)c_input * h_input * w_input * output_size;
    if ((quant_map) && (out_offs + weights_size <= *packed_weights_size)) {
      // Input is in 1CHW format, weights are NCHW where N=(chw) itself:
      // weights are: (N=(c_output, h_output, w_output), C=c_input, H=h_input, W=w_input)
      // => (w_output, h_output, 8)+, (w_input, h_input, 8)+.
      const uint8_t *wi = (const uint8_t*)weights;
      uint8_t *wo = packed_weights + out_offs;
      for (int c_out_start = 0, o_offs = 0; c_out_start < c_output; c_out_start += 8) {
        const int c_out_end = c_out_start + 8 <= c_output ? c_out_start + 8 : c_output;
        for (int w_out = 0; w_out < w_output; ++w_out) {
          for (int h_out = 0; h_out < h_output; ++h_out) {
            for (int c_out = c_out_start; c_out < c_out_end; ++c_out) {
              for (int c_in_start = 0; c_in_start < c_input; c_in_start += 8) {
                const int c_in_end = c_in_start + 8 <= c_input ? c_in_start + 8 : c_input;
                for (int w_in = 0; w_in < w_input; ++w_in) {
                  for (int h_in = 0; h_in < h_input; ++h_in) {
                    for (int c_in = c_in_start; c_in < c_in_end; ++c_in, ++o_offs) {
                      wo[o_offs] = wi[c_out * s4 + h_out * s3 + w_out * s2 + c_in * s1 + h_in * w_input + w_in];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    else if ((!quant_map) && (out_offs + weights_size * 2 <= *packed_weights_size)) {
      const uint16_t *wi = (const uint16_t*)weights;
      uint16_t *wo = (uint16_t*)(packed_weights + out_offs);
      for (int c_out_start = 0, o_offs = 0; c_out_start < c_output; c_out_start += 8) {
        const int c_out_end = c_out_start + 8 <= c_output ? c_out_start + 8 : c_output;
        for (int w_out = 0; w_out < w_output; ++w_out) {
          for (int h_out = 0; h_out < h_output; ++h_out) {
            for (int c_out = c_out_start; c_out < c_out_end; ++c_out) {
              for (int c_in_start = 0; c_in_start < c_input; c_in_start += 8) {
                const int c_in_end = c_in_start + 8 <= c_input ? c_in_start + 8 : c_input;
                for (int w_in = 0; w_in < w_input; ++w_in) {
                  for (int h_in = 0; h_in < h_input; ++h_in) {
                    for (int c_in = c_in_start; c_in < c_in_end; ++c_in, ++o_offs) {
                      wo[o_offs] = wi[c_out * s4 + h_out * s3 + w_out * s2 + c_in * s1 + h_in * w_input + w_in];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    out_offs += quant_map ? weights_size : weights_size << 1;
  }

  // bias must be 16-bytes aligned
  if (out_offs & 15) {
    const int d = 16 - (out_offs & 15);
    if (out_offs + d <= *packed_weights_size) {
      memset(packed_weights + out_offs, 0, d);
    }
    out_offs += d;
  }

  if (out_offs + output_size * 2 <= *packed_weights_size) {
    memcpy(packed_weights + out_offs, bias, output_size * 2);
  }
  out_offs += output_size * 2;

  if (out_offs & 15) {  // zero-pad output to 16-bytes
    const int d = 16 - (out_offs & 15);
    if (out_offs + d <= *packed_weights_size) {
      memset(packed_weights + out_offs, 0, d);
    }
    out_offs += d;
  }

  int res = 0;
  if ((*packed_weights_size) && (*packed_weights_size < out_offs)) {
    SET_ERR("Not all weights were filled: provided buffer size %zu while %zu is required", *packed_weights_size, out_offs);
    res = -1;
  }

  *packed_weights_size = out_offs;
  return res;
}
