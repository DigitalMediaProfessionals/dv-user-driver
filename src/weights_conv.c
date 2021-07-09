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
/// @brief Weights-packing helper functions for convolutional layer.

#include "common.h"


/// @brief Minimum for integers.
static inline int imin(int a, int b) {
  return a < b ? a : b;
}


/// @brief Maximum for integers.
static inline int imax(int a, int b) {
  return a > b ? a : b;
}


/// @brief Writes bias to packed weights buffer.
static inline void write_bias(int m_start, int m_stop, size_t *out_offs, size_t *output_size, uint8_t *output, const uint16_t *bias) {
  size_t n = (m_stop - m_start) << 1;
  if (*out_offs + n <= *output_size) {
    memcpy(output + *out_offs, bias + m_start, n);
  }
  *out_offs += n;
  const int bias_pad = m_start + 8 - m_stop;
  if (bias_pad > 0) {
    n = bias_pad << 1;
    if (*out_offs + n <= *output_size) {
      memset(output + *out_offs, 0, n);
    }
    *out_offs += n;
  }
}


/// @brief Packs convolution layer weights and biases into output array.
/// @param n_channels Number of input channels, for depthwise convolution this must be set to 1.
/// @param kx Kernel width.
/// @param ky Kernel height.
/// @param n_kernels Number of output channels.
/// @param quant_map Quantization table for weights (but not bias), 256 elements, can be NULL.
/// @param weights If quant_map is NULL, array of half precision floating point weights in NCHW format, else array of 1-byte indices.
/// @param bias Array of half precision floating point biases of size n_kernels.
/// @param prelu Array of half precision floating point values for PReLU activation of size n_kernels, can be NULL.
/// @param packed_weights Output buffer for packed weights information (can be NULL if packed_weights_size is 0).
/// @param packed_weights_size On input, contains the size of the packed_weights buffer in bytes (can be 0, in such case it will be filled with the required buffer size), on output will contain the required buffer size.
/// @return 0 on success, non-zero otherwise.
/// @details When packing weights for deconvolution, HW plane must be rotated by 180 degrees.
///          It is thread-safe.
int dmp_dv_pack_conv_weights(
    int n_channels, int kx, int ky, int n_kernels,
    const uint16_t quant_map[256],
    const void *weights, const uint16_t *bias, const uint16_t *prelu,
    uint8_t *packed_weights, size_t *packed_weights_size) {

  const int p = imax(kx, ky) | 1;  // next odd number

  if ((p > 7) || (imin(kx, ky) <= 0)) {
    SET_ERR("Only kernels of sizes {1, 2, 3, 4, 5, 6, 7} are supported, got %dx%d", kx, ky);
    return -1;
  }
  if (n_channels <= 0) {
    SET_ERR("Number of input channels must be positive, got %d", n_channels);
    return -1;
  }
  if (n_kernels <= 0) {
    SET_ERR("Number of output channels must be positive, got %d", n_kernels);
    return -1;
  }
  if (!packed_weights_size) {
    SET_ERR("packed_weights_size must not be NULL");
    return -1;
  }
  if ((!packed_weights) && (*packed_weights_size)) {
    SET_ERR("packed_weights is NULL but *packed_weights_size is non-zero");
    return -1;
  }
  if ((packed_weights) && (((size_t)packed_weights) & 15)) {
    SET_ERR("packed_weights must be 16-bytes aligned");
    return EINVAL;
  }

  int retval = 0;

  if (*packed_weights_size) {
    memset(packed_weights, 0, *packed_weights_size);
  }

  size_t out_offs = 0;
  if (quant_map) {
    if (out_offs + 512 <= *packed_weights_size) {
      memcpy(packed_weights + out_offs, quant_map, 512);
    }
    out_offs += 512;
  }

  // weights.shape = (n_kernels, n_channels, ky, kx)
  const int s2 = kx;
  const int s1 = ky * kx;
  const int s0 = n_channels * ky * kx;

  uint8_t buf8[12][6];
  uint16_t buf16[12][6];
  if (quant_map) {
    memset(&buf8[0][0], 0, sizeof(buf8));
  }
  else {
    memset(&buf16[0][0], 0, sizeof(buf16));
  }
  if ((sizeof(buf8) != 12 * 6) || (sizeof(buf16) != 12 * 6 * 2)) {
    SET_ERR("sizeof mismatch: sizeof(buf8)=%zu sizeof(buf16)=%zu while expecting %d, %d",
            sizeof(buf8), sizeof(buf16), 12 * 6, 12 * 6 * 2);
    return -1;
  }

  switch (p) {
    case 7:
    {
      for (int m_start = 0; m_start < n_kernels; m_start += 8) {  // loop by kernels (chunks of size 8) i.e. output channels
        const int m_stop = imin(m_start + 8, n_kernels);

        write_bias(m_start, m_stop, &out_offs, packed_weights_size, packed_weights, bias);  // write bias values for a specific chunk with zero padding to 8
        if (prelu) {
          write_bias(m_start, m_stop, &out_offs, packed_weights_size, packed_weights, prelu);  // write PReLU values for a specific chunk with zero padding to 8
        }

        for (int c_start = 0; c_start < n_channels; c_start += 8) {  // loop by input channels (chunks of size 8)
          const int c_stop = imin(c_start + 8, n_channels);
          for (int m = m_start; m < m_stop; ++m) {  // loop by specific kernel inside chunk
            for (int c = c_start; c < c_stop; ++c) {  // loop by specific channel inside chunk
              const int offs2 = m * s0 + c * s1;
              if (quant_map) {  // Quantized 8-bit weights
                if (out_offs + sizeof(buf8) <= *packed_weights_size) {
                  const uint8_t *w = (const uint8_t*)weights;
                  for (int y = 0; y < ky; ++y) {
                    for (int x = 0; x < imin(6, kx); ++x) {
                      buf8[5 + y + (p - ky)][x] = w[offs2 + y * s2 + x];
                    }
                  }
                  if (kx > 6) {
                    static const int remap[7] = {
                        2 * 6 + 5, 0 * 6 + 3, 1 * 6 + 3, 2 * 6 + 3, 0 * 6 + 0, 1 * 6 + 0, 2 * 6 + 0
                    };
                    uint8_t *buf8p = &buf8[0][0];
                    for (int y = 0; y < ky; ++y) {
                      buf8p[remap[y + (p - ky)]] = w[offs2 + y * s2 + 6];
                    }
                  }
                  memcpy(packed_weights + out_offs, &buf8[0][0], sizeof(buf8));
                }
                out_offs += sizeof(buf8);
              }
              else {  // Half float 16-bit weights
                if (out_offs + sizeof(buf16) <= *packed_weights_size) {
                  const uint16_t *w = (const uint16_t*)weights;
                  for (int y = 0; y < ky; ++y) {
                    for (int x = 0; x < imin(6, kx); ++x) {
                      buf16[5 + y + (p - ky)][x] = w[offs2 + y * s2 + x];
                    }
                  }
                  if (kx > 6) {
                    static const int remap[7] = {
                        2 * 6 + 5, 0 * 6 + 3, 1 * 6 + 3, 2 * 6 + 3, 0 * 6 + 0, 1 * 6 + 0, 2 * 6 + 0
                    };
                    uint16_t *buf16p = &buf16[0][0];
                    for (int y = 0; y < ky; ++y) {
                      buf16p[remap[y + (p - ky)]] = w[offs2 + y * s2 + 6];
                    }
                  }
                  memcpy(packed_weights + out_offs, &buf16[0][0], sizeof(buf16));
                }
                out_offs += sizeof(buf16);
              }
            }
          }
        }
      }
      break;
    }
    case 5:
    {
      for (int m_start = 0; m_start < n_kernels; m_start += 8) {
        const int m_stop = imin(m_start + 8, n_kernels);

        write_bias(m_start, m_stop, &out_offs, packed_weights_size, packed_weights, bias);
        if (prelu) {
          write_bias(m_start, m_stop, &out_offs, packed_weights_size, packed_weights, prelu);
        }

        for (int c_start = 0; c_start < n_channels; c_start += 8) {
          const int c_stop = imin(c_start + 8, n_channels);
          for (int m = m_start; m < m_stop; ++m) {
            if (quant_map) {  // Quantized 8-bit weights
              const uint8_t *w = (const uint8_t*)weights;
              for (int c = c_start; c < c_stop; ++c) {
                const int offs2 = m * s0 + c * s1;
                const int t = c & 1;
                if ((t == 0) && (c == c_stop - 1)) {
                  memset(&buf8[0][0], 0, sizeof(buf8));
                }
                if (out_offs + sizeof(buf8) <= *packed_weights_size) {
                  for (int y = 0; y < ky; ++y) {
                    for (int x = 0; x < kx; ++x) {
                      buf8[7 - t * 6 + y + (p - ky)][x] = w[offs2 + y * s2 + x];
                    }
                  }
                }
                if ((t == 1) || ((t == 0) && (c == c_stop - 1))) {
                  if (out_offs + sizeof(buf8) <= *packed_weights_size) {
                    memcpy(packed_weights + out_offs, &buf8[0][0], sizeof(buf8));
                  }
                  out_offs += sizeof(buf8);
                }
              }
            }
            else {  // Half float 16-bit weights
              const uint16_t *w = (const uint16_t*)weights;
              for (int c = c_start; c < c_stop; ++c) {
                const int offs2 = m * s0 + c * s1;
                const int t = c & 1;
                if ((t == 0) && (c == c_stop - 1)) {
                  memset(&buf16[0][0], 0, sizeof(buf16));
                }
                if (out_offs + sizeof(buf16) <= *packed_weights_size) {
                  for (int y = 0; y < ky; ++y) {
                    for (int x = 0; x < kx; ++x) {
                      buf16[7 - t * 6 + y + (p - ky)][x] = w[offs2 + y * s2 + x];
                    }
                  }
                }
                if ((t == 1) || ((t == 0) && (c == c_stop - 1))) {
                  if (out_offs + sizeof(buf16) <= *packed_weights_size) {
                    memcpy(packed_weights + out_offs, &buf16[0][0], sizeof(buf16));
                  }
                  out_offs += sizeof(buf16);
                }
              }
            }
          }
        }
      }
      break;
    }
    case 3:
    {
      for (int m_start = 0; m_start < n_kernels; m_start += 8) {
        const int m_stop = imin(m_start + 8, n_kernels);

        write_bias(m_start, m_stop, &out_offs, packed_weights_size, packed_weights, bias);
        if (prelu) {
          write_bias(m_start, m_stop, &out_offs, packed_weights_size, packed_weights, prelu);
        }

        for (int c_start = 0; c_start < n_channels; c_start += 8) {
          const int c_stop = imin(c_start + 8, n_channels);
          if (c_stop - c_start != 8) {
            if (quant_map) {
              memset(&buf8[0][0], 0, sizeof(buf8));
            }
            else {
              memset(&buf16[0][0], 0, sizeof(buf16));
            }
          }
          for (int m = m_start; m < m_stop; ++m) {
            if (quant_map) {  // Quantized 8-bit weights
              if (out_offs + sizeof(buf8) <= *packed_weights_size) {
                const uint8_t *w = (const uint8_t*)weights;
                for (int c = c_start; c < c_stop; ++c) {
                  const int offs2 = m * s0 + c * s1;
                  const int t = c & 7;
                  for (int y = 0; y < ky; ++y) {
                    for (int x = 0; x < kx; ++x) {
                      buf8[9 - (t >> 1) * 3 + y + (p - ky)][(t & 1) * 3 + x] = w[offs2 + y * s2 + x];
                    }
                  }
                }
                memcpy(packed_weights + out_offs, &buf8[0][0], sizeof(buf8));
              }
              out_offs += sizeof(buf8);
            }
            else {  // Half float 16-bit weights
              if (out_offs + sizeof(buf16) <= *packed_weights_size) {
                const uint16_t *w = (const uint16_t*)weights;
                for (int c = c_start; c < c_stop; ++c) {
                  const int offs2 = m * s0 + c * s1;
                  const int t = c & 7;
                  for (int y = 0; y < ky; ++y) {
                    for (int x = 0; x < kx; ++x) {
                      buf16[9 - (t >> 1) * 3 + y + (p - ky)][(t & 1) * 3 + x] = w[offs2 + y * s2 + x];
                    }
                  }
                }
                memcpy(packed_weights + out_offs, &buf16[0][0], sizeof(buf16));
              }
              out_offs += sizeof(buf16);
            }
          }
        }
      }
      break;
    }
    case 1:
    {
      for (int m_start = 0; m_start < n_kernels; m_start += 8) {
        const int m_stop = imin(m_start + 8, n_kernels);

        write_bias(m_start, m_stop, &out_offs, packed_weights_size, packed_weights, bias);
        if (prelu) {
          write_bias(m_start, m_stop, &out_offs, packed_weights_size, packed_weights, prelu);
        }

        for (int c_start = 0; c_start < n_channels; c_start += 64) {
          const int c_stop = imin(c_start + 64, n_channels);
          if (c_stop - c_start != 64) {
            if (quant_map) {
              memset(&buf8[0][0], 0, sizeof(buf8));
            }
            else {
              memset(&buf16[0][0], 0, sizeof(buf16));
            }
          }
          for (int m = m_start; m < m_stop; ++m) {
            if (quant_map) {  // Quantized 8-bit weights
              if (out_offs + sizeof(buf8) <= *packed_weights_size) {
                const uint8_t *w = (const uint8_t*)weights;
                for (int c = c_start; c < c_stop; ++c) {
                  const int t = c & 7;
                  const int x = ((c & 63) >> 3) % 3;
                  const int y = ((c & 63) >> 3) / 3;
                  buf8[11 - (t >> 1) * 3 - y][(t & 1) * 3 + x] = w[m * s0 + c * s1];
                }
                memcpy(packed_weights + out_offs, &buf8[0][0], sizeof(buf8));
              }
              out_offs += sizeof(buf8);
            }
            else {  // Half float 16-bit weights
              if (out_offs + sizeof(buf16) <= *packed_weights_size) {
                const uint16_t *w = (const uint16_t*)weights;
                for (int c = c_start; c < c_stop; ++c) {
                  const int t = c & 7;
                  const int x = ((c & 63) >> 3) % 3;
                  const int y = ((c & 63) >> 3) / 3;
                  buf16[11 - (t >> 1) * 3 - y][(t & 1) * 3 + x] = w[m * s0 + c * s1];
                }
                memcpy(packed_weights + out_offs, &buf16[0][0], sizeof(buf16));
              }
              out_offs += sizeof(buf16);
            }
          }
        }
      }
      break;
    }
    default:
    {
      SET_ERR("Unsupported kernel configuration %dx%d", kx, ky);
      return -1;
    }
  }

  if (out_offs & 15) {  // zero-pad output to 16-bytes
    const int d = 16 - (out_offs & 15);
    if (out_offs + d <= *packed_weights_size) {
      memset(packed_weights + out_offs, 0, d);
    }
    out_offs += d;
  }

  if ((!retval) && (*packed_weights_size) && (*packed_weights_size < out_offs)) {
    SET_ERR("Not all weights were filled: provided buffer size %zu while %zu is required", *packed_weights_size, out_offs);
    retval = -1;
  }

  *packed_weights_size = out_offs;
  return retval;
}
