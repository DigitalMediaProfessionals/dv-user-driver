/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief Weights-packing helper functions.
 */
#include <set>

#include "common.h"


/// @brief Writes bias to packed weights buffer.
static inline void write_bias(int& m_start, const int& m_stop, size_t &out_offs, size_t *output_size, uint8_t *output, const uint16_t *bias) {
  size_t n = (m_stop - m_start) << 1;
  if (out_offs + n <= *output_size) {
    memcpy(output + out_offs, bias + m_start, n);
  }
  out_offs += n;
  const int bias_pad = m_start + 8 - m_stop;
  if (bias_pad > 0) {
    n = bias_pad << 1;
    if (out_offs + n <= *output_size) {
      memset(output + out_offs, 0, n);
    }
    out_offs += n;
  }
}


/// @brief Packs convolution layer weights and biases into output array.
/// @param n_channels Number of input channels.
/// @param kx Kernel width.
/// @param ky Kernel height.
/// @param n_kernels Number of output channels.
/// @param quant_map Quantization table for weights (but not bias), when NULL, no quantization is assumed.
/// @param weights If quant_map is NULL, array of half precision floating point weights in NCHW format, else array of 1-byte indices.
/// @param bias Array of half precision floating point biases of size n_kernels.
/// @param output Output buffer for packed weights information (can be NULL if output_size is 0).
/// @param output_size On input, contains the size of the output buffer in bytes (can be 0, in such case it will be filled with the required output size), on output will contain the required output size.
/// @param msg Message with error description.
/// @param msg_size Size of msg in bytes.
/// @return 0 on success, non-zero otherwise.
extern "C"
int dmp_dv_pack_conv_weights(
    int n_channels, int kx, int ky, int n_kernels,
    const uint16_t quant_map[256],
    const void *weights, const uint16_t *bias,
    uint8_t *output, size_t *output_size) {

  // TODO: Optimize it to become O(n) (now it is completely non-optimal and very slow).
  const int p = std::max(kx, ky) | 1;  // next odd number

  if ((p > 7) || (std::min(kx, ky) <= 0)) {
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
  if (!output_size) {
    SET_ERR("output_size must not be NULL");
    return -1;
  }
  if ((!output) && (*output_size)) {
    SET_ERR("output is NULL but *output_size is non-zero");
    return -1;
  }

  int retval = 0;

  if (*output_size) {
    memset(output, 0, *output_size);
  }

  size_t out_offs = 0;
  size_t n;
  if (quant_map) {
    n = 512;
    if (out_offs + n <= *output_size) {
      memcpy(output + out_offs, quant_map, n);
    }
    out_offs += n;
  }

  // labels.shape = (n_kernels, n_channels, ky, kx)
  const int s2 = kx;
  const int s1 = ky * kx;
  const int s0 = n_channels * ky * kx;

  std::set<int> w_idx;  // for debugging purposes  // TODO: Remove it when optimization will be done.

  uint8_t buf8[12][6];
  memset(&buf8[0][0], 0, sizeof(buf8));

  switch (p) {
    case 7:
    {
      for (int m_start = 0; m_start < n_kernels; m_start += 8) {
        const int m_stop = std::min(m_start + 8, n_kernels);

        write_bias(m_start, m_stop, out_offs, output_size, output, bias);

        for (int c_start = 0; c_start < n_channels; c_start += 8) {
          const int c_stop = std::min(c_start + 8, n_channels);
          for (int m = m_start; m < m_stop; ++m) {
            for (int c = c_start; c < c_stop; ++c) {
              if (quant_map) {
                n = sizeof(buf8);
                if (out_offs + n <= *output_size) {
                  const uint8_t *w = (const uint8_t*)weights;
                  for (int y = 0; y < ky; ++y) {
                    for (int x = 0; x < 6; ++x) {
                      buf8[5 + y][x] = w[m * s0 + c * s1 + y * s2 + x];
                      w_idx.emplace(m * s0 + c * s1 + y * s2 + x);
                    }
                  }
                  buf8[2][5] = w[m * s0 + c * s1 + 0 * s2 + 6];
                  w_idx.emplace(m * s0 + c * s1 + 0 * s2 + 6);
                  for (int y = 0; y < 3; ++y) {
                    buf8[y][3] = w[m * s0 + c * s1 + (y + 1) * s2 + 6];
                    w_idx.emplace(m * s0 + c * s1 + (y + 1) * s2 + 6);
                    buf8[y][0] = w[m * s0 + c * s1 + (y + 4) * s2 + 6];
                    w_idx.emplace(m * s0 + c * s1 + (y + 4) * s2 + 6);
                  }
                  memcpy(output + out_offs, buf8, n);
                }
                out_offs += n;
              }
              else {
                SET_ERR("Branch is disabled for now");
                return -1;
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
        const int m_stop = std::min(m_start + 8, n_kernels);

        write_bias(m_start, m_stop, out_offs, output_size, output, bias);

        for (int c_start = 0; c_start < n_channels; c_start += 8) {
          const int c_stop = std::min(c_start + 8, n_channels);
          for (int m = m_start; m < m_stop; ++m) {
            if (quant_map) {
              n = sizeof(buf8);
              const uint8_t *w = (const uint8_t*)weights;
              for (int c = c_start; c < c_stop; ++c) {
                const int t = c % 2;
                if ((t == 0) && (c == c_stop - 1)) {
                  memset(&buf8[0][0], 0, sizeof(buf8));
                }
                if (out_offs + n <= *output_size) {
                  for (int y = 0; y < ky; ++y) {
                    for (int x = 0; x < kx; ++x) {
                      buf8[7 - t * 6 + y][x] = w[m * s0 + c * s1 + y * s2 + x];
                      w_idx.emplace(m * s0 + c * s1 + y * s2 + x);
                    }
                  }
                }
                if ((t == 1) || ((t == 0) && (c == c_stop - 1))) {
                  if (out_offs + n <= *output_size) {
                    memcpy(output + out_offs, buf8, n);
                  }
                  out_offs += n;
                }
              }
            }
            else {
              SET_ERR("Branch is disabled for now");
              return -1;
            }
          }
        }
      }
      break;
    }
    case 3:
    {
      for (int m_start = 0; m_start < n_kernels; m_start += 8) {
        const int m_stop = std::min(m_start + 8, n_kernels);

        write_bias(m_start, m_stop, out_offs, output_size, output, bias);

        for (int c_start = 0; c_start < n_channels; c_start += 8) {
          const int c_stop = std::min(c_start + 8, n_channels);
          for (int m = m_start; m < m_stop; ++m) {
            if (quant_map) {
              if ((c_stop - c_start >= 1) && (c_stop - c_start <= 7)) {
                memset(&buf8[0][0], 0, sizeof(buf8));
              }
              n = sizeof(buf8);
              if (out_offs + n <= *output_size) {
                const uint8_t *w = (const uint8_t*)weights;
                for (int c = c_start; c < c_stop; ++c) {
                  const int t = c % 8;
                  for (int y = 0; y < ky; ++y) {
                    for (int x = 0; x < kx; ++x) {
                      buf8[9 - t / 2 * 3 + y][t % 2 * 3 + x] = w[m * s0 + c * s1 + y * s2 + x];
                      w_idx.emplace(m * s0 + c * s1 + y * s2 + x);
                    }
                  }
                }
                memcpy(output + out_offs, buf8, n);
              }
              out_offs += n;
            }
            else {
              SET_ERR("Branch is disabled for now");
              return -1;
            }
          }
        }
      }
      break;
    }
    case 1:
    {
      for (int m_start = 0; m_start < n_kernels; m_start += 8) {
        const int m_stop = std::min(m_start + 8, n_kernels);

        write_bias(m_start, m_stop, out_offs, output_size, output, bias);

        for (int c_start = 0; c_start < n_channels; c_start += 64) {
          const int c_stop = std::min(c_start + 64, n_channels);
          for (int m = m_start; m < m_stop; ++m) {
            if (quant_map) {
              if ((c_stop - c_start >= 1) && (c_stop - c_start <= 63)) {
                memset(&buf8[0][0], 0, sizeof(buf8));
              }
              n = sizeof(buf8);
              if (out_offs + n <= *output_size) {
                const uint8_t *w = (const uint8_t*)weights;
                for (int c = c_start; c < c_stop; ++c) {
                  const int t = c % 8;
                  const int x = c % 64 / 8 % 3;
                  const int y = c % 64 / 8 / 3;
                  buf8[11 - t / 2 * 3 - y][t % 2 * 3 + x] = w[m * s0 + c * s1];
                  w_idx.emplace(m * s0 + c * s1);
                }
                memcpy(output + out_offs, buf8, n);
              }
              out_offs += n;
            }
            else {
              SET_ERR("Branch is disabled for now");
              return -1;
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

  if ((!retval) && (*output_size) && (*output_size < out_offs)) {
    retval = -1;
  }

  if ((!retval) && (*output_size)) {  // sanity check for debugging purposes
    const int n_idx = n_channels * kx * ky * n_kernels;
    for (int i = 0; i < n_idx; ++i) {
      auto it = w_idx.find(i);
      if (it == w_idx.end()) {
        SET_ERR("Index %d was not handled (must handle all %d indices)", i, n);
        return -1;
      }
    }
    if ((int)w_idx.size() != n_idx) {
      SET_ERR("Handled different number of indices than expected: %d vs %d", (int)w_idx.size(), n_idx);
      return EINVAL;
    }
  }

  *output_size = out_offs;
  return retval;
}
