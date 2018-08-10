/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief Weights-packing helper functions for fully connected layer.
 */

#include "common.h"


/// @brief Packs fully connected layer weights and biases into output array.
/// @param n_channels Number of input channels (or input size in case of 1D input).
/// @param height Input height (set to 1 in case of 1D input).
/// @param width Input width (set to 1 in case of 1D input).
/// @param output_size Output size in elements.
/// @param quant_map Quantization table for weights (but not bias), can be NULL.
/// @param weights If quant_map is NULL, array of half precision floating point weights in NCHW format (N=output_size), else array of 1-byte indices.
/// @param bias Array of half precision floating point biases of size output_size.
/// @param packed_weights Output buffer for packed weights information (can be NULL if packed_weights_size is 0).
/// @param packed_weights_size On input, contains the size of the packed_weights buffer in bytes (can be 0, in such case it will be filled with the required buffer size), on output will contain the required buffer size.
/// @return 0 on success, non-zero otherwise.
/// @details It is thread-safe.
int dmp_dv_pack_fc_weights(
    int n_channels, int height, int width, int output_size,
    const uint16_t quant_map[256],
    const void *weights, const uint16_t *bias,
    uint8_t *packed_weights, size_t *packed_weights_size) {

  if (n_channels <= 0) {
    SET_ERR("Number of input channels must be positive, got %d", n_channels);
    return -1;
  }
  if (height <= 0) {
    SET_ERR("Height must be positive, got %d", height);
    return -1;
  }
  if (width <= 0) {
    SET_ERR("Width must be positive, got %d", width);
    return -1;
  }
  if (output_size <= 0) {
    SET_ERR("Output size must be positive, got %d", output_size);
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

  size_t out_offs = 0;
  if (quant_map) {
    if (out_offs + 512 <= *packed_weights_size) {
      memcpy(packed_weights + out_offs, quant_map, 512);
    }
    out_offs += 512;
  }

  if ((height == 1) && (width == 1)) {  // 1D input
    size_t weights_size = (size_t)n_channels * output_size;
    if (!quant_map) {
      weights_size <<= 1;
    }
    if (out_offs + weights_size <= *packed_weights_size) {
      memcpy(packed_weights + out_offs, weights, weights_size);
    }
    out_offs += weights_size;
  }
  else {
    SET_ERR("CHW input is not yet supported");
    return -1;
  }

  if (out_offs + output_size * 2 <= *packed_weights_size) {
    memcpy(packed_weights + out_offs, bias, output_size * 2);
  }
  out_offs += output_size * 2;

  int res = 0;
  if ((*packed_weights_size) && (*packed_weights_size < out_offs)) {
    SET_ERR("Not all weights were filled: provided buffer size %zu while %zu is required", *packed_weights_size, out_offs);
    res = -1;
  }

  *packed_weights_size = out_offs;
  return res;
  /*
    centers, labels = calc_kmeans(node._weight)
    index8 = labels.astype(np.uint8)
    bias16 = node._bias.astype(np.float16)

    centers.tofile(of)
    if conv_node is not None:
        if len(conv_node._output_dim) == 3:
            w, h, c = conv_node._output_dim
        elif len(conv_node._output_dim) == 1:
            w, h, c = 1, 1, conv_node._output_dim[0]
        m = node._param.num_output
        if w != 1 or h != 1:
            logging.info('Reordering FC weight for node: %s.', node._name)
            index8.shape = (m, c, h, w)
            for n in range(m):
                for d in range(0, c, 8):
                    e = d + 8 if d + 8 < c else c
                    tr_index8 = index8[n, d:e, :, :].transpose(2, 1, 0)
                    index8[n, d:e, :, :] = tr_index8.reshape(e - d, h, w)
    index8.tofile(of)
    bias16.tofile(of)
   */
}
