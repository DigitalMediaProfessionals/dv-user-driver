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
/// @details The function packs weights in Caffe NCHW format to the DV input format WHC8
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
    size_t weights_size = (size_t)c_input * c_output;
    if (!quant_map) {
      weights_size <<= 1;
    }
    if (out_offs + weights_size <= *packed_weights_size) {
      memcpy(packed_weights + out_offs, weights, weights_size);
    }
    out_offs += weights_size;
    output_size = c_output;
  }
  else {
    SET_ERR("CHW input/output is not yet supported");
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
