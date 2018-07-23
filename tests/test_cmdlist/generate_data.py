#!/usr/bin/python3
"""
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------

Generates weights, input and computes output.
"""
import numpy
import os

import caffe


class Main(object):
    def __init__(self):
        for kx in (1, 2, 3, 4, 5, 6, 7):
            ky = kx
            for act in (0, 1, 3, 5):  # none, tanh, sigmoid, elu
                self.generate(1, 1, 1, kx, ky, 1, kx >> 1, 1, act)
                self.generate(64, 32, 3, kx, ky, 32, kx >> 1, 1, act)
                self.generate(64, 32, 15, kx, ky, 31, kx >> 1, 1, act)

    def get_ox(self, width, kx, pad, stride):
        return (pad + width + pad - kx) // stride + 1

    def generate(self, width, height, n_channels, kx, ky, n_kernels,
                 pad, stride, activation):
        try:
            os.mkdir("data")
        except OSError:
            pass

        prefix = ("data/%dx%dx%d_%dx%dx%d_%d_%d_%d" %
                  (width, height, n_channels, kx, ky, n_kernels,
                   pad, stride, activation))

        bias = numpy.random.uniform(-1.0, 1.0, n_kernels).astype(numpy.float16)
        bias.tofile("%s.b.bin" % prefix)

        input = numpy.random.uniform(
            -1.0, 1.0, width * height * n_channels).astype(numpy.float16)
        input.tofile("%s.i.bin" % prefix)

        quant = numpy.random.uniform(-1.0, 1.0, 256).astype(numpy.float16)
        quant[0] = 0
        quant.tofile("%s.q.bin" % prefix)

        assert len(quant) == 256
        assert numpy.count_nonzero(numpy.isnan(quant)) == 0
        i_weights = numpy.random.randint(
            0, 256, kx * ky * n_channels * n_kernels).astype(numpy.uint8)
        i_weights.tofile("%s.w.bin" % prefix)
        weights = quant[i_weights].copy().reshape(
            n_kernels, n_channels, ky, kx)
        assert numpy.count_nonzero(numpy.isnan(weights)) == 0
        del i_weights
        del quant

        caffe.set_mode_cpu()

        with open("data/test.prototxt", "w") as fout:
            fout.write("""name: "Test"
state {
  phase: TEST
  level: 0
}
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: %d
      dim: %d
      dim: %d
    }
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: %d
    pad: %d
    kernel_size: %d
    stride: %d
  }
}
""" % (n_channels, height, width, n_kernels, pad, kx, stride))
            if activation == 5:
                fout.write("""
layer {
  name: "conv1/ELU"
  type: "ELU"
  bottom: "conv1"
  top: "conv1"
}
""")
            elif activation == 3:
                fout.write("""
layer {
  name: "conv1/Sigmoid"
  type: "Sigmoid"
  bottom: "conv1"
  top: "conv1"
}
""")
            elif activation == 1:
                fout.write("""
layer {
  name: "conv1/TanH"
  type: "TanH"
  bottom: "conv1"
  top: "conv1"
}
""")

        net = caffe.Net("data/test.prototxt", caffe.TEST)
        net.params["conv1"][0].data[:] = weights.astype(
            numpy.float32).reshape(n_kernels, n_channels, ky, kx)
        del weights
        net.params["conv1"][1].data[:] = bias.astype(numpy.float32)
        del bias

        net.blobs["data"].data[0, :, :, :] = input.astype(
            numpy.float32).reshape(n_channels, height, width)
        del input

        results = net.forward()
        output = results["conv1"].copy()

        ox = self.get_ox(width, kx, pad, stride)
        oy = self.get_ox(height, ky, pad, stride)
        assert output.shape == (1, n_kernels, oy, ox)

        output.astype(numpy.float16).tofile("%s.o.bin" % prefix)

        print("Successfully generated test input/weights/output for %s" %
              prefix)


if __name__ == "__main__":
    Main()
