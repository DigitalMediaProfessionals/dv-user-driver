#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Copyright 2018 Digital Media Professionals Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
"""
Generates data for testing convolutional layer.
"""
import argparse
import numpy
import os

import caffe


def roundup(a, b):
    d = a % b
    return a + (b - d) if d else a


class Main(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-b", "--big", action="store_true",
                            help="Generate big configuration")
        parser.add_argument("-f", "--float", action="store_true",
                            help="Generate float random weights")
        parser.add_argument("-s", "--square", action="store_true",
                            help="Generate only square sizes")
        parser.add_argument("-o", "--odd", action="store_true",
                            help="Generate only odd sizes")
        parser.add_argument("-m", "--multiple", type=int, default=1,
                            help="Force generated channels to be multiple "
                                 "of this value (default: 1)")
        args = parser.parse_args()

        if args.big:
            self.generate_big(args)
        else:
            self.generate_small(args)

    def generate_small(self, args):
        if args.odd:
            kxx = (1, 3, 5, 7)
        else:
            kxx = (1, 2, 3, 4, 5, 6, 7)
        for kx in kxx:
            if args.square:
                kyy = (kx,)
            else:
                kyy = kxx
            for ky in kyy:
                pad = (kx >> 1, ky >> 1, kx >> 1, ky >> 1)
                for stride in ((1, 1),):
                    for act in (0,):  # 0, 1, 3, 5: none, tanh, sigmoid, elu
                        for x in (3, 17):
                            for y in (3, 17):
                                for c in (3, 65):
                                    for m in (3, 65):
                                        self.generate(
                                            x, y, roundup(c, args.multiple),
                                            kx, ky, roundup(m, args.multiple),
                                            pad, stride, act, args)

    def generate_big(self, args):
        # Generate tests without activation function
        if args.odd:
            kxx = (1, 3, 5, 7)
        else:
            kxx = (1, 2, 3, 4, 5, 6, 7)
        for kx in kxx:
            if args.square:
                kyy = (kx,)
            else:
                kyy = kxx
            for ky in kyy:
                pad = (kx >> 1, ky >> 1, kx >> 1, ky >> 1)
                for stride in ((1, 1), (2, 2)):
                    for act in (0,):  # 0, 1, 3, 5: none, tanh, sigmoid, elu
                        for x in (1, 3, 5, 8, 17, 128):
                            for y in (1, 3, 5, 8, 17, 128):
                                for c in (1, 3, 9, 16, 65):
                                    for m in (1, 3, 9, 16, 65):
                                        self.generate(
                                            x, y, roundup(c, args.multiple),
                                            kx, ky, roundup(m, args.multiple),
                                            pad, stride, act, args)

        # Generate tests with activation function
        if args.odd:
            kxx = (1, 3, 5, 7)
        else:
            kxx = (1, 2, 3, 4, 5, 6, 7)
        for kx in kxx:
            if args.square:
                kyy = (kx,)
            else:
                kyy = kxx
            for ky in kyy:
                pad = (kx >> 1, ky >> 1, kx >> 1, ky >> 1)
                for stride in ((1, 1), (2, 2)):
                    for act in (1, 3, 5):  # 0, 1, 3, 5: none, tanh, sigmoid, elu
                        for x in (11, 128):
                            for y in (11, 128):
                                for c in (1, 3, 9, 32):
                                    for m in (1, 3, 9, 32):
                                        self.generate(
                                            x, y, roundup(c, args.multiple),
                                            kx, ky, roundup(m, args.multiple),
                                            pad, stride, act, args)

    def get_ox(self, width, kx, pad_left, pad_right, stride):
        return (pad_left + width + pad_right - kx) // stride + 1

    def generate(self, width, height, n_channels, kx, ky, n_kernels,
                 pad_ltrb, stride_xy, activation, args):
        """Generates test data for convolutional layer and invokes caffe
        to generate gold output.

        Parameters:
            width: input width.
            height: input height.
            n_channels: number of channels in input.
            kx: kernel width.
            ky: kernel height.
            n_kernels: number of convolutional kernels
                       (aka number of channels in output).
            pad_ltrb: padding (left, top, right, bottom).
            stride_xy: stride (x, y).
            activation: activation function (0 - none, 1 - tanh,
                        3 - sigmoid, 5 - elu).
        """
        if pad_ltrb[0] != pad_ltrb[2] or pad_ltrb[1] != pad_ltrb[3]:
            raise ValueError("Caffe doesn't support padding "
                             "(left, top, right, bottom): %s" % pad_ltrb)

        try:
            os.mkdir("data")
        except OSError:
            pass
        s_dir = "data/%dx%dx%d" % (width, height, n_channels)
        try:
            os.mkdir(s_dir)
        except OSError:
            pass

        s_pad = "%dx%dx%dx%d" % pad_ltrb
        s_stride = "%dx%d" % stride_xy

        prefix = ("%s/%dx%dx%d_pad%s_stride%s_act%d" %
                  (s_dir, kx, ky, n_kernels, s_pad, s_stride, activation))

        numpy.random.seed(12345)

        if args.float:
            values = numpy.random.uniform(
                -1.0, 1.0, 1001).astype(numpy.float32)
        else:
            values = numpy.array([-2.0, -1.0, 0.0, 1.0, 2.0],
                                 dtype=numpy.float32)

        bias = numpy.random.choice(values, n_kernels).astype(numpy.float16)
        bias.tofile("%s.b.bin" % prefix)

        input = numpy.random.choice(
            values, width * height * n_channels).astype(numpy.float16)
        input.tofile("%s.i.bin" % prefix)

        quant = numpy.random.choice(values, 256).astype(numpy.float16)
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

        s_kern = ("kernel_size: %d" % kx if kx == ky
                  else "kernel_w: %d\n    kernel_h: %d" % (kx, ky))

        s_pad = ("pad: %d" % pad_ltrb[0] if pad_ltrb[0] == pad_ltrb[1] else
                 "pad_w: %d\n    pad_h: %d" % (pad_ltrb[0], pad_ltrb[1]))

        s_stride = ("stride: %d" % stride_xy[0] if stride_xy[0] == stride_xy[1]
                    else "stride_w: %d\n    stride_h: %d" % stride_xy)

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
    %s
    %s
    %s
  }
}
""" % (n_channels, height, width, n_kernels, s_pad, s_kern, s_stride))
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

        ox = self.get_ox(width, kx, pad_ltrb[0], pad_ltrb[2], stride_xy[0])
        oy = self.get_ox(height, ky, pad_ltrb[1], pad_ltrb[3], stride_xy[1])
        assert output.shape == (1, n_kernels, oy, ox)

        output.astype(numpy.float16).tofile("%s.o.bin" % prefix)

        print("Successfully generated test input/weights/output for %s" %
              prefix)

        try:
            os.unlink("data/test.prototxt")
        except OSError:
            pass


if __name__ == "__main__":
    Main()