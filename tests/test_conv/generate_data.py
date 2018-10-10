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
from itertools import product
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
        parser.add_argument("-d", "--dilated", action="store_true",
                            help="Generate only tests for dilated convolutions")
        args = parser.parse_args()

        self.generate_tests(args)

    def generate_tests(self, args):
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
                for stride, act, x, y, c, m, dw, dil in product(
                        [(1, 1), (2, 2)][0:], [0],
                        [5, 7, 9, 15][0:], [5, 7, 9, 15][0:],
                        [1, 3, 9, 64, 65][0:], [1, 3, 9, 64, 65][0:],
                        [False, True], [1, 2]):
                    dil = max(dil, 1)
                    kxfull = (kx - 1) * dil + 1
                    kyfull = (ky - 1) * dil + 1
                    pads = [(kxfull >> 1, kyfull >> 1,
                             kxfull >> 1, kyfull >> 1)]
                    if dil == 1 and pads[0] != (0, 0, 0, 0):
                        pads.append((0, 0, 0, 0))
                    for pad in pads:
                        self.generate(
                            x, y, roundup(c, args.multiple),
                            kx, ky, roundup(m, args.multiple),
                            pad, stride, act, dw, dil, args)

    def get_ox(self, width, kx, pad_left, pad_right, stride):
        return (pad_left + width + pad_right - kx) // stride + 1

    def generate(self, width, height, n_channels, kx, ky, n_kernels,
                 pad_ltrb, stride_xy, activation, dw, dil, args):
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
            dw: use depth-wise (caffe's group=n_channels == n_kernels).
        """
        if dw and n_kernels != n_channels:  # check if dw is applicable
            return

        if dw and dil > 1:
            return

        if (dil > 1 and (width < (dil * (kx - 1) + 1) or
                         height < (dil * (ky - 1) + 1) or
                         kx % 2 == 0 or ky % 2 == 0)):
            return

        if dil < 2 and args.dilated:
            return

        if ((pad_ltrb[0] + width + pad_ltrb[2] < kx) or
                (pad_ltrb[1] + height + pad_ltrb[3] < ky)):
            return

        try:
            os.mkdir("data")
        except OSError:
            pass
        s_dir = "data/%dx%dx%d_t%d" % (width, height, n_channels,
                                       1 if dw else (0 if dil < 2 else dil))
        try:
            os.mkdir(s_dir)
        except OSError:
            pass

        s_pad = "%dx%dx%dx%d" % pad_ltrb
        s_stride = "%dx%d" % stride_xy

        prefix = ("%s/%dx%dx%d_pad%s_stride%s_act%d" %
                  (s_dir, kx, ky, n_kernels, s_pad, s_stride, activation))

        numpy.random.seed(1234)

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

        weights_dim_1 = 1 if dw else n_channels

        assert len(quant) == 256
        assert numpy.count_nonzero(numpy.isnan(quant)) == 0
        i_weights = numpy.random.randint(
            0, 256, kx * ky * weights_dim_1 * n_kernels).astype(
                numpy.uint8)
        i_weights.tofile("%s.w.bin" % prefix)
        weights = quant[i_weights].copy().reshape(
            n_kernels, weights_dim_1, ky, kx)
        assert numpy.count_nonzero(numpy.isnan(weights)) == 0
        del i_weights
        del quant

        caffe.set_mode_cpu()

        s_kern = ("kernel_size: %d" % kx if kx == ky
                  else "kernel_w: %d\n    kernel_h: %d" % (kx, ky))

        s_pad = ("pad: %d" % pad_ltrb[0] if min(pad_ltrb) == max(pad_ltrb) else
                 "pad_w: %d\n    pad_h: %d" % (max(pad_ltrb[0], pad_ltrb[2]),
                                               max(pad_ltrb[1], pad_ltrb[3])))

        s_stride = ("stride: %d" % stride_xy[0] if stride_xy[0] == stride_xy[1]
                    else "stride_w: %d\n    stride_h: %d" % stride_xy)

        s_dw = "\n    group: %d" % n_channels if dw else ""

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
    dilation: %d
    num_output: %d
    %s
    %s
    %s%s
  }
}
""" % (n_channels, height, width, dil, n_kernels,
       s_pad, s_kern, s_stride, s_dw))
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
            numpy.float32).reshape(n_kernels, weights_dim_1, ky, kx)
        del weights
        net.params["conv1"][1].data[:] = bias.astype(numpy.float32)
        # del bias

        net.blobs["data"].data[0, :, :, :] = input.astype(
            numpy.float32).reshape(n_channels, height, width)
        del input

        results = net.forward()
        output = results["conv1"].copy()

        ox = self.get_ox(width, (kx - 1) * dil + 1, pad_ltrb[0], pad_ltrb[2], stride_xy[0])
        oy = self.get_ox(height, (ky - 1) * dil + 1, pad_ltrb[1], pad_ltrb[3], stride_xy[1])
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
