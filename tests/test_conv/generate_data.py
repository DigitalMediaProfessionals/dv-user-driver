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
        parser.add_argument("--float", action="store_true",
                            help="Generate float random weights")
        parser.add_argument("--square", action="store_true",
                            help="Generate only square sizes")
        parser.add_argument("--odd", action="store_true",
                            help="Generate only odd sizes")
        parser.add_argument("--multiple", type=int, default=1,
                            help="Force generated channels to be multiple "
                                 "of this value (default: 1)")
        parser.add_argument("--dilated", action="store_true",
                            help="Generate only tests for dilated "
                            "convolutions")
        parser.add_argument("--deconv", action="store_true",
                            help="Generate only tests for deconvolutions")
        parser.add_argument("--seed", type=int, default=1234,
                            help="Random seed")
        parser.add_argument("--bz", action="store_true",
                            help="Set bias to zero")
        args = parser.parse_args()

        self.generate_tests(args)

    def generate_tests(self, args):
        for deconv in (False, True):
            self.generate_deconv(deconv, args)

    def generate_deconv(self, deconv, args):
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
                        [(1, 1), (2, 2)], [0],
                        [1, 9, 15], [1, 9, 15],
                        [1, 3, 9, 64, 65], [1, 3, 9, 64, 65],
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
                            pad, stride, act, dw, dil, deconv, args)

        # Test activations
        for kx in (1, 3, 5, 7):
            kyy = (kx,)
            for ky in kyy:
                for stride, act, x, y, c, m, dw, dil in product(
                        [(1, 1), (2, 2)], [1, 3, 4, 5],
                        [1, 9, 15], [1, 9, 15],
                        [1, 65], [1, 65],
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
                            pad, stride, act, dw, dil, deconv, args)

        # Generate big input
        kx = 3
        ky = 3
        width = 128
        height = 64
        n_channels = 64
        n_kernels = 32
        pad_ltrb = (1, 1, 1, 1)
        stride_xy = (2, 2)
        activation = 0
        dw = False
        dil = 1
        self.generate(
            width, height, n_channels, kx, ky, n_kernels,
            pad_ltrb, stride_xy, activation, dw, dil, deconv, args)

    def get_ox(self, width, kx, pad_left, pad_right, stride, deconv):
        return (pad_left + ((width - 1) * stride + 1 if deconv else width) +
                pad_right - kx) / (1 if deconv else stride) + 1

    def generate(self, width, height, n_channels, kx, ky, n_kernels,
                 pad_ltrb, stride_xy, activation, dw, dil, deconv, args):
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
                        3 - sigmoid, 4 - prelu, 5 - elu).
            dw: use depth-wise (caffe's group=n_channels == n_kernels).
        """
        dil = max(1, dil)

        if not deconv and args.deconv:
            return

        if deconv and dil > 1 and max(stride_xy) > 1:
            return

        if dw and n_kernels != n_channels:  # check if dw is applicable
            return

        if dil > 1 and (dw or activation == 4):  # check dilated limitations
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

        ox = self.get_ox(width, (kx - 1) * dil + 1,
                         pad_ltrb[0], pad_ltrb[2], stride_xy[0], deconv)
        oy = self.get_ox(height, (ky - 1) * dil + 1,
                         pad_ltrb[1], pad_ltrb[3], stride_xy[1], deconv)
        if ox < 1 or oy < 1:
            return
        if deconv:
            ox2 = self.get_ox(ox, (kx - 1) * dil + 1,
                              pad_ltrb[0], pad_ltrb[2], stride_xy[0], 0)
            oy2 = self.get_ox(oy, (ky - 1) * dil + 1,
                              pad_ltrb[1], pad_ltrb[3], stride_xy[1], 0)
            if ox2 != width or oy2 != height:
                return  # deconvolution configuration is invalid

        try:
            os.mkdir("data")
        except OSError:
            pass
        s_dir = ("data/%dx%dx%d_t%dd%d" %
                 (width, height, n_channels,
                  1 if dw else (0 if dil < 2 else dil),
                  int(deconv)))
        try:
            os.mkdir(s_dir)
        except OSError:
            pass

        s_pad = "%dx%dx%dx%d" % pad_ltrb
        s_stride = "%dx%d" % stride_xy

        prefix = ("%s/%dx%dx%d_pad%s_stride%s_act%d" %
                  (s_dir, kx, ky, n_kernels, s_pad, s_stride, activation))

        numpy.random.seed(args.seed)

        if args.float:
            values = numpy.random.uniform(
                -1.0, 1.0, 1001).astype(numpy.float32)
        else:
            values = numpy.array([-2.0, -1.0, 1.0, 2.0],
                                 dtype=numpy.float32)

        bias = numpy.random.choice(values, n_kernels).astype(numpy.float16)
        if args.bz:
            bias[:] = 0
        bias.tofile("%s.b.bin" % prefix)

        prelu = numpy.random.choice(values, n_kernels).astype(numpy.float16)

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
            1, 256, kx * ky * weights_dim_1 * n_kernels).astype(
                numpy.uint8)
        if deconv:
            iw = numpy.rot90(
                i_weights.reshape(
                    (weights_dim_1, n_kernels, ky, kx) if not dw else
                    (n_kernels, weights_dim_1, ky, kx)),
                2, (2, 3))
            iw = numpy.moveaxis(iw, [0, 1, 2, 3], [1, 0, 2, 3])
            iw = iw.ravel()
            print("ROTATED")
        else:
            iw = i_weights
        iw.tofile("%s.w.bin" % prefix)
        del iw
        weights = quant[i_weights].copy()
        if deconv and not dw:
            weights.shape = weights_dim_1, n_kernels, ky, kx
        else:
            weights.shape = n_kernels, weights_dim_1, ky, kx
        assert numpy.count_nonzero(numpy.isnan(weights)) == 0
        del i_weights
        del quant

        caffe.set_mode_cpu()

        s_type = "Deconvolution" if deconv else "Convolution"

        s_kern = ("kernel_size: %d" % kx if kx == ky
                  else "kernel_w: %d\n    kernel_h: %d" % (kx, ky))

        s_pad = ("pad: %d" % pad_ltrb[0] if min(pad_ltrb) == max(pad_ltrb) else
                 "pad_w: %d\n    pad_h: %d" % (max(pad_ltrb[0], pad_ltrb[2]),
                                               max(pad_ltrb[1], pad_ltrb[3])))

        s_stride = ("stride: %d" % stride_xy[0] if stride_xy[0] == stride_xy[1]
                    else "stride_w: %d\n    stride_h: %d" % stride_xy)

        s_dw = "\n    group: %d" % n_channels if dw else ""
        s_dil = "\n    dilation: %d" % dil if dil > 1 else ""

        with open("data/test.prototxt", "w") as fout:
            s = """name: "Test"
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
  type: "%s"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: %d
    %s
    %s
    %s%s%s
  }
}
"""
            fout.write(s % (n_channels, height, width, s_type, n_kernels,
                            s_pad, s_kern, s_stride, s_dw, s_dil))

            if activation == 5:
                fout.write("""
layer {
  name: "conv1/ELU"
  type: "ELU"
  bottom: "conv1"
  top: "conv1"
}
""")
            if activation == 4:
                fout.write("""
layer {
  name: "conv1/PReLU"
  type: "PReLU"
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
        net.params["conv1"][0].data[:] = weights.astype(numpy.float32)
        del weights
        net.params["conv1"][1].data[:] = bias.astype(numpy.float32)
        del bias
        if activation == 4:
            net.params["conv1/PReLU"][0].data[:] = prelu.astype(numpy.float32)
            prelu.tofile("%s.prelu.bin" % prefix)

        net.blobs["data"].data[0, :, :, :] = input.astype(
            numpy.float32).reshape(n_channels, height, width)
        del input

        results = net.forward()
        output = results["conv1"].copy()

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
