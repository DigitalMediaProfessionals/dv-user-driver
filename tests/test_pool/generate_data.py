#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function
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
        parser.add_argument("-f", "--float", action="store_true",
                            help="Generate float random weights")
        args = parser.parse_args()
        self.generate(7, 7, 1024,
                      2, (7, 7), (1, 1), (0, 0, 0, 0), args)
        self.generate(960 // 4, 384 // 4, 16,
                      1, (2, 2), (2, 2), (0, 0, 0, 0), args)
        self.generate(960 // 2, 384 // 2, 16,
                      1, (2, 2), (2, 2), (0, 0, 0, 0), args)
        self.generate(960, 384, 16,
                      1, (2, 2), (2, 2), (0, 0, 0, 0), args)

    def generate(self, width, height, n_channels,
                 pool_type, pool_size, pool_stride, pool_pad, args):
        """Generates test data for LRN layer and invokes caffe
        to generate gold output.

        Parameters:
            width: input width.
            height: input height.
            n_channels: number of channels in input.
        """
        if pool_pad[0] != pool_pad[1] or pool_pad[2] != pool_pad[3]:
            return  # caffe only supports symmetric padding

        try:
            os.mkdir("data")
        except OSError:
            pass
        s_dir = "data/%dx%dx%d_t%d" % (width, height, n_channels, pool_type)
        try:
            os.mkdir(s_dir)
        except OSError:
            pass

        prefix = ("%s/%dx%d_s%dx%d_p%dx%dx%dx%d" %
                  (s_dir, pool_size[0], pool_size[1],
                   pool_stride[0], pool_stride[1],
                   pool_pad[0], pool_pad[1], pool_pad[2], pool_pad[3]))

        numpy.random.seed(12345)

        if args.float:
            values = numpy.random.uniform(
                -1.0, 1.0, 1001).astype(numpy.float32)
        else:
            values = numpy.array([-2.0, -1.0, 0.0, 1.0, 2.0],
                                 dtype=numpy.float32)

        input = numpy.random.choice(
            values, width * height * n_channels).astype(numpy.float16)
        input.tofile("%s.i.bin" % prefix)

        caffe.set_mode_cpu()

        with open("data/test.prototxt", "w") as fout:
            fmt = (n_channels, height, width,
                   "AVE" if pool_type == 2 else "MAX",
                   pool_size[1], pool_size[0],
                   pool_stride[1], pool_stride[0],
                   pool_pad[2], pool_pad[0])
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
  name: "pool1"
  type: "Pooling"
  bottom: "data"
  top: "pool1"
  pooling_param {
    pool: %s
    kernel_h: %d
    kernel_w: %d
    stride_h: %d
    stride_w: %d
    pad_h: %d
    pad_w: %d
  }
}
""" % fmt)

        net = caffe.Net("data/test.prototxt", caffe.TEST)

        net.blobs["data"].data[0, :, :, :] = input.astype(
            numpy.float32).reshape(n_channels, height, width)
        del input

        results = net.forward()
        output = results["pool1"].copy()

        output.astype(numpy.float16).tofile("%s.o.bin" % prefix)

        print("Successfully generated test input/output for %s" %
              prefix)

        try:
            os.unlink("data/test.prototxt")
        except OSError:
            pass


if __name__ == "__main__":
    Main()
