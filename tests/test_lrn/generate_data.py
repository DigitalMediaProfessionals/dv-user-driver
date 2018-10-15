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
        parser.add_argument("-f", "--float", action="store_true",
                            help="Generate float random weights")
        args = parser.parse_args()
        self.generate(56, 56, 64, args)
        self.generate(1, 1, 32, args)
        self.generate(2, 2, 16, args)
        self.generate(2, 2, 64, args)
        self.generate(4, 4, 64, args)
        self.generate(64, 64, 64, args)
        self.generate(53, 53, 64, args)

    def generate(self, width, height, n_channels, args):
        """Generates test data for LRN layer and invokes caffe
        to generate gold output.

        Parameters:
            width: input width.
            height: input height.
            n_channels: number of channels in input.
        """
        try:
            os.mkdir("data")
        except OSError:
            pass
        s_dir = "data/%dx%d" % (width, height)
        try:
            os.mkdir(s_dir)
        except OSError:
            pass

        prefix = "%s/%d" % (s_dir, n_channels)

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
  name: "lrn1"
  type: "LRN"
  bottom: "data"
  top: "lrn1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
""" % (n_channels, height, width))

        net = caffe.Net("data/test.prototxt", caffe.TEST)

        net.blobs["data"].data[0, :, :, :] = input.astype(
            numpy.float32).reshape(n_channels, height, width)
        del input

        results = net.forward()
        output = results["lrn1"].copy()

        output.astype(numpy.float16).tofile("%s.o.bin" % prefix)

        print("Successfully generated test input/output for %s" %
              prefix)

        try:
            os.unlink("data/test.prototxt")
        except OSError:
            pass


if __name__ == "__main__":
    Main()
