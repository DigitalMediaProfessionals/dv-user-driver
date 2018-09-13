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
Generates data for testing fully connected layer.
"""
import argparse
import numpy
import os

import caffe


def roundup(a, b):
    d = a % b
    return a + (b - d) if d else a


ACT_MAP = {
    0: "",
    1: "ReLU",
    2: "TanH",
    4: "Sigmoid"
}


class Main(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--float", action="store_true",
                            help="Force generated channels to be multiple "
                                 "of this value (default: 1)")
        args = parser.parse_args()

        self.generate_fc(args)

    def generate_fc(self, args):
        # Generate 1D inputs
        for input_size in (512, 1024):
            for output_size in (512, 1024):
                for act in ACT_MAP.keys():
                    self.generate((input_size, 1, 1), output_size, act, args)

        # Generate 3D inputs
        for c in (8, 128):
            for h in (8, 16):
                for w in (8, 16):
                    for output_size in (256, 512):
                        for act in (0,):
                            self.generate((c, h, w), output_size, act, args)

        # Generate big inputs
        self.generate((16383, 1, 1), 512, 0, args)
        self.generate((16384, 1, 1), 512, 0, args)

    def generate(self, input_shape, output_size, activation, args):
        """Generates test data for fully connected layer and invokes caffe
        to generate gold output.

        Parameters:
            input_shape: input shape in CHW format.
            output_size: size of the output in elements.
            activation: activation function: 0 = None, 1 = ReLU, 2 = Tanh,
                        3 = Leaky ReLU, 4 = Sigmoid, 5 = PReLU
                        (PReLU must be used with POST-OP=1)).
        """
        assert len(input_shape) == 3
        assert all(x > 0 for x in input_shape)
        input_size = int(numpy.prod(input_shape))
        if input_size > 16484 or output_size > 16484:
            return

        try:
            os.mkdir("data")
        except OSError:
            pass
        s_dir = "data/%dx%dx%d" % input_shape
        try:
            os.mkdir(s_dir)
        except OSError:
            pass

        prefix = "%s/%d_act%d" % (s_dir, output_size, activation)

        numpy.random.seed(12345)

        if args.float:
            values = numpy.random.uniform(
                -1.0, 1.0, 1001).astype(numpy.float32)
        else:
            values = numpy.array([-2.0, -1.0, 0.0, 1.0, 2.0],
                                 dtype=numpy.float32)

        bias = numpy.random.choice(values, output_size).astype(numpy.float16)
        bias.tofile("%s.b.bin" % prefix)

        input = numpy.random.choice(values, input_size).reshape(
            input_shape).astype(numpy.float16)
        input.tofile("%s.i.bin" % prefix)

        quant = numpy.random.choice(values, 256).astype(numpy.float16)
        quant[0] = 0
        quant.tofile("%s.q.bin" % prefix)

        assert len(quant) == 256
        assert numpy.count_nonzero(numpy.isnan(quant)) == 0
        i_weights = numpy.random.randint(
            0, 256, input_size * output_size).astype(numpy.uint8)
        i_weights.tofile("%s.w.bin" % prefix)
        weights = quant[i_weights].copy().reshape(
            output_size, input_shape[0], input_shape[1], input_shape[2])
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
  name: "fc1"
  type: "InnerProduct"
  bottom: "data"
  top: "fc1"
  inner_product_param {
    num_output: %d
  }
}
""" % (input_shape[0], input_shape[1], input_shape[2], output_size))
            if activation:
                fout.write("""
layer {
  name: "fc1/ACT"
  type: "%s"
  bottom: "fc1"
  top: "fc1"
}
""" % ACT_MAP[activation])

        net = caffe.Net("data/test.prototxt", caffe.TEST)
        net.params["fc1"][0].data[:] = weights.astype(
            numpy.float32).reshape(output_size, input_size)
        del weights
        net.params["fc1"][1].data[:] = bias.astype(numpy.float32)
        del bias

        net.blobs["data"].data[0, :, :, :] = input.astype(numpy.float32)
        del input

        results = net.forward()
        output = results["fc1"].copy()

        assert output.shape == (1, output_size), \
            "output.shape = %s" % str(output.shape)

        output.astype(numpy.float16).tofile("%s.o.bin" % prefix)

        print("Successfully generated test input/weights/output for %s" %
              prefix)

        try:
            os.unlink("data/test.prototxt")
        except OSError:
            pass


if __name__ == "__main__":
    Main()
