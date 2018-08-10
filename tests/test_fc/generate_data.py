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
                            help="Force generated channels to be multiple "
                                 "of this value (default: 1)")
        args = parser.parse_args()

        self.generate_fc(args)

    def generate_fc(self, args):
        for input_size in (512, 1024):
            for output_size in (512, 1024):
                for act in (0,):
                    self.generate(input_size, output_size, act, args)

    def generate(self, input_size, output_size, activation, args):
        """Generates test data for fully connected layer and invokes caffe
        to generate gold output.

        Parameters:
            input_size: size of the input in elements.
            output_size: size of the output in elements.
            activation: activation function (0 - none, 1 - tanh,
                        3 - sigmoid, 5 - elu).
        """
        try:
            os.mkdir("data")
        except OSError:
            pass
        s_dir = "data/%d" % input_size
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

        input = numpy.random.choice(values, input_size).astype(numpy.float16)
        input.tofile("%s.i.bin" % prefix)

        quant = numpy.random.choice(values, 256).astype(numpy.float16)
        quant[0] = 0
        quant.tofile("%s.q.bin" % prefix)

        assert len(quant) == 256
        assert numpy.count_nonzero(numpy.isnan(quant)) == 0
        i_weights = numpy.random.randint(
            0, 256, input_size * output_size).astype(numpy.uint8)
        i_weights.tofile("%s.w.bin" % prefix)
        weights = quant[i_weights].copy().reshape(output_size, input_size)
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
      dim: 1
      dim: 1
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
""" % (input_size, output_size))
            if activation == 5:
                fout.write("""
layer {
  name: "fc1/ELU"
  type: "ELU"
  bottom: "fc1"
  top: "fc1"
}
""")
            elif activation == 3:
                fout.write("""
layer {
  name: "fc1/Sigmoid"
  type: "Sigmoid"
  bottom: "fc1"
  top: "fc1"
}
""")
            elif activation == 1:
                fout.write("""
layer {
  name: "fc1/TanH"
  type: "TanH"
  bottom: "fc1"
  top: "fc1"
}
""")

        net = caffe.Net("data/test.prototxt", caffe.TEST)
        net.params["fc1"][0].data[:] = weights.astype(
            numpy.float32).reshape(output_size, input_size)
        del weights
        net.params["fc1"][1].data[:] = bias.astype(numpy.float32)
        del bias

        net.blobs["data"].data[0, 0, 0, :] = input.astype(numpy.float32)
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
