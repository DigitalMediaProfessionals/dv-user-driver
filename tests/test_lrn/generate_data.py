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
import numpy as np
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
        self.generate(1, 1, 16, args)

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

        np.random.seed(12345)

        if args.float:
            values = np.random.normal(
                size=1001).astype(np.float32) * 100.0
        else:
            values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0],
                              dtype=np.float32)

        input = np.random.choice(
            values, width * height * n_channels).astype(np.float16)
        input.tofile("%s.i.bin" % prefix)

        if width == 1 and height == 1 and n_channels == 16:
            input = np.array(
                [0, 0, 0, 0, 18.6875, 1.31640625, 792.0, 49.53125,
                 20.46875, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.float16)
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

        inp = input.astype(
            np.float32).reshape(n_channels, height, width)
        net.blobs["data"].data[0, :, :, :] = inp
        del input

        results = net.forward()
        output = results["lrn1"][0].copy()

        self.compute_hw_output(inp, output)

        output.astype(np.float16).tofile("%s.o.bin" % prefix)

        print("Successfully generated test input/output for %s" %
              prefix)

        try:
            os.unlink("data/test.prototxt")
        except OSError:
            pass

    def compute_hw_output(self, inp, output):
        """Computes output by formula from current hardware implementation.
        """
        coef0 = np.array(
            [9.99985000e-01,
             9.99940004e-01,
             9.99760067e-01,
             9.99041074e-01,
             9.96177123e-01,
             9.84910181e-01,
             9.42656873e-01,
             8.08499843e-01,
             5.33567671e-01,
             2.53198576e-01,
             9.85382677e-02,
             3.57579523e-02], dtype=np.float32)

        coef1 = np.array(
            [-4.49960628e-05,
             -1.79937019e-04,
             -7.18993240e-04,
             -2.86395109e-03,
             -1.12669417e-02,
             -4.22533085e-02,
             -1.34157029e-01,
             -2.74932173e-01,
             -2.80369094e-01,
             -1.54660309e-01,
             -6.27803153e-02,
             -2.30311792e-02], dtype=np.float32)

        h_inv = np.array(
            [3.33333333e-01,
             8.33333333e-02,
             2.08333333e-02,
             5.20833333e-03,
             1.30208333e-03,
             3.25520833e-04,
             8.13802083e-05,
             2.03450521e-05,
             5.08626302e-06,
             1.27156576e-06,
             3.17891439e-07,
             7.94728597e-08], dtype=np.float32)

        x_sample = np.array(
            [1.00000000e+00,
             4.00000000e+00,
             1.60000000e+01,
             6.40000000e+01,
             2.56000000e+02,
             1.02400000e+03,
             4.09600000e+03,
             1.63840000e+04,
             6.55360000e+04,
             2.62144000e+05,
             1.04857600e+06,
             4.19430400e+06], dtype=np.float32)

        comb = h_inv * coef1

        sf_log2 = 5

        def get_addr(x):
            SIGNF_BITS = 23
            ADDR_STEP_LOG = 1
            expo_pt = int(np.float32(x).view(np.int32))
            expo = (0xff & (expo_pt >> SIGNF_BITS)) - 127
            expo += 2 * sf_log2
            if expo < 0:
                expo = 0  # if x < 1, use the first segment
            return expo >> ADDR_STEP_LOG if ADDR_STEP_LOG > 0 else expo

        def current_func(x):
            addr = get_addr(x)
            return (x * (1 << (2 * sf_log2)) -
                    x_sample[addr]) * comb[addr] + coef0[addr]

        assert len(inp.shape) == 3
        assert output.shape == inp.shape
        window = np.zeros(5, dtype=np.float32)
        for idx, vle in np.ndenumerate(inp):
            c_start = max(0, idx[0] - 2)
            c_end = min(inp.shape[0], idx[0] + 3)
            window[:] = 0
            window[c_start - idx[0] + 2:c_end - idx[0] + 2] = inp[
                c_start:c_end, idx[1], idx[2]]
            window *= 1.0 / (1 << sf_log2)
            np.square(window, window)
            x = window.sum()
            output[idx] = current_func(x) * vle


if __name__ == "__main__":
    Main()
