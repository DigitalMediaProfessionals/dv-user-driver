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


class Main(object):
    def __init__(self):
        self.generate_weights()
        self.generate_input()
        self.compute_output()

    def generate_weights(self):
        try:
            os.mkdir("data")
        except OSError:
            pass

        quant_map = numpy.random.uniform(-1.0, 1.0, 256).astype(numpy.float16)
        quant_map[0] = 0
        fnme = "data/map.f16"
        quant_map.tofile(fnme)
        print("Saved quantization map to %s" % fnme)

        # TODO: implement.

    def generate_input(self):
        # TODO: implement.
        pass

    def compute_output(self):
        # TODO: implement.
        pass


if __name__ == "__main__":
    Main()
