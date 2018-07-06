#!/usr/bin/python3
"""
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------

Generates weights/compares result for test_cmdlist.
"""
import argparse
import numpy
import os


class Main(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-g", "--generate", action="store_true",
                            help="Generate weights")
        args = parser.parse_args()
        if args.generate:
            self.generate_weights()

    def generate_weights(self):
        try:
            os.mkdir("test_cmdlist_data")
        except OSError:
            pass

        quant_map = numpy.random.rand(256).astype(numpy.float16)
        quant_map[0] = 0
        fnme = "test_cmdlist_data/map.f16"
        quant_map.tofile(fnme)
        print("Saved quantization map to %s" % fnme)

        # TODO: continue here.


if __name__ == "__main__":
    Main()
