#!/bin/bash
sudo env LD_LIBRARY_PATH=`pwd` sh -c 'cd tests/test_context && ./test_context && cd ../test_mem && ./test_mem 64 && cd ../test_weights && ./test_weights && cd ../test_conv && ./test_conv && cd ../test_fc && ./test_fc'
