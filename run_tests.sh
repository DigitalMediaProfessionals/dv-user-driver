#!/bin/bash
sudo env LD_LIBRARY_PATH=`pwd` sh -c 'cd tests/test_mem && ./test_mem 64 && cd ../test_weights && ./test_weights && cd ../test_cmdlist && ./test_cmdlist'
