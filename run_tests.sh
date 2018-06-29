#!/bin/bash
make
sudo env LD_LIBRARY_PATH=. ./test_dv $@
