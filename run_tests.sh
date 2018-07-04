#!/bin/bash
sudo env LD_LIBRARY_PATH=. sh -c './test_mem 64 && ./test_cmdlist'
