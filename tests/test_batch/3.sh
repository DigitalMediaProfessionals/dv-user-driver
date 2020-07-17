#!/bin/bash
export SEED=1234
export REPEAT=2
N=512 H=16 W=16 KX=3 PAD=1 STRIDE=1 C=128 M=256 ./test_batch
