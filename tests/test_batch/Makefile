include ../../../env.mk

.PHONY:	all clean

all:	test_batch

test_batch:	test_batch.cpp
	$(GPP) test_batch.cpp -o test_batch -std=c++11 -Wall -Werror -D_GNU_SOURCE -I../../include -I../common $(OPT) -ldmpdv

clean:
	rm -f test_batch
