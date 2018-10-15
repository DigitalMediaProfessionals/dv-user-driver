include ../../../env.mk

.PHONY:	all clean

all:	test_pool

test_pool:	test_pool.cpp
	$(GPP) test_pool.cpp -o test_pool -std=c++11 -Wall -Werror -I../../include $(OPT) -L../.. -ldmpdv -lcrypto

clean:
	rm -f test_pool
