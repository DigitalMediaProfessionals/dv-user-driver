include ../../../env.mk

.PHONY:	all clean

all:	test_add_act_pool

test_add_act_pool:	test_add_act_pool.cpp
	$(GPP) test_add_act_pool.cpp -o test_add_act_pool -std=c++11 -Wall -Werror -D_GNU_SOURCE -I../../include $(OPT) -ldmpdv

clean:
	rm -f test_add_act_pool
