include ../../../env.mk

.PHONY:	all clean

all:	test_mem mem_perf

test_mem:	test_mem.cpp ../../libdmpdv.so
	$(GPP) test_mem.cpp -o test_mem -std=c++11 -Wall -Werror -I../../include $(OPT) -L../.. -ldmpdv

mem_perf:	mem_perf.c ../../libdmpdv.so
	$(GCC) mem_perf.c -o mem_perf -std=c99 -Wall -Werror -D_GNU_SOURCE -I../../include $(OPT) -L../.. -ldmpdv -lstdc++

clean:
	rm -f test_mem mem_perf
