include ../../../env.mk

.PHONY:	all clean

all:	test_mem

test_mem:	test_mem.cpp ../../libdmpdv.so
	$(GPP) test_mem.cpp -o test_mem -std=c++11 -Wall -Werror -I../../include $(OPT) -L../.. -ldmpdv

clean:
	rm -f test_mem
