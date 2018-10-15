include ../../../env.mk

.PHONY:	all clean

all:	test_multirun

test_multirun:	test_multirun.c ../../libdmpdv.so
	$(GCC) test_multirun.c -o test_multirun -std=c99 -Wall -Werror -I../../include $(OPT) -L../.. -ldmpdv -lstdc++

clean:
	rm -f test_multirun
