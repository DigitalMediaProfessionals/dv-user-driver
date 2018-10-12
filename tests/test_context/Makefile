include ../../../env.mk

.PHONY:	all clean

all:	test_context

test_context:	test_context.c ../../libdmpdv.so
	$(GCC) test_context.c -o test_context -std=c99 -Wall -Werror -I../../include $(OPT) -L../.. -ldmpdv -lstdc++

clean:
	rm -f test_context
