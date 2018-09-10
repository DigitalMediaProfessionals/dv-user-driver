include ../../../env.mk

.PHONY:	all clean

all:	test_context

test_context:	test_context.cpp ../../libdmpdv.so
	$(GPP) test_context.cpp -o test_context -std=c++11 -Wall -Werror -I../../include $(OPT) -L../.. -ldmpdv

clean:
	rm -f test_context
