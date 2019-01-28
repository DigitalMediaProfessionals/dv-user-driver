include ../../../env.mk

.PHONY:	all clean

all:	test_upsampling

test_upsampling:	test_upsampling.c ../../libdmpdv.so
	$(GCC) test_upsampling.c -o test_upsampling -std=c99 -Wall -Werror -I../../include $(OPT) -L../.. -ldmpdv -lstdc++

clean:
	rm -f test_upsampling
