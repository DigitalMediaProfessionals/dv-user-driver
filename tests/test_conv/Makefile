include ../../../env.mk

.PHONY:	all clean

all:	test_conv

test_conv:	test_conv.cpp ../../libdmpdv.so
	$(GPP) test_conv.cpp -o test_conv -std=c++11 -Wall -Werror -I../../include $(OPT) -L../.. -ldmpdv -lcrypto

clean:
	rm -f test_conv
