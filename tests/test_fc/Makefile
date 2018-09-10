include ../../../env.mk

.PHONY:	all clean

all:	test_fc

test_fc:	test_fc.cpp ../../libdmpdv.so
	$(GPP) test_fc.cpp -o test_fc -std=c++11 -Wall -Werror -I../../include $(OPT) -L../.. -ldmpdv -lcrypto

clean:
	rm -f test_fc
