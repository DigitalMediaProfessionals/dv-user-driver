include ../../../env.mk

.PHONY:	all clean

all:	test_lrn

test_lrn:	test_lrn.cpp
	$(GPP) test_lrn.cpp -o test_lrn -std=c++11 -Wall -Werror -I../../include $(OPT) -L../.. -ldmpdv -lcrypto

clean:
	rm -f test_lrn
