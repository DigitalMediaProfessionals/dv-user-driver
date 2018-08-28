include ../../../env.mk

.PHONY:	all clean

all:	test_weights

test_weights:	test_weights.cpp
	$(GPP) test_weights.cpp -o test_weights -std=c++11 -Wall -Werror -Wno-unused-function -I../../include $(OPT) -L../.. -ldmpdv -lcrypto

clean:
	rm -f test_weights
