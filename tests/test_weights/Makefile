.PHONY:	all clean

all:	test_weights

test_weights:	test_weights.cpp
	g++ test_weights.cpp -o test_weights -std=c++11 -Wall -Werror -I../../include -O3 -march=native -mtune=native -mfp16-format=ieee -L../.. -ldmpdv -lcrypto

clean:
	rm -f test_weights
