.PHONY:	all clean

all:	test_fc

test_fc:	test_fc.cpp
	g++ test_fc.cpp -o test_fc -std=c++11 -Wall -Werror -I../../include -O3 -march=native -mtune=native -mfp16-format=ieee -L../.. -ldmpdv -lcrypto

clean:
	rm -f test_fc
