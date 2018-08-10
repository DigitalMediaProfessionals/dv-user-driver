.PHONY:	all clean

all:	test_conv

test_conv:	test_conv.cpp
	g++ test_conv.cpp -o test_conv -std=c++11 -Wall -Werror -I../../include -O3 -march=native -mtune=native -mfp16-format=ieee -L../.. -ldmpdv -lcrypto

clean:
	rm -f test_conv
