.PHONY:	all clean

all:	test_context

test_context:	test_context.cpp
	g++ test_context.cpp -o test_context -std=c++11 -Wall -Werror -I../../include -O3 -march=native -mtune=native -L../.. -ldmpdv

clean:
	rm -f test_context
