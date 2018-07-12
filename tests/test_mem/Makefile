.PHONY:	all clean

all:	test_mem

test_mem:	test_mem.cpp
	g++ test_mem.cpp -o test_mem -std=c++11 -Wall -Werror -I../../include -O3 -march=native -mtune=native -L../.. -ldmpdv

clean:
	rm -f test_mem
