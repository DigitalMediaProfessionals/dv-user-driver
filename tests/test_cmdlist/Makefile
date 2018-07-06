.PHONY:	all clean

all:	test_cmdlist

test_cmdlist:	test_cmdlist.cpp
	g++ test_cmdlist.cpp -o test_cmdlist -std=c++11 -Wall -Werror -I../.. -O3 -march=native -mtune=native -mfp16-format=ieee -L../.. -ldv

clean:
	rm -f test_cmdlist
