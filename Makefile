.PHONY:	all clean

all:	libdv.so test_mem test_cmdlist

libdv.so:	dv.cpp dv.h
	g++ -fPIC -shared dv.cpp -o libdv.so -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -fvisibility=hidden

test_mem:	test_mem.cpp libdv.so
	g++ test_mem.cpp -o test_mem -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -L. -ldv

test_cmdlist:	test_cmdlist.cpp libdv.so
	g++ test_cmdlist.cpp -o test_cmdlist -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -L. -ldv

clean:
	rm -f libdv.so test_mem test_cmdlist
