.PHONY:	all clean

all:	libdv.so test_dv

libdv.so:	dv.cpp dv.h
	g++ -fPIC -shared dv.cpp -o libdv.so -std=c++11 -Wall -Werror -O3 -march=native -mtune=native

test_dv:	test_dv.cpp libdv.so
	g++ test_dv.cpp -o test_dv -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -L. -ldv

clean:
	rm -f libdv.so test_dv
