.PHONY:	all clean

all:	libdv.so test_mem test_cmdlist

weights.o:	weights.cpp dv.h
	g++ -fPIC -c weights.cpp -o weights.o -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -fvisibility=hidden

dv.o:	dv.cpp dv.h
	g++ -fPIC -c dv.cpp -o dv.o -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -fvisibility=hidden

libdv.so:	dv.o weights.o
	g++ -fPIC -shared dv.o weights.o -o libdv.so -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -fvisibility=hidden

test_mem:	test_mem.cpp libdv.so
	g++ test_mem.cpp -o test_mem -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -L. -ldv

test_cmdlist:	test_cmdlist.cpp libdv.so
	g++ test_cmdlist.cpp -o test_cmdlist -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -mfp16-format=ieee -L. -ldv

clean:
	rm -f libdv.so test_mem test_cmdlist *.o
