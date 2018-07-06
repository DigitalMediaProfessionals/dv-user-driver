.PHONY:	all clean tests

all:	libdv.so tests

weights.o:	weights.cpp dv.h
	g++ -fPIC -c weights.cpp -o weights.o -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -fvisibility=hidden

dv.o:	dv.cpp dv.h
	g++ -fPIC -c dv.cpp -o dv.o -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -fvisibility=hidden

libdv.so:	dv.o weights.o
	g++ -fPIC -shared dv.o weights.o -o libdv.so -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -fvisibility=hidden

tests:	libdv.so
	$(MAKE) -C tests $@

clean:
	rm -f libdv.so *.o
	$(MAKE) -C tests $@
