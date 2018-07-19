.PHONY:	all clean tests

all:	libdmpdv.so tests

weights.o:	src/weights.c include/dmp_dv.h
	gcc -fPIC -c src/weights.c -o weights.o -std=c99 -Wall -Werror -I./include -O3 -march=native -mtune=native -fvisibility=hidden

dmp_dv.o:	src/dmp_dv.cpp include/dmp_dv.h
	g++ -fPIC -c src/dmp_dv.cpp -o dmp_dv.o -std=c++11 -Wall -Werror -I./include -O3 -march=native -mtune=native -fvisibility=hidden

libdmpdv.so:	dmp_dv.o weights.o
	g++ -fPIC -shared dmp_dv.o weights.o -o libdmpdv.so -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -fvisibility=hidden

tests:	libdmpdv.so
	$(MAKE) -C tests $@

clean:
	rm -f libdmpdv.so *.o
	$(MAKE) -C tests $@
