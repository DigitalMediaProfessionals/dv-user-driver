.PHONY:	all clean tests

all:	libdmpdv.so tests

weights_conv.o:	src/weights_conv.c include/dmp_dv.h
	gcc -fPIC -c src/weights_conv.c -o weights_conv.o -std=c99 -Wall -Werror -Wno-unused-function -I./include -O3 -march=native -mtune=native -fvisibility=hidden

weights_fc.o:	src/weights_fc.c include/dmp_dv.h
	gcc -fPIC -c src/weights_fc.c -o weights_fc.o -std=c99 -Wall -Werror -Wno-unused-function -I./include -O3 -march=native -mtune=native -fvisibility=hidden

dmp_dv.o:	src/dmp_dv.cpp include/dmp_dv.h
	g++ -fPIC -c src/dmp_dv.cpp -o dmp_dv.o -std=c++11 -Wall -Werror -Wno-unused-function -Wno-psabi -I./include -O3 -march=native -mtune=native -fvisibility=hidden

libdmpdv.so:	dmp_dv.o weights_conv.o weights_fc.o
	g++ -fPIC -shared dmp_dv.o weights_conv.o weights_fc.o -o libdmpdv.so -std=c++11 -Wall -Werror -O3 -march=native -mtune=native -fvisibility=hidden

tests:	libdmpdv.so
	$(MAKE) -C tests $@

clean:
	rm -f libdmpdv.so *.o
	$(MAKE) -C tests $@
