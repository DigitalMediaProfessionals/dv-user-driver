include ../env.mk

.PHONY:	all clean tests

all:	libdmpdv.so

weights_conv.o:	src/weights_conv.c include/dmp_dv.h
	$(GCC) -fPIC -c src/weights_conv.c -o weights_conv.o -std=c99 -Wall -Werror -Wno-unused-function -I./include $(OPT) -fvisibility=hidden

weights_dil.o:	src/weights_dil.c include/dmp_dv.h
	$(GCC) -fPIC -c src/weights_dil.c -o weights_dil.o -std=c99 -Wall -Werror -Wno-unused-function -I./include $(OPT) -fvisibility=hidden

weights_fc.o:	src/weights_fc.c include/dmp_dv.h
	$(GCC) -fPIC -c src/weights_fc.c -o weights_fc.o -std=c99 -Wall -Werror -Wno-unused-function -I./include $(OPT) -fvisibility=hidden

dmp_dv.o:	src/dmp_dv.cpp include/*.h include/*.hpp
	$(GPP) -fPIC -c src/dmp_dv.cpp -o dmp_dv.o -std=c++11 -Wall -Werror -Wno-unused-function -I./include $(OPT) -fvisibility=hidden

libdmpdv.so:	dmp_dv.o weights_conv.o weights_dil.o weights_fc.o
	$(GCC) -fPIC -shared dmp_dv.o weights_conv.o weights_dil.o weights_fc.o -o libdmpdv.so -std=c++11 -Wall -Werror $(OPT) -fvisibility=hidden

tests:	libdmpdv.so
	$(MAKE) -C tests $@

.SILENT:	install

install:
	echo Copying libdmpdv.so to /usr/lib/
	cp libdmpdv.so /usr/lib/
	echo ldconfig
	ldconfig
	echo Copying header files to /usr/include/
	cp include/dmp_dv.h include/dmp_dv_cmdraw_v*.h /usr/include/
	echo Successfully installed

clean:
	rm -f libdmpdv.so *.o
	$(MAKE) -C tests $@
