include ../../../env.mk

.PHONY:	all clean

TARGET=test_maximizer

all:	$(TARGET)

$(TARGET):	test_maximizer.cpp ../../libdmpdv.so
	$(GPP) test_maximizer.cpp -o $(TARGET) -std=c++11 -Wall -Werror -I../../include $(OPT) -L../.. -ldmpdv

clean:
	rm -f $(TARGET)
