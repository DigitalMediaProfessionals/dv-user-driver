include ../../env.mk

INC = -I../include

# source files.
SRC = hw_module_manager.cpp ieeehalfprecision.cpp dmp_network.cpp

OBJ = $(SRC:.cpp=.o)

OUT = ../lib/libdv700.a

# include directories
INCLUDES = -I. -I../include

# C++ compiler flags (-g -O2 -Wall)
CCFLAGS =  -pthread -std=c++11 -O3 -D$(BDDEF) -Wall

# compiler
CCC = $(GPP)

# library paths
LIBS = -L../ -L/usr/local/lib -lm

# compile flags
LDFLAGS = -O3

.SUFFIXES: .cpp

all: $(OUT)

.cpp.o:
	$(CCC) $(INCLUDES) $(CCFLAGS) -c $< -o $@

$(OUT): $(OBJ)
	ar rcs $(OUT) $(OBJ)

clean:
	rm -f $(OBJ) $(OUT)
