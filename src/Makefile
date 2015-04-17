DEV = $(HOME)/dev
#SRC_LIBMF := libMF/src/main.cpp libMF/src/convert.cpp libMF/src/train.cpp libMF/src/predict.cpp libMF/src/view.cpp libMF/src/mf.h libMF/src/mf.cpp 

SRC =	libMF/src/convert.cpp \
		libMF/src/train.cpp \
		libMF/src/predict.cpp \
		libMF/src/view.cpp \
		libMF/src/mf.cpp 


INCLUDES = -I$(DEV)/arma/include/ \
		   -I$(DEV)/matlab/extern/include \
		   -I$(DEV)/unittest-cpp \
		   -I$(HOME)/repos/utils \
		   -IlibMF/src

CXXFLAGS = $(INCLUDES) -g -pthread -std=c++11 -march=native -mtune=native -funroll-loops -Wall -fPIC

OBJ = $(patsubst %.cpp,%.o,$(SRC)) 

all: mex

mex: mexLibMF.o  $(OBJ)
	echo $(OBJ)
	mex -v -cxx -largeArrayDims $^ -lmwlapack -lmwblas  

clean: 
	rm -rf *.o $(OBJ)
