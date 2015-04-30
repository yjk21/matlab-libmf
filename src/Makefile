DEV = $(HOME)/dev
#SRC_LIBMF := libMF/src/main.cpp libMF/src/convert.cpp libMF/src/train.cpp libMF/src/predict.cpp libMF/src/view.cpp libMF/src/mf.h libMF/src/mf.cpp 

SRC =	libMF/src/convert.cpp \
		libMF/src/train.cpp \
		libMF/src/predict.cpp \
		libMF/src/view.cpp \
		libMF/src/mf.cpp 

### BLAS Support for Julia ccall wrappers ###
# Julia comes with its own openblas shared library, which is configured for ILP64.
# To avoid potential naming conflicts when external libs link to BLAS implementations with 32 bit ints, the standard BLAS names are augmented with a suffix "64_" (e.g. dgemm_ => dgemm_64_)
# Ideally, we would like to link to the same blas, that julia uses, to save memory.
# Taking account of this modification becomes particularly easy, when using armadillo.
# Armadillo needs to be configured for ILP64 bit ints for indexing (config.hpp) and to append the aforementioned suffix to the low-level 
# BLAS calls. The latter can be done in compiler_setup.hpp.
INCLUDES = -I$(DEV)/arma/include/ \
		   -I$(DEV)/matlab/extern/include \
		   -I$(DEV)/unittest-cpp \
		   -I$(HOME)/repos/utils \
		   -IlibMF/src

CXXFLAGS = $(INCLUDES) -g -pthread -std=c++11 -march=native -mtune=native -funroll-loops -Wall -fPIC
CFLAGS = -g -fPIC

OBJ = $(patsubst %.cpp,%.o,$(SRC)) 

all: mex

mex: mexLibMF.o  $(OBJ)
	echo $(OBJ)
	mex -v -cxx -largeArrayDims $^ -lmwlapack -lmwblas  

clean: 
	rm -rf *.o $(OBJ) *.so

ifjl: jltest.o 
	gcc -shared -o jltest.so -L/home/ko/dev/julia-cb9bcae93a/lib/julia/ $^ -lopenblas

runjl: ifjl
	LD_LIBRARY_PATH=/home/ko/dev/matlab/extern/lib/glnxa64/:/home/ko/repos/matlab-libmf/src julia runLibMF.jl

