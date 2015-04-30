
DEV = $(HOME)/dev
SRC = mf.cpp  \
	  memalign.cpp

INCLUDES = -I$(DEV)/arma/include/ \
		   -I$(DEV)/matlab/extern/include \
		   -I.


OBJ = $(patsubst %.cpp,%.o,$(SRC)) 

CXXFLAGS =  $(INCLUDES) -g -pthread -std=c++11 -Wall -fPIC -msse3
train: myTrain.o $(OBJ)
	mex -v -cxx -largeArrayDims $^ -lmwlapack -lmwblas  /usr/lib/gcc/x86_64-linux-gnu/4.8/libstdc++.a

clean: 
	rm -rf *.o $(OBJ) *.so
