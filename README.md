# matlab-libmf

This is a mex interface to call libMF directly from MATLAB.

The libMF official website is http://www.csie.ntu.edu.tw/~cjlin/libmf.

We use the port that was used as backend for the recosystem R package:
http://cran.r-project.org/web/packages/recosystem/index.html

Currently, we use functionality from the armadillo linear algebra library: 
http://arma.sourceforge.net/

Compilation requires a c++11 compliant compiler (tested only with g++ 4.8).
