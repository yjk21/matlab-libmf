#ifndef IFLIBMF_H
#define  IFLIBMF_H

#include <vector>
#include <string>
#include <iostream>
//#include <mf.h>

using std::string;
using std::vector;
using std::cout;
using std::endl;

struct libMFParams{
    int D;
    int maxIt;
    float lambdap;
    float lambdaq;
    float eta;
    bool ub;
    bool ib;
    bool avg;
    int nThreads;
    int M;
    int N;

//    libMFParams(const int pD, const int pMaxIt, const float pLambdap, const float pLambdaq, const float pEta, const bool pUb, const bool pIb, const bool pAvg, const int pNThreads) : D(pD), maxIt(pMaxIt), lambdap(pLambdap),lambdaq(pLambdaq), eta(pEta), ub(pUb), ib(pIb), avg(pAvg), nThreads(pNThreads){}
};


void params2vec(const libMFParams & params, vector<string> & argAll);

std::shared_ptr<TrainOption> initLibMF(const libMFParams & params );

#endif
