#include "mex.h"
#include "mf.h"
#include "train.h"
#include <armadillo>
#include <iostream>


/*
   Checks rating data encoded as triplets <user:int32> <item:int32> <rating:float>
   */
void getTripletPrs(const int offset, const mxArray *prhs[], int *&IPr, int  *&JPr, float  *&VPr, int &n)
{
   //check consistent sizes
   const mxArray &mxI = *prhs[offset];
   const mxArray &mxJ = *prhs[offset + 1];
   const mxArray &mxV = *prhs[offset + 2];

   size_t nI = ::mxGetNumberOfElements(&mxI);
   size_t nJ = ::mxGetNumberOfElements(&mxJ);
   size_t nV = ::mxGetNumberOfElements(&mxV);

   if (nI != nJ || nI != nV || nJ != nV)
      ::mexErrMsgTxt("Input matrices have inconsistent dimensions\n");

   //check input types
   if (!(::mxIsInt32(&mxI) && ::mxIsInt32(&mxJ) && ::mxIsSingle(&mxV)))
      ::mexErrMsgTxt("Input matrices have invalid types\n");

   //retrieve pointers
   IPr = (int *) ::mxGetData(&mxI);
   JPr = (int *) ::mxGetData(&mxJ);
   VPr = (float *) ::mxGetData(&mxV);
   n = nI;
}

void initOutputPr(const int outIdx, mxArray *plhs[], int dim1, int dim2, float  *&outPr)
{
   plhs[outIdx] = mxCreateNumericMatrix(dim1, dim2, mxSINGLE_CLASS, mxREAL);
   outPr = (float *)::mxGetData(plhs[outIdx]);
   std::fill(outPr, outPr + dim1 * dim2, 0.0);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   //The interface should be mexLibMF(Itr,Jtr,Vtr,Ite,Jte,Vte,  D,lambda u, lambad v, gamma, maxIt, M, N)
   if (nrhs < 16)
      ::mexErrMsgTxt("Not enough input parameters. Usage\nmexLibMF(Itr,Jtr,Vtr,Ite,Jte,Vte,  D,lambda u, lambad v, gamma, maxIt)\n");


   int *ItrPr, * ItePr, * JtrPr, * JtePr,  nTr, nTe;
   float *VtrPr, * VtePr;

   getTripletPrs(0, prhs, ItrPr, JtrPr, VtrPr, nTr); // obtain pointers to I,J,V training
   getTripletPrs(3, prhs, ItePr, JtePr, VtePr, nTe); // obtain pointers to I,J,V test

   arma::Col<int> Itr(ItrPr, nTr, false, true); //wrap pointers
   arma::Col<int> Jtr(JtrPr, nTr, false, true); // in vectors
   arma::Col<float> Vtr(VtrPr, nTr, false, true); //training

   arma::Col<int> Ite(ItePr, nTe, false, true); //wrap poitners
   arma::Col<int> Jte(JtePr, nTe, false, true); // in vectors
   arma::Col<float> Vte(VtePr, nTe, false, true); //test

   int D = int(::mxGetScalar(prhs[6])); //get latent dimension
   double lambdap = (::mxGetScalar(prhs[7])); //get regularizer
   double lambdaq = (::mxGetScalar(prhs[8]));
   double gamma = (::mxGetScalar(prhs[9]));
   int maxIt = int(::mxGetScalar(prhs[10]));
   int M = int(::mxGetScalar(prhs[11]));
   int N = int(::mxGetScalar(prhs[12]));
   int useub = int(::mxGetScalar(prhs[13]));
   int useib = int(::mxGetScalar(prhs[14]));
   int useavg = int(::mxGetScalar(prhs[15]));

   if (M - 1 > arma::max(Itr) || M - 1 > arma::max(Ite) || N - 1 > arma::max(Jte) || N - 1 > arma::max(Jte)) {
      ::mexErrMsgTxt("User or item index out of bounds!\n");
   }


   if (D <= 0 || lambdap < 0 || lambdaq < 0 || gamma <= 0 || maxIt < 0)
      ::mexErrMsgTxt("Scalar parameters: invalid range\n");



   Model *model = new Model;

   Monitor *monitor = new Monitor;

   //This seems to be a command line parser.
   //int argc = 0;
   //char ** argv = NULL;
   //TrainOption *option = new TrainOption(argc, argv, model, monitor);

   //HERE: Set model parameters directly
   model->dim = D;
   model->nr_thrs = 4;
   model->iter = maxIt;
   model->nr_gubs = 2 * model->nr_thrs;
   model->nr_gibs = model->nr_gubs;
   model->lp = lambdap;
   model->lq = lambdaq;
   model->lub = model->lp;
   model->lib = model->lq;
   model->gamma = gamma; //learning rate
   model->avg = 0.0;
   model->en_rand_shuffle = false;
   model->en_avg = useavg > 0;
   model->en_ub = useub > 0;
   model->en_ib = useib > 0;
   model->map_uf = NULL;
   model->map_if = NULL;
   model->map_ub = NULL;
   model->map_ib = NULL;

   delete [] model->map_uf;

   monitor->en_show_tr_rmse = true;
   monitor->en_show_obj = true;

   Matrix *Tr, *Va = NULL;
   GridMatrix *TrG;
   float avg = model->en_avg ? arma::mean(Vtr) : 0.0;

   Tr = new Matrix(nTr, N, M, avg);
   for (int it = 0 ; it < nTr; it++) {
      Tr->M[it].iid = Itr(it);
      Tr->M[it].uid = Jtr(it);
      Tr->M[it].rate = Vtr(it);
   }

   model->initialize(Tr);


   Va = new Matrix(nTe, N, M, avg);
   for (int it = 0 ; it < nTe; it++) {
      Va->M[it].iid = Ite(it);
      Va->M[it].uid = Jte(it);
      Va->M[it].rate = Vte(it);
   }

   //this means that there cannot be items/users in the test set that are not present in the training set
   if (Va && (Va->nr_us > Tr->nr_us || Va->nr_is > Tr->nr_is)) {
      ::mexErrMsgTxt("Validation set out of range.\n");
   }


   monitor->model = model;
   monitor->Va = Va;
   //this counts ratings per user and ratings per item
   monitor->scan_tr(Tr);

   TrG = new GridMatrix(Tr, model->map_uf, model->map_if, model->nr_gubs, model->nr_gibs, model->nr_thrs);

   delete Tr; //discard temporary triplets


   gsgd(TrG, model, monitor);


   float *qPr , *pPr , *ibPr , *ubPr , *avgPr ;

   initOutputPr(0, plhs, D, M,  qPr);
   initOutputPr(1, plhs, D, N,  pPr);
   initOutputPr(2, plhs, 1, M,  ibPr);
   initOutputPr(3, plhs, 1, N,  ubPr);
   initOutputPr(4, plhs, 1, 1,  avgPr);

   arma::Mat<float> Qout(qPr, D, M, false, true);
   arma::Mat<float> Pout(pPr, D, N, false, true);

   //Internal representation is padded to multiple of 4 for SSE
   arma::Mat<float> Qm(model->Q, model->dim_off, M, false, true);
   arma::Mat<float> Pm(model->P, model->dim_off, N, false, true);

   //Copy factors
   Qout = Qm.rows(0, D - 1);
   Pout = Pm.rows(0, D - 1);

   //Copy bias terms
   if (model->en_ib) {
      std::copy(model->IB, model->IB + M, ibPr);
   }
   if (model->en_ub) {
      std::copy(model->UB, model->UB + N, ubPr);
   }
   if (model->en_avg) {
      *avgPr = model->avg;
   }

   //Cleanup
   delete model;
   delete monitor;
   delete TrG;
}

