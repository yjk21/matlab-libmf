#include "mex.h"
#include "mf.h"
#include "train.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   int argc = 0;
   char **argv = NULL;

   Model *model = new Model;

   Monitor *monitor = new Monitor;

   TrainOption *option = new TrainOption(argc, argv, model, monitor);

   Matrix *Tr, *Va = NULL;
   GridMatrix *TrG;

   Tr = new Matrix(option->tr_path);

   model->initialize(Tr);

   if (model->en_rand_shuffle) model->gen_rand_map();

   if (option->va_path) {
      if (model->en_rand_shuffle) Va = new Matrix(option->va_path, model->map_uf, model->map_if);
      else Va = new Matrix(option->va_path);
   }

   if (Va && (Va->nr_us > Tr->nr_us || Va->nr_is > Tr->nr_is)) {
      fprintf(stderr, "Validation set out of range.\n");
      exit(1);
   }

   if (model->en_rand_shuffle) model->shuffle();

   monitor->model = model;
   monitor->Va = Va;
   monitor->scan_tr(Tr);

   TrG = new GridMatrix(Tr, model->map_uf, model->map_if, model->nr_gubs, model->nr_gibs, model->nr_thrs);

   delete Tr;

   gsgd(TrG, model, monitor);

   if (model->en_rand_shuffle) model->inv_shuffle();

   model->write(option->model_path);

   delete model;
   delete monitor;
   delete option;
   delete TrG;
}

