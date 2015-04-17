#ifndef TRAIN_H
#define TRAIN_H
struct Monitor {
   int iter, *nr_tr_usrs, *nr_tr_isrs;
   float tr_time;
   bool en_show_tr_rmse, en_show_obj;
   Matrix *Va;
   Model *model;
   Monitor();
   void print_header(); //打印训练过程中表格信息的表头
   void show(float iter_time, double loss, float tr_rmse); //打印训练过程中表格信息的每一行信息
   void scan_tr(Matrix *Tr);
   double calc_reg();
   ~Monitor();
};


struct GridMatrix {
   int nr_gubs, nr_gibs; //nr_gubs代表user被分成的块数，nr_gibs代表item被分成的块数（与Model中参数意思一样）
   long nr_rs; //总的评分数
   Matrix **GMS; //存放每一个分块好的数据block，一共nr_gubs * nr_gibs个
   GridMatrix(Matrix *G, int *map_u, int *map_i, int nr_gubs, int nr_gibs, int nr_thrs);
   static void sort_ratings(Matrix *M, std::mutex *mtx, int *nr_thrs);
   ~GridMatrix();
};

struct TrainOption {
   char *tr_path, *va_path, *model_path; //*tr_path代表训练集，*va_path代表验证集validation set
   TrainOption(int argc, char **argv, Model *model, Monitor *monitor);
   static void exit_train();
   ~TrainOption();
};


void gsgd(GridMatrix *TrG, Model *model, Monitor *monitor);
#endif

