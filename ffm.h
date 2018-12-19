#ifndef _LIBFFM_H
#define _LIBFFM_H

#include <string>

extern "C" {

namespace ffm {

using namespace std;

typedef float ffm_float;
typedef double ffm_double;
typedef int ffm_int;
typedef long long ffm_long;

struct ffm_node {
    ffm_int f; // field index
    ffm_int j; // feature index
    ffm_float v; // value
};

struct ffm_model {
    ffm_int n; // number of features
    ffm_int m; // number of fields
    ffm_int k; // number of latent factors
    ffm_float *W = nullptr;
    bool normalization;
    ~ffm_model();
};

struct ffm_parameter {
    ffm_float eta = 0.2; // learning rate
    ffm_float lambda = 0.00002; // regularization parameter
    ffm_int nr_iters = 15;
    ffm_int k = 4; // number of latent factors
    bool normalization = true;
    bool auto_stop = false;
};

void ffm_read_problem_to_disk(string txt_path, string bin_path);

void ffm_save_model(ffm_model &model, string path);

ffm_model ffm_load_model(string path);

ffm_model ffm_train_on_disk(string Tr_path, string Va_path, ffm_parameter param);

ffm_float ffm_predict(ffm_node *begin, ffm_node *end, ffm_model &model);


// new structs and methods for the wrapper

struct ffm_line {
    ffm_node* data;
    ffm_float label;
    ffm_int size;
};

struct ffm_problem {
    ffm_int size = 0;
    ffm_long num_nodes = 0;

    ffm_node* data;
    ffm_long* pos;
    ffm_float* labels;
    ffm_float* scales;

    ffm_int n = 0;
    ffm_int m = 0;
};

ffm_model ffm_load_model_c_string(char *path);

void ffm_save_model_c_string(ffm_model &model, char *path);

ffm_problem ffm_convert_data(ffm_line *data, ffm_int num_lines);

void ffm_cleanup_data(ffm_problem *p);

ffm_model ffm_init_model(ffm_problem &data, ffm_parameter params);

ffm_float ffm_train_iteration(ffm_problem &data, ffm_model &model, ffm_parameter params);

ffm_float ffm_predict_array(ffm_node *nodes, int len, ffm_model &model);

ffm_float* ffm_predict_batch(ffm_problem &data, ffm_model &model);

void ffm_cleanup_prediction(ffm_float* f);

ffm_int ffm_get_k_aligned(ffm_int k);

ffm_int ffm_get_kALIGN();

} // namespace ffm

#endif // _LIBFFM_H
}
