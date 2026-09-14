#ifndef BACKPROP_H
#define BACKPROP_H

#include "triangle.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    int vocab_size;
    int embed_dim;
    double *embeddings; // [vocab_size * embed_dim]
    double *weights_ih;
    double *weights_ho;
    double *bias_h;
    double *bias_o;
} BackpropNetwork;

typedef struct {
    int epoch;
    double loss;
    double accuracy;
} TrainingLog;

typedef struct {
    BackpropNetwork *network;
    TrainingLog *logs;
    int log_count;
    int max_epochs;
    double learning_rate;
} BackpropTrainer;

BackpropTrainer *backprop_create(int vocab_size, int embed_dim, int hidden_size, int output_size, int max_epochs, double lr);
void backprop_free(BackpropTrainer *trainer);
void backprop_train(BackpropTrainer *trainer, const TriangleChain *chain);
int backprop_train_cuda(BackpropTrainer *trainer, const TriangleChain *chain, int requested_gpus);
int backprop_predict(BackpropTrainer *trainer, double *input, double *output);
void backprop_print_logs(const BackpropTrainer *trainer);
void backprop_save_model(const BackpropTrainer *trainer, const char *filename);
BackpropTrainer *backprop_load_model(const char *filename);
TriangleChain *backprop_reduce_vocab(const TriangleChain *chain, int top_n);

#ifdef __cplusplus
}
#endif

#endif
