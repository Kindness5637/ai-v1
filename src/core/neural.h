#ifndef NEURAL_H
#define NEURAL_H

#include "triangle.h"

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double *weights_ih;
    double *weights_ho;
    double *bias_h;
    double *bias_o;
} NeuralNetwork;

typedef struct {
    double *input;
    double *target;
    int input_size;
    int target_size;
} TrainingSample;

typedef struct {
    TrainingSample *samples;
    size_t count;
    size_t capacity;
} TrainingSet;

typedef struct {
    int predicted_id;
    int actual_id;
    double confidence;
    int correct;
} PredictionResult;

typedef struct {
    PredictionResult *results;
    size_t count;
    double accuracy;
    double deviation;
} EvaluationOutput;

TrainingSet *training_create(const TriangleChain *chain);
void training_free(TrainingSet *set);

NeuralNetwork *nn_create(int input_size, int hidden_size, int output_size);
void nn_free(NeuralNetwork *nn);
void nn_train(NeuralNetwork *nn, TrainingSet *set, int epochs, double learning_rate);
int nn_predict(NeuralNetwork *nn, double *input, double *output);
void nn_save(const NeuralNetwork *nn, const char *filename);
NeuralNetwork *nn_load(const char *filename);

EvaluationOutput *evaluate(NeuralNetwork *nn, const TriangleChain *chain);
void evaluation_free(EvaluationOutput *eval);
void evaluation_print(const EvaluationOutput *eval);

#endif
