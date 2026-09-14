#include "neural.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

static double random_weight(void) {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

TrainingSet *training_create(const TriangleChain *chain) {
    if (!chain) return NULL;

    int max_vocab = 100;
    if ((int)chain->vocab->count < max_vocab) {
        max_vocab = chain->vocab->count;
    }

    TrainingSet *set = malloc(sizeof(TrainingSet));
    if (!set) return NULL;

    size_t max_samples = 200;
    if (chain->count * 3 < max_samples) {
        max_samples = chain->count * 3;
    }

    set->count = max_samples;
    set->capacity = max_samples;
    set->samples = malloc(set->count * sizeof(TrainingSample));
    if (!set->samples) {
        free(set);
        return NULL;
    }

    size_t idx = 0;
    for (size_t i = 0; i < chain->count && idx < max_samples; i++) {
        for (int p = 0; p < 3 && idx < max_samples; p++) {
            set->samples[idx].input_size = 4;
            set->samples[idx].target_size = max_vocab;
            set->samples[idx].input = calloc(4, sizeof(double));
            set->samples[idx].target = calloc(max_vocab, sizeof(double));

            if (!set->samples[idx].input || !set->samples[idx].target) {
                for (size_t j = 0; j < idx; j++) {
                    free(set->samples[j].input);
                    free(set->samples[j].target);
                }
                free(set->samples);
                free(set);
                return NULL;
            }

            set->samples[idx].input[0] = (double)i / chain->count;
            set->samples[idx].input[1] = (double)(p + 1) / 3.0;
            set->samples[idx].input[2] = (double)(chain->triangles[i].word_ids[(p + 1) % 3]) / chain->vocab->count;
            set->samples[idx].input[3] = (double)(chain->triangles[i].word_ids[(p + 2) % 3]) / chain->vocab->count;

            int word_id = chain->triangles[i].word_ids[p];
            if (word_id > 0 && word_id <= max_vocab) {
                set->samples[idx].target[word_id - 1] = 1.0;
            }
            idx++;
        }
    }

    set->count = idx;
    return set;
}

void training_free(TrainingSet *set) {
    if (!set) return;
    for (size_t i = 0; i < set->count; i++) {
        free(set->samples[i].input);
        free(set->samples[i].target);
    }
    free(set->samples);
    free(set);
}

NeuralNetwork *nn_create(int input_size, int hidden_size, int output_size) {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    if (!nn) return NULL;

    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    nn->weights_ih = malloc(input_size * hidden_size * sizeof(double));
    nn->weights_ho = malloc(hidden_size * output_size * sizeof(double));
    nn->bias_h = malloc(hidden_size * sizeof(double));
    nn->bias_o = malloc(output_size * sizeof(double));

    if (!nn->weights_ih || !nn->weights_ho || !nn->bias_h || !nn->bias_o) {
        free(nn->weights_ih);
        free(nn->weights_ho);
        free(nn->bias_h);
        free(nn->bias_o);
        free(nn);
        return NULL;
    }

    srand(time(NULL));
    for (int i = 0; i < input_size * hidden_size; i++) {
        nn->weights_ih[i] = random_weight();
    }
    for (int i = 0; i < hidden_size * output_size; i++) {
        nn->weights_ho[i] = random_weight();
    }
    for (int i = 0; i < hidden_size; i++) {
        nn->bias_h[i] = random_weight();
    }
    for (int i = 0; i < output_size; i++) {
        nn->bias_o[i] = random_weight();
    }

    return nn;
}

void nn_free(NeuralNetwork *nn) {
    if (!nn) return;
    free(nn->weights_ih);
    free(nn->weights_ho);
    free(nn->bias_h);
    free(nn->bias_o);
    free(nn);
}

void nn_train(NeuralNetwork *nn, TrainingSet *set, int epochs, double learning_rate) {
    if (!nn || !set) return;

    double *hidden = malloc(nn->hidden_size * sizeof(double));
    double *output = malloc(nn->output_size * sizeof(double));
    double *error_output = malloc(nn->output_size * sizeof(double));
    double *error_hidden = malloc(nn->hidden_size * sizeof(double));

    if (!hidden || !output || !error_output || !error_hidden) {
        free(hidden);
        free(output);
        free(error_output);
        free(error_hidden);
        return;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0;

        for (size_t s = 0; s < set->count; s++) {
            for (int h = 0; h < nn->hidden_size; h++) {
                hidden[h] = 0;
                for (int i = 0; i < nn->input_size; i++) {
                    hidden[h] += set->samples[s].input[i] * nn->weights_ih[i * nn->hidden_size + h];
                }
                hidden[h] = sigmoid(hidden[h] + nn->bias_h[h]);
            }

            for (int o = 0; o < nn->output_size; o++) {
                output[o] = 0;
                for (int h = 0; h < nn->hidden_size; h++) {
                    output[o] += hidden[h] * nn->weights_ho[h * nn->output_size + o];
                }
                output[o] = sigmoid(output[o] + nn->bias_o[o]);
            }

            for (int o = 0; o < nn->output_size; o++) {
                error_output[o] = (set->samples[s].target[o] - output[o]) * sigmoid_derivative(output[o]);
                total_error += pow(set->samples[s].target[o] - output[o], 2);
            }

            for (int h = 0; h < nn->hidden_size; h++) {
                error_hidden[h] = 0;
                for (int o = 0; o < nn->output_size; o++) {
                    error_hidden[h] += error_output[o] * nn->weights_ho[h * nn->output_size + o];
                }
                error_hidden[h] *= sigmoid_derivative(hidden[h]);
            }

            for (int h = 0; h < nn->hidden_size; h++) {
                for (int i = 0; i < nn->input_size; i++) {
                    nn->weights_ih[i * nn->hidden_size + h] += learning_rate * error_hidden[h] * set->samples[s].input[i];
                }
                nn->bias_h[h] += learning_rate * error_hidden[h];
            }

            for (int o = 0; o < nn->output_size; o++) {
                for (int h = 0; h < nn->hidden_size; h++) {
                    nn->weights_ho[h * nn->output_size + o] += learning_rate * error_output[o] * hidden[h];
                }
                nn->bias_o[o] += learning_rate * error_output[o];
            }
        }

        if (epoch % 5 == 0) {
            printf("  Epoch %d: error = %.6f\n", epoch, total_error / set->count);
        }
    }

    free(hidden);
    free(output);
    free(error_output);
    free(error_hidden);
}

int nn_predict(NeuralNetwork *nn, double *input, double *output) {
    if (!nn || !input || !output) return -1;

    double *hidden = malloc(nn->hidden_size * sizeof(double));
    if (!hidden) return -1;

    for (int h = 0; h < nn->hidden_size; h++) {
        hidden[h] = 0;
        for (int i = 0; i < nn->input_size; i++) {
            hidden[h] += input[i] * nn->weights_ih[i * nn->hidden_size + h];
        }
        hidden[h] = sigmoid(hidden[h] + nn->bias_h[h]);
    }

    for (int o = 0; o < nn->output_size; o++) {
        output[o] = 0;
        for (int h = 0; h < nn->hidden_size; h++) {
            output[o] += hidden[h] * nn->weights_ho[h * nn->output_size + o];
        }
        output[o] = sigmoid(output[o] + nn->bias_o[o]);
    }

    free(hidden);

    int best = 0;
    for (int o = 1; o < nn->output_size; o++) {
        if (output[o] > output[best]) {
            best = o;
        }
    }

    return best;
}

EvaluationOutput *evaluate(NeuralNetwork *nn, const TriangleChain *chain) {
    if (!nn || !chain) return NULL;

    EvaluationOutput *eval = malloc(sizeof(EvaluationOutput));
    if (!eval) return NULL;

    eval->count = chain->count * 3;
    eval->results = malloc(eval->count * sizeof(PredictionResult));
    if (!eval->results) {
        free(eval);
        return NULL;
    }

    double *output = malloc(nn->output_size * sizeof(double));
    if (!output) {
        free(eval->results);
        free(eval);
        return NULL;
    }

    int correct = 0;
    double total_confidence = 0;
    size_t idx = 0;

    for (size_t i = 0; i < chain->count && idx < eval->count; i++) {
        for (int p = 0; p < 3 && idx < eval->count; p++) {
            double input[4];
            input[0] = (double)i / chain->count;
            input[1] = (double)(p + 1) / 3.0;
            input[2] = (double)(chain->triangles[i].word_ids[(p + 1) % 3]) / chain->vocab->count;
            input[3] = (double)(chain->triangles[i].word_ids[(p + 2) % 3]) / chain->vocab->count;

            int predicted = nn_predict(nn, input, output);
            int actual = chain->triangles[i].word_ids[p] - 1;

            eval->results[idx].predicted_id = predicted + 1;
            eval->results[idx].actual_id = chain->triangles[i].word_ids[p];
            eval->results[idx].confidence = output[predicted];
            eval->results[idx].correct = (predicted == actual);

            if (predicted == actual) correct++;
            total_confidence += output[predicted];
            idx++;
        }
    }

    eval->count = idx;
    eval->accuracy = (double)correct / eval->count;
    eval->deviation = 1.0 - eval->accuracy;

    free(output);
    return eval;
}

void evaluation_free(EvaluationOutput *eval) {
    if (!eval) return;
    free(eval->results);
    free(eval);
}

void evaluation_print(const EvaluationOutput *eval) {
    if (!eval) return;

    printf("\n=== Neural Network Evaluation ===\n");
    printf("Total predictions: %zu\n", eval->count);
    printf("Accuracy: %.1f%%\n", eval->accuracy * 100);
    printf("Deviation: %.1f%%\n", eval->deviation * 100);

    int correct = 0;
    for (size_t i = 0; i < eval->count; i++) {
        if (eval->results[i].correct) correct++;
    }
    printf("Correct predictions: %d/%zu\n", correct, eval->count);

    printf("\nSample predictions (first 20):\n");
    size_t show = eval->count < 20 ? eval->count : 20;
    for (size_t i = 0; i < show; i++) {
        printf("  Prediction %zu: predicted=%d actual=%d confidence=%.2f %s\n",
               i + 1,
               eval->results[i].predicted_id,
               eval->results[i].actual_id,
               eval->results[i].confidence,
               eval->results[i].correct ? "OK" : "MISS");
    }
}

void nn_save(const NeuralNetwork *nn, const char *filename) {
    if (!nn || !filename) return;

    FILE *f = fopen(filename, "wb");
    if (!f) return;

    fwrite(&nn->input_size, sizeof(int), 1, f);
    fwrite(&nn->hidden_size, sizeof(int), 1, f);
    fwrite(&nn->output_size, sizeof(int), 1, f);
    fwrite(nn->weights_ih, sizeof(double), nn->input_size * nn->hidden_size, f);
    fwrite(nn->weights_ho, sizeof(double), nn->hidden_size * nn->output_size, f);
    fwrite(nn->bias_h, sizeof(double), nn->hidden_size, f);
    fwrite(nn->bias_o, sizeof(double), nn->output_size, f);

    fclose(f);
}

NeuralNetwork *nn_load(const char *filename) {
    if (!filename) return NULL;

    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;

    int input_size, hidden_size, output_size;
    if (fread(&input_size, sizeof(int), 1, f) != 1 ||
        fread(&hidden_size, sizeof(int), 1, f) != 1 ||
        fread(&output_size, sizeof(int), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    NeuralNetwork *nn = nn_create(input_size, hidden_size, output_size);
    if (!nn) {
        fclose(f);
        return NULL;
    }

    if (fread(nn->weights_ih, sizeof(double), input_size * hidden_size, f) != (size_t)(input_size * hidden_size) ||
        fread(nn->weights_ho, sizeof(double), hidden_size * output_size, f) != (size_t)(hidden_size * output_size) ||
        fread(nn->bias_h, sizeof(double), hidden_size, f) != (size_t)hidden_size ||
        fread(nn->bias_o, sizeof(double), output_size, f) != (size_t)output_size) {
        nn_free(nn);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return nn;
}
