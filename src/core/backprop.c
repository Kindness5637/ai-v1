#include "backprop.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

static double random_weight(void) {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

/* Triangle vocabulary IDs are one-based; model arrays are zero-based. */
static int model_word_id(int vocabulary_id, int vocab_size) {
    if (vocabulary_id <= 0) return 0; /* padding */
    if (vocabulary_id > vocab_size) return vocab_size - 1;
    return vocabulary_id - 1;
}

static void fill_token_features(double *input, int offset, int word_id,
                                int role_id, const BackpropNetwork *network) {
    int feature_dim = network->embed_dim + TRIANGLE_ROLE_FEATURE_DIM;
    int word_index = model_word_id(word_id, network->vocab_size);
    for (int d = 0; d < network->embed_dim; d++) {
        input[offset * feature_dim + d] =
            network->embeddings[word_index * network->embed_dim + d];
    }
    for (int r = 0; r < TRIANGLE_ROLE_FEATURE_DIM; r++) {
        input[offset * feature_dim + network->embed_dim + r] =
            (role_id == r + 1) ? 1.0 : 0.0;
    }
}

BackpropTrainer *backprop_create(int vocab_size, int embed_dim, int hidden_size, int output_size, int max_epochs, double lr) {
    BackpropTrainer *trainer = malloc(sizeof(BackpropTrainer));
    if (!trainer) return NULL;

    trainer->max_epochs = max_epochs;
    trainer->learning_rate = lr;
    trainer->log_count = 0;
    trainer->logs = malloc(max_epochs * sizeof(TrainingLog));
    if (!trainer->logs) {
        free(trainer);
        return NULL;
    }

    trainer->network = malloc(sizeof(BackpropNetwork));
    if (!trainer->network) {
        free(trainer->logs);
        free(trainer);
        return NULL;
    }

    int input_size = 2 * (embed_dim + TRIANGLE_ROLE_FEATURE_DIM);
    trainer->network->vocab_size = vocab_size;
    trainer->network->embed_dim = embed_dim;
    trainer->network->input_size = input_size;
    trainer->network->hidden_size = hidden_size;
    /* output_size is the flattened output dimension (3 * embed_dim). */
    trainer->network->output_size = output_size;

    trainer->network->embeddings = malloc(vocab_size * embed_dim * sizeof(double));
    trainer->network->weights_ih = malloc(input_size * hidden_size * sizeof(double));
    trainer->network->weights_ho = malloc(hidden_size * output_size * sizeof(double));
    trainer->network->bias_h = malloc(hidden_size * sizeof(double));
    trainer->network->bias_o = malloc(output_size * sizeof(double));

    if (!trainer->network->embeddings || !trainer->network->weights_ih || !trainer->network->weights_ho ||
        !trainer->network->bias_h || !trainer->network->bias_o) {
        free(trainer->network->embeddings);
        free(trainer->network->weights_ih);
        free(trainer->network->weights_ho);
        free(trainer->network->bias_h);
        free(trainer->network->bias_o);
        free(trainer->network);
        free(trainer->logs);
        free(trainer);
        return NULL;
    }

    srand(time(NULL));
    double embed_scale = 1.0 / sqrt(embed_dim);
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        trainer->network->embeddings[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * embed_scale;
    }
    for (int i = 0; i < input_size * hidden_size; i++) {
        trainer->network->weights_ih[i] = random_weight() / sqrt(input_size);
    }
    for (int i = 0; i < hidden_size * output_size; i++) {
        trainer->network->weights_ho[i] = random_weight() / sqrt(hidden_size);
    }
    for (int i = 0; i < hidden_size; i++) {
        trainer->network->bias_h[i] = 0.0;
    }
    for (int i = 0; i < output_size; i++) {
        trainer->network->bias_o[i] = 0.0;
    }

    return trainer;
}

void backprop_free(BackpropTrainer *trainer) {
    if (!trainer) return;
    if (trainer->network) {
        free(trainer->network->embeddings);
        free(trainer->network->weights_ih);
        free(trainer->network->weights_ho);
        free(trainer->network->bias_h);
        free(trainer->network->bias_o);
        free(trainer->network);
    }
    free(trainer->logs);
    free(trainer);
}

void backprop_train(BackpropTrainer *trainer, const TriangleChain *chain) {
    if (!trainer || !chain) return;

    BackpropNetwork *nn = trainer->network;
    int vocab_size = chain->vocab->count;

    printf("Training backpropagation with Word Embeddings...\n");
    printf("Embedding Dim: %d | Input Dim: %d | Hidden: %d | Output vectors: %d\n",
           nn->embed_dim, nn->input_size, nn->hidden_size, nn->output_size);
    printf("Vocabulary: %d words\n", vocab_size);
    printf("Threads: %d\n\n", omp_get_max_threads());

    int batch_size = 256;
    int num_threads = omp_get_max_threads();

    for (int epoch = 0; epoch < trainer->max_epochs; epoch++) {
        double total_loss = 0;
        int correct = 0;

        for (size_t batch_start = 0; batch_start < chain->count; batch_start += batch_size) {
            size_t batch_end = batch_start + batch_size;
            if (batch_end > chain->count) batch_end = chain->count;
            size_t cur_batch_size = batch_end - batch_start;

            double *grad_ih = calloc(num_threads * nn->input_size * nn->hidden_size, sizeof(double));
            double *grad_ho = calloc(num_threads * nn->hidden_size * nn->output_size, sizeof(double));
            double *grad_bias_h = calloc(num_threads * nn->hidden_size, sizeof(double));
            double *grad_bias_o = calloc(num_threads * nn->output_size, sizeof(double));
            double *thread_loss = calloc(num_threads, sizeof(double));
            int *thread_correct = calloc(num_threads, sizeof(int));

            double scale = trainer->learning_rate / (double)(cur_batch_size * 3);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                double *local_input = malloc(nn->input_size * sizeof(double));
                double *local_hidden = malloc(nn->hidden_size * sizeof(double));
                double *local_output = malloc(nn->output_size * sizeof(double));

                #pragma omp for schedule(static)
                for (size_t t = batch_start; t < batch_end; t++) {
                    int w_ids[3];
                    for (int p = 0; p < 3; p++) {
                        w_ids[p] = model_word_id(chain->triangles[t].word_ids[p], vocab_size);
                    }

                    for (int target_position = 0; target_position < 3; target_position++) {
                        for (int q = 0; q < 2; q++) {
                            int source_position = (target_position + q + 1) % 3;
                            fill_token_features(local_input, q,
                                chain->triangles[t].word_ids[source_position],
                                chain->triangles[t].role_ids[source_position], nn);
                        }

                        double target[nn->output_size];
                        memset(target, 0, nn->output_size * sizeof(double));
                        for (int d = 0; d < nn->embed_dim; d++) {
                            target[target_position * nn->embed_dim + d] =
                                nn->embeddings[w_ids[target_position] * nn->embed_dim + d];
                        }

                        for (int h = 0; h < nn->hidden_size; h++) {
                            local_hidden[h] = 0;
                            for (int i = 0; i < nn->input_size; i++) {
                                local_hidden[h] += local_input[i] * nn->weights_ih[i * nn->hidden_size + h];
                            }
                            local_hidden[h] = sigmoid(local_hidden[h] + nn->bias_h[h]);
                        }

                        for (int o = 0; o < nn->output_size; o++) {
                            local_output[o] = 0;
                            for (int h = 0; h < nn->hidden_size; h++) {
                                local_output[o] += local_hidden[h] * nn->weights_ho[h * nn->output_size + o];
                            }
                            local_output[o] += nn->bias_o[o];
                        }

                        double local_error_o[nn->output_size];
                        for (int o = 0; o < nn->output_size; o++) {
                            double target_val = target[o];
                            int active_start = target_position * nn->embed_dim;
                            int active_end = active_start + nn->embed_dim;
                            if (o >= active_start && o < active_end) {
                                local_error_o[o] = target_val - local_output[o];
                                thread_loss[tid] += pow(target_val - local_output[o], 2);
                            } else {
                                local_error_o[o] = 0.0;
                            }
                        }

                        double local_error_h[nn->hidden_size];
                        for (int h = 0; h < nn->hidden_size; h++) {
                            local_error_h[h] = 0;
                            for (int o = 0; o < nn->output_size; o++) {
                                local_error_h[h] += local_error_o[o] * nn->weights_ho[h * nn->output_size + o];
                            }
                            local_error_h[h] *= sigmoid_derivative(local_hidden[h]);
                        }

                        for (int h = 0; h < nn->hidden_size; h++) {
                            for (int i = 0; i < nn->input_size; i++) {
                                grad_ih[tid * nn->input_size * nn->hidden_size + i * nn->hidden_size + h] += local_error_h[h] * local_input[i];
                            }
                            grad_bias_h[tid * nn->hidden_size + h] += local_error_h[h];
                        }

                        for (int o = 0; o < nn->output_size; o++) {
                            for (int h = 0; h < nn->hidden_size; h++) {
                                grad_ho[tid * nn->hidden_size * nn->output_size + h * nn->output_size + o] += local_error_o[o] * local_hidden[h];
                            }
                            grad_bias_o[tid * nn->output_size + o] += local_error_o[o];
                        }
                    }

                    /* Keep the target space fixed. The decoder learns to map
                     * input embeddings back into this vocabulary space. */

                    /* Nearest-neighbor decoding is intentionally done at evaluation
                     * time; scanning the full vocabulary for every training sample
                     * would dominate the training cost. */
                }

                free(local_input);
                free(local_hidden);
                free(local_output);
            }

            for (int t = 0; t < num_threads; t++) {
                total_loss += thread_loss[t];
                correct += thread_correct[t];
                for (int i = 0; i < nn->input_size * nn->hidden_size; i++) {
                    nn->weights_ih[i] += scale * grad_ih[t * nn->input_size * nn->hidden_size + i];
                }
                for (int h = 0; h < nn->hidden_size; h++) {
                    nn->bias_h[h] += scale * grad_bias_h[t * nn->hidden_size + h];
                }
                for (int i = 0; i < nn->hidden_size * nn->output_size; i++) {
                    nn->weights_ho[i] += scale * grad_ho[t * nn->hidden_size * nn->output_size + i];
                }
                for (int o = 0; o < nn->output_size; o++) {
                    nn->bias_o[o] += scale * grad_bias_o[t * nn->output_size + o];
                }
            }

            free(grad_ih);
            free(grad_ho);
            free(grad_bias_h);
            free(grad_bias_o);
            free(thread_loss);
            free(thread_correct);
        }

        double avg_loss = total_loss / (chain->count * nn->output_size);
        trainer->logs[epoch].epoch = epoch;
        trainer->logs[epoch].loss = avg_loss;
        trainer->logs[epoch].accuracy = 0.0;
        trainer->log_count++;

        if (epoch % 5 == 0 || epoch == trainer->max_epochs - 1) {
            printf("Epoch %4d: loss=%.6f\n", epoch, avg_loss);
            fflush(stdout);
        }
    }
}

int backprop_predict(BackpropTrainer *trainer, double *input, double *output) {
    if (!trainer || !input || !output) return -1;

    BackpropNetwork *nn = trainer->network;

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
    for (int o = 1; o < 3; o++) {
        if (output[o] > output[best]) best = o;
    }

    return best;
}

void backprop_print_logs(const BackpropTrainer *trainer) {
    if (!trainer) return;

    printf("\n=== Training Summary ===\n");
    printf("Total epochs: %d\n", trainer->log_count);
    if (trainer->log_count > 0) {
        printf("Final loss: %.6f\n", trainer->logs[trainer->log_count - 1].loss);
    }
}

void backprop_save_model(const BackpropTrainer *trainer, const char *filename) {
    if (!trainer || !filename) return;

    FILE *f = fopen(filename, "wb");
    if (!f) return;

    fwrite(&trainer->network->vocab_size, sizeof(int), 1, f);
    fwrite(&trainer->network->embed_dim, sizeof(int), 1, f);
    fwrite(&trainer->network->hidden_size, sizeof(int), 1, f);
    fwrite(&trainer->network->output_size, sizeof(int), 1, f);
    fwrite(trainer->network->embeddings, sizeof(double), trainer->network->vocab_size * trainer->network->embed_dim, f);
    fwrite(trainer->network->weights_ih, sizeof(double), trainer->network->input_size * trainer->network->hidden_size, f);
    fwrite(trainer->network->weights_ho, sizeof(double), trainer->network->hidden_size * trainer->network->output_size, f);
    fwrite(trainer->network->bias_h, sizeof(double), trainer->network->hidden_size, f);
    fwrite(trainer->network->bias_o, sizeof(double), trainer->network->output_size, f);

    fclose(f);
    printf("Model saved to %s\n", filename);
}

BackpropTrainer *backprop_load_model(const char *filename) {
    if (!filename) return NULL;

    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;

    int vocab_size, embed_dim, hidden_size, output_size;
    if (fread(&vocab_size, sizeof(int), 1, f) != 1 ||
        fread(&embed_dim, sizeof(int), 1, f) != 1 ||
        fread(&hidden_size, sizeof(int), 1, f) != 1 ||
        fread(&output_size, sizeof(int), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    BackpropTrainer *trainer = backprop_create(vocab_size, embed_dim, hidden_size, output_size, 1, 0.1);
    if (!trainer) {
        fclose(f);
        return NULL;
    }

    if (fread(trainer->network->embeddings, sizeof(double), vocab_size * embed_dim, f) != (size_t)(vocab_size * embed_dim) ||
        fread(trainer->network->weights_ih, sizeof(double), trainer->network->input_size * hidden_size, f) != (size_t)(trainer->network->input_size * hidden_size) ||
        fread(trainer->network->weights_ho, sizeof(double), hidden_size * output_size, f) != (size_t)(hidden_size * output_size) ||
        fread(trainer->network->bias_h, sizeof(double), hidden_size, f) != (size_t)hidden_size ||
        fread(trainer->network->bias_o, sizeof(double), output_size, f) != (size_t)output_size) {
        backprop_free(trainer);
        fclose(f);
        return NULL;
    }

    fclose(f);
    printf("Model loaded from %s\n", filename);
    return trainer;
}

typedef struct {
    int old_id;
    int count;
} WordFreq;

static int compare_freq(const void *a, const void *b) {
    return ((WordFreq*)b)->count - ((WordFreq*)a)->count;
}

TriangleChain *backprop_reduce_vocab(const TriangleChain *chain, int top_n) {
    if (!chain || !chain->vocab || top_n <= 0) return NULL;

    int total_words = chain->vocab->count;
    if (total_words == 0) return NULL;

    WordFreq *freq = malloc(total_words * sizeof(WordFreq));
    if (!freq) return NULL;

    for (int i = 0; i < total_words; i++) {
        freq[i].old_id = i;
        freq[i].count = 0;
    }

    for (size_t i = 0; i < chain->count; i++) {
        Triangle *t = &chain->triangles[i];
        for (int p = 0; p < 3; p++) {
            int wid = t->word_ids[p];
            if (wid >= 1 && wid <= total_words) {
                freq[wid - 1].count++;
            }
        }
    }

    qsort(freq, total_words, sizeof(WordFreq), compare_freq);

    int new_vocab_size = (top_n < total_words) ? top_n : total_words;

    int *id_map = malloc((total_words + 1) * sizeof(int));
    if (!id_map) {
        free(freq);
        return NULL;
    }

    for (int i = 0; i <= total_words; i++) {
        id_map[i] = -1;
    }

    for (int i = 0; i < new_vocab_size; i++) {
        int old_id = freq[i].old_id + 1;
        id_map[old_id] = i;
    }

    TriangleChain *new_chain = malloc(sizeof(TriangleChain));
    if (!new_chain) {
        free(freq);
        free(id_map);
        return NULL;
    }

    new_chain->triangles = malloc(chain->count * sizeof(Triangle));
    if (!new_chain->triangles) {
        free(freq);
        free(id_map);
        free(new_chain);
        return NULL;
    }

    new_chain->count = 0;

    new_chain->vocab = vocab_create();
    if (!new_chain->vocab) {
        free(freq);
        free(id_map);
        free(new_chain->triangles);
        free(new_chain);
        return NULL;
    }

    new_chain->vocab->words = malloc(new_vocab_size * sizeof(Word));
    new_chain->vocab->capacity = new_vocab_size;
    if (!new_chain->vocab->words) {
        free(freq);
        free(id_map);
        free(new_chain->triangles);
        free(new_chain->vocab);
        free(new_chain);
        return NULL;
    }

    for (int i = 0; i < new_vocab_size; i++) {
        int old_id = freq[i].old_id + 1;
        const char *word_text = vocab_get_word(chain->vocab, old_id);
        new_chain->vocab->words[i].id = i;
        new_chain->vocab->words[i].text = strdup(word_text ? word_text : "");
    }
    new_chain->vocab->count = new_vocab_size;

    new_chain->registry = registry_create();

    for (size_t i = 0; i < chain->count; i++) {
        Triangle *t = &chain->triangles[i];

        int new_a = id_map[t->word_ids[0]];
        int new_b = id_map[t->word_ids[1]];
        int new_c = id_map[t->word_ids[2]];

        if (new_a >= 0 && new_b >= 0 && new_c >= 0) {
            Triangle *new_t = &new_chain->triangles[new_chain->count];
            new_t->id = t->id;
            new_t->words[0] = strdup(t->words[0]);
            new_t->words[1] = strdup(t->words[1]);
            new_t->words[2] = strdup(t->words[2]);
            new_t->word_ids[0] = new_a;
            new_t->word_ids[1] = new_b;
            new_t->word_ids[2] = new_c;
            new_chain->count++;
        }
    }

    printf("Reduced vocabulary: %d -> %d words (%zu triangles kept)\n",
           total_words, new_vocab_size, new_chain->count);

    free(freq);
    free(id_map);
    return new_chain;
}
