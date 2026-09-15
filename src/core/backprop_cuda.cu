#include "backprop.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define NEGATIVE_COUNT 32

static inline void cuda_check(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

__device__ static inline void atomic_add_double(double *address, double value) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, value);
#else
    unsigned long long *as_ull = (unsigned long long *)address;
    unsigned long long old = *as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(as_ull, assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

__device__ static double device_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__global__ static void train_kernel(
    const int *word_ids, const int *role_ids, const double *embeddings,
    const double *weights_ih, const double *weights_ho,
    const double *bias_h, const double *bias_o,
    double *grad_ih, double *grad_ho, double *grad_bias_h,
    double *grad_bias_o, double *grad_embeddings, double *loss,
    int triangle_count, int sample_start, int sample_end,
    int embed_dim, int input_size, int hidden_size, int output_size,
    int vocab_size) {
    int sample = sample_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= sample_end) return;

    int triangle = sample / 3;
    int target_position = sample % 3;
    int source_a = (target_position + 1) % 3;
    int source_b = (target_position + 2) % 3;

    double input[128];
    double hidden[256];
    double output[512];
    double error_o[512];
    double error_h[256];
    int candidates[NEGATIVE_COUNT + 1];
    double candidate_scores[NEGATIVE_COUNT + 1];
    double candidate_probs[NEGATIVE_COUNT + 1];

    if (input_size > 128 || hidden_size > 256 || output_size > 512) return;

    int word_a = word_ids[triangle * 3 + source_a];
    int word_b = word_ids[triangle * 3 + source_b];
    int feature_dim = embed_dim + TRIANGLE_ROLE_FEATURE_DIM;
    for (int d = 0; d < embed_dim; d++) {
        input[d] = embeddings[word_a * embed_dim + d];
        input[feature_dim + d] = embeddings[word_b * embed_dim + d];
    }
    for (int r = 0; r < TRIANGLE_ROLE_FEATURE_DIM; r++) {
        input[embed_dim + r] =
            role_ids[triangle * 3 + source_a] == r + 1 ? 1.0 : 0.0;
        input[feature_dim + embed_dim + r] =
            role_ids[triangle * 3 + source_b] == r + 1 ? 1.0 : 0.0;
    }

    for (int h = 0; h < hidden_size; h++) {
        double value = bias_h[h];
        for (int i = 0; i < input_size; i++) {
            value += input[i] * weights_ih[i * hidden_size + h];
        }
        hidden[h] = device_sigmoid(value);
    }

    for (int o = 0; o < output_size; o++) {
        double value = bias_o[o];
        for (int h = 0; h < hidden_size; h++) {
            value += hidden[h] * weights_ho[h * output_size + o];
        }
        output[o] = value;
        error_o[o] = 0.0;
    }

    int target_word = word_ids[triangle * 3 + target_position];
    int target_start = target_position * embed_dim;
    candidates[0] = target_word;
    for (int k = 0; k < NEGATIVE_COUNT; k++) {
        unsigned long long seed = (unsigned long long)(sample + 1) * 2862933555777941757ULL;
        seed += (unsigned long long)(k + 1) * 3037000493ULL;
        int candidate = (int)(seed % (unsigned long long)vocab_size);
        if (candidate == target_word) candidate = (candidate + 1) % vocab_size;
        candidates[k + 1] = candidate;
    }

    double max_score = -1.0e300;
    for (int k = 0; k <= NEGATIVE_COUNT; k++) {
        double score = 0.0;
        for (int d = 0; d < embed_dim; d++) {
            score += output[target_start + d] * embeddings[candidates[k] * embed_dim + d];
        }
        candidate_scores[k] = score;
        if (score > max_score) max_score = score;
    }

    double denominator = 0.0;
    for (int k = 0; k <= NEGATIVE_COUNT; k++) {
        candidate_probs[k] = exp(candidate_scores[k] - max_score);
        denominator += candidate_probs[k];
    }
    for (int k = 0; k <= NEGATIVE_COUNT; k++) candidate_probs[k] /= denominator;
    atomic_add_double(loss, -log(candidate_probs[0]));

    for (int d = 0; d < embed_dim; d++) {
        double expected = 0.0;
        for (int k = 0; k <= NEGATIVE_COUNT; k++) {
            expected += candidate_probs[k] * embeddings[candidates[k] * embed_dim + d];
        }
        error_o[target_start + d] =
            embeddings[target_word * embed_dim + d] - expected;
    }

    /* Contrastive gradients for the target and sampled negative embeddings. */
    for (int k = 0; k <= NEGATIVE_COUNT; k++) {
        double coefficient = (k == 0 ? 1.0 : 0.0) - candidate_probs[k];
        for (int d = 0; d < embed_dim; d++) {
            atomic_add_double(
                &grad_embeddings[candidates[k] * embed_dim + d],
                coefficient * output[target_start + d]);
        }
    }

    for (int h = 0; h < hidden_size; h++) {
        double value = 0.0;
        for (int o = target_start; o < target_start + embed_dim; o++) {
            value += error_o[o] * weights_ho[h * output_size + o];
        }
        error_h[h] = value * hidden[h] * (1.0 - hidden[h]);
        atomic_add_double(&grad_bias_h[h], error_h[h]);
    }

    for (int i = 0; i < input_size; i++) {
        double input_error = 0.0;
        for (int h = 0; h < hidden_size; h++) {
            input_error += error_h[h] * weights_ih[i * hidden_size + h];
            atomic_add_double(&grad_ih[i * hidden_size + h], error_h[h] * input[i]);
        }
        int source_word = i < embed_dim ? word_a : word_b;
        int dimension = i < embed_dim ? i : i - embed_dim;
        atomic_add_double(&grad_embeddings[source_word * embed_dim + dimension], input_error);
    }

    for (int o = target_start; o < target_start + embed_dim; o++) {
        atomic_add_double(&grad_bias_o[o], error_o[o]);
        for (int h = 0; h < hidden_size; h++) {
            atomic_add_double(&grad_ho[h * output_size + o], error_o[o] * hidden[h]);
        }
    }
}

struct DeviceState {
    int device;
    int *word_ids;
    int *role_ids;
    double *embeddings, *weights_ih, *weights_ho, *bias_h, *bias_o;
    double *grad_ih, *grad_ho, *grad_bias_h, *grad_bias_o, *grad_embeddings, *loss;
    double *host_grad_ih, *host_grad_ho, *host_grad_bias_h, *host_grad_bias_o, *host_grad_embeddings;
    double host_loss;
};

static void normalize_embeddings(BackpropNetwork *network) {
    for (int word = 0; word < network->vocab_size; word++) {
        double norm = 0.0;
        for (int d = 0; d < network->embed_dim; d++) {
            double value = network->embeddings[word * network->embed_dim + d];
            norm += value * value;
        }
        norm = sqrt(norm);
        if (norm < 1.0e-12) continue;
        for (int d = 0; d < network->embed_dim; d++) {
            network->embeddings[word * network->embed_dim + d] /= norm;
        }
    }
}

static void allocate_state(DeviceState *state, int device,
                           const TriangleChain *chain, const BackpropNetwork *nn,
                           const int *word_ids, const int *role_ids) {
    state->device = device;
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
    size_t words_bytes = chain->count * 3 * sizeof(int);
    size_t embedding_bytes = nn->vocab_size * nn->embed_dim * sizeof(double);
    size_t ih_bytes = nn->input_size * nn->hidden_size * sizeof(double);
    size_t ho_bytes = nn->hidden_size * nn->output_size * sizeof(double);
    size_t bh_bytes = nn->hidden_size * sizeof(double);
    size_t bo_bytes = nn->output_size * sizeof(double);

    cuda_check(cudaMalloc(&state->word_ids, words_bytes), "word_ids");
    cuda_check(cudaMalloc(&state->role_ids, words_bytes), "role_ids");
    cuda_check(cudaMalloc(&state->embeddings, embedding_bytes), "embeddings");
    cuda_check(cudaMalloc(&state->weights_ih, ih_bytes), "weights_ih");
    cuda_check(cudaMalloc(&state->weights_ho, ho_bytes), "weights_ho");
    cuda_check(cudaMalloc(&state->bias_h, bh_bytes), "bias_h");
    cuda_check(cudaMalloc(&state->bias_o, bo_bytes), "bias_o");
    cuda_check(cudaMalloc(&state->grad_ih, ih_bytes), "grad_ih");
    cuda_check(cudaMalloc(&state->grad_ho, ho_bytes), "grad_ho");
    cuda_check(cudaMalloc(&state->grad_bias_h, bh_bytes), "grad_bias_h");
    cuda_check(cudaMalloc(&state->grad_bias_o, bo_bytes), "grad_bias_o");
    cuda_check(cudaMalloc(&state->grad_embeddings, embedding_bytes), "grad_embeddings");
    cuda_check(cudaMalloc(&state->loss, sizeof(double)), "loss");

    cuda_check(cudaMemcpy(state->word_ids, word_ids, words_bytes, cudaMemcpyHostToDevice), "copy word_ids");
    cuda_check(cudaMemcpy(state->role_ids, role_ids, words_bytes, cudaMemcpyHostToDevice), "copy role_ids");
    cuda_check(cudaMemcpy(state->embeddings, nn->embeddings, embedding_bytes, cudaMemcpyHostToDevice), "copy embeddings");

    state->host_grad_ih = (double *)malloc(ih_bytes);
    state->host_grad_ho = (double *)malloc(ho_bytes);
    state->host_grad_bias_h = (double *)malloc(bh_bytes);
    state->host_grad_bias_o = (double *)malloc(bo_bytes);
    state->host_grad_embeddings = (double *)malloc(embedding_bytes);
}

static void free_state(DeviceState *state) {
    cudaSetDevice(state->device);
    cudaFree(state->word_ids); cudaFree(state->embeddings);
    cudaFree(state->role_ids);
    cudaFree(state->weights_ih); cudaFree(state->weights_ho);
    cudaFree(state->bias_h); cudaFree(state->bias_o);
    cudaFree(state->grad_ih); cudaFree(state->grad_ho);
    cudaFree(state->grad_bias_h); cudaFree(state->grad_bias_o);
    cudaFree(state->grad_embeddings);
    cudaFree(state->loss);
    free(state->host_grad_ih); free(state->host_grad_ho);
    free(state->host_grad_bias_h); free(state->host_grad_bias_o);
    free(state->host_grad_embeddings);
}

extern "C" int backprop_train_cuda(BackpropTrainer *trainer,
                                     const TriangleChain *chain,
                                     int requested_gpus) {
    if (!trainer || !chain || chain->count == 0) return -1;
    BackpropNetwork *nn = trainer->network;
    int available = 0;
    cuda_check(cudaGetDeviceCount(&available), "cudaGetDeviceCount");
    int gpu_count = requested_gpus < available ? requested_gpus : available;
    if (gpu_count < 1) {
        fprintf(stderr, "No CUDA GPUs available.\n");
        return -1;
    }
    if (gpu_count > 2) gpu_count = 2;

    size_t triangle_count = chain->count;
    int *word_ids = (int *)malloc(triangle_count * 3 * sizeof(int));
    int *role_ids = (int *)malloc(triangle_count * 3 * sizeof(int));
    for (size_t t = 0; t < triangle_count; t++) {
        for (int p = 0; p < 3; p++) {
            int id = chain->triangles[t].word_ids[p] - 1;
            if (id < 0) id = 0;
            if (id >= nn->vocab_size) id = nn->vocab_size - 1;
            word_ids[t * 3 + p] = id;
            role_ids[t * 3 + p] = chain->triangles[t].role_ids[p];
        }
    }

    DeviceState *states = (DeviceState *)calloc(gpu_count, sizeof(DeviceState));
    for (int g = 0; g < gpu_count; g++)
        allocate_state(&states[g], g, chain, nn, word_ids, role_ids);

    size_t ih_count = nn->input_size * nn->hidden_size;
    size_t ho_count = nn->hidden_size * nn->output_size;
    double *sum_ih = (double *)calloc(ih_count, sizeof(double));
    double *sum_ho = (double *)calloc(ho_count, sizeof(double));
    double *sum_bh = (double *)calloc(nn->hidden_size, sizeof(double));
    double *sum_bo = (double *)calloc(nn->output_size, sizeof(double));
    size_t embedding_count = nn->vocab_size * nn->embed_dim;
    double *sum_embeddings = (double *)calloc(embedding_count, sizeof(double));
    int sample_count = (int)(triangle_count * 3);
    int block_size = 256;

    printf("CUDA sampled-contrastive training: %d GPU(s), %d rotation samples, %d negatives, block size %d\n",
           gpu_count, sample_count, NEGATIVE_COUNT, block_size);
    fflush(stdout);

    for (int epoch = 0; epoch < trainer->max_epochs; epoch++) {
        memset(sum_ih, 0, ih_count * sizeof(double));
        memset(sum_ho, 0, ho_count * sizeof(double));
        memset(sum_bh, 0, nn->hidden_size * sizeof(double));
        memset(sum_bo, 0, nn->output_size * sizeof(double));
        double total_loss = 0.0;

        for (int g = 0; g < gpu_count; g++) {
            int start = (sample_count * g) / gpu_count;
            int end = (sample_count * (g + 1)) / gpu_count;
            DeviceState *state = &states[g];
            cuda_check(cudaSetDevice(state->device), "cudaSetDevice");
            cuda_check(cudaMemcpy(state->weights_ih, nn->weights_ih, ih_count * sizeof(double), cudaMemcpyHostToDevice), "weights_ih");
            cuda_check(cudaMemcpy(state->weights_ho, nn->weights_ho, ho_count * sizeof(double), cudaMemcpyHostToDevice), "weights_ho");
            cuda_check(cudaMemcpy(state->bias_h, nn->bias_h, nn->hidden_size * sizeof(double), cudaMemcpyHostToDevice), "bias_h");
            cuda_check(cudaMemcpy(state->bias_o, nn->bias_o, nn->output_size * sizeof(double), cudaMemcpyHostToDevice), "bias_o");
            cuda_check(cudaMemcpy(state->embeddings, nn->embeddings, embedding_count * sizeof(double), cudaMemcpyHostToDevice), "embeddings");
            cuda_check(cudaMemset(state->grad_ih, 0, ih_count * sizeof(double)), "clear grad_ih");
            cuda_check(cudaMemset(state->grad_ho, 0, ho_count * sizeof(double)), "clear grad_ho");
            cuda_check(cudaMemset(state->grad_bias_h, 0, nn->hidden_size * sizeof(double)), "clear grad_bias_h");
            cuda_check(cudaMemset(state->grad_bias_o, 0, nn->output_size * sizeof(double)), "clear grad_bias_o");
            cuda_check(cudaMemset(state->grad_embeddings, 0, embedding_count * sizeof(double)), "clear grad_embeddings");
            cuda_check(cudaMemset(state->loss, 0, sizeof(double)), "clear loss");

            int blocks = (end - start + block_size - 1) / block_size;
            train_kernel<<<blocks, block_size>>>(
                state->word_ids, state->role_ids, state->embeddings, state->weights_ih,
                state->weights_ho, state->bias_h, state->bias_o,
                state->grad_ih, state->grad_ho, state->grad_bias_h,
                state->grad_bias_o, state->grad_embeddings, state->loss, (int)triangle_count,
                start, end, nn->embed_dim, nn->input_size,
                nn->hidden_size, nn->output_size, nn->vocab_size);
            cuda_check(cudaGetLastError(), "train_kernel launch");
            cuda_check(cudaDeviceSynchronize(), "train_kernel synchronize");
            cuda_check(cudaMemcpy(state->host_grad_ih, state->grad_ih, ih_count * sizeof(double), cudaMemcpyDeviceToHost), "copy grad_ih");
            cuda_check(cudaMemcpy(state->host_grad_ho, state->grad_ho, ho_count * sizeof(double), cudaMemcpyDeviceToHost), "copy grad_ho");
            cuda_check(cudaMemcpy(state->host_grad_bias_h, state->grad_bias_h, nn->hidden_size * sizeof(double), cudaMemcpyDeviceToHost), "copy grad_bias_h");
            cuda_check(cudaMemcpy(state->host_grad_bias_o, state->grad_bias_o, nn->output_size * sizeof(double), cudaMemcpyDeviceToHost), "copy grad_bias_o");
            cuda_check(cudaMemcpy(state->host_grad_embeddings, state->grad_embeddings, embedding_count * sizeof(double), cudaMemcpyDeviceToHost), "copy grad_embeddings");
            cuda_check(cudaMemcpy(&state->host_loss, state->loss, sizeof(double), cudaMemcpyDeviceToHost), "copy loss");

            for (size_t i = 0; i < ih_count; i++) sum_ih[i] += state->host_grad_ih[i];
            for (size_t i = 0; i < ho_count; i++) sum_ho[i] += state->host_grad_ho[i];
            for (int i = 0; i < nn->hidden_size; i++) sum_bh[i] += state->host_grad_bias_h[i];
            for (int i = 0; i < nn->output_size; i++) sum_bo[i] += state->host_grad_bias_o[i];
            for (size_t i = 0; i < embedding_count; i++) sum_embeddings[i] += state->host_grad_embeddings[i];
            total_loss += state->host_loss;
        }

        double scale = trainer->learning_rate / sample_count;
        for (size_t i = 0; i < ih_count; i++) nn->weights_ih[i] += scale * sum_ih[i];
        for (size_t i = 0; i < ho_count; i++) nn->weights_ho[i] += scale * sum_ho[i];
        for (int i = 0; i < nn->hidden_size; i++) nn->bias_h[i] += scale * sum_bh[i];
        for (int i = 0; i < nn->output_size; i++) nn->bias_o[i] += scale * sum_bo[i];
        for (size_t i = 0; i < embedding_count; i++) nn->embeddings[i] += scale * sum_embeddings[i];
        normalize_embeddings(nn);

        trainer->logs[epoch].epoch = epoch;
        trainer->logs[epoch].loss = total_loss / sample_count;
        trainer->logs[epoch].accuracy = 0.0;
        trainer->log_count++;
        if (epoch % 5 == 0 || epoch == trainer->max_epochs - 1) {
            printf("Epoch %4d: loss=%.6f\n", epoch, trainer->logs[epoch].loss);
            fflush(stdout);
        }
    }

    for (int g = 0; g < gpu_count; g++) free_state(&states[g]);
    free(states); free(word_ids); free(role_ids); free(sum_ih); free(sum_ho); free(sum_bh); free(sum_bo); free(sum_embeddings);
    return 0;
}
