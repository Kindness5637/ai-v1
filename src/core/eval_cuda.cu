#include "eval_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <climits>

struct DeviceThresholdResult {
    unsigned long long gold_recovered;
    unsigned long long fully_supported_recovered;
    unsigned long long candidates_emitted;
};

static int transition_compare(const void *a, const void *b) {
    const RelationalTransition *x = (const RelationalTransition *)a;
    const RelationalTransition *y = (const RelationalTransition *)b;
    if (x->from_word_id != y->from_word_id)
        return x->from_word_id - y->from_word_id;
    if (x->transition_type != y->transition_type)
        return x->transition_type - y->transition_type;
    if (x->to_word_id != y->to_word_id)
        return x->to_word_id - y->to_word_id;
    return 0;
}

__device__ static uint64_t transition_count_device(
    const RelationalTransition *transitions, int transition_count,
    int from_id, int to_id, int transition_type) {
    int low = 0;
    int high = transition_count - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        RelationalTransition item = transitions[mid];
        if (item.from_word_id < from_id ||
            (item.from_word_id == from_id && item.transition_type < transition_type) ||
            (item.from_word_id == from_id && item.transition_type == transition_type &&
             item.to_word_id < to_id)) {
            low = mid + 1;
        } else if (item.from_word_id > from_id ||
                   (item.from_word_id == from_id && item.transition_type > transition_type) ||
                   (item.from_word_id == from_id && item.transition_type == transition_type &&
                    item.to_word_id > to_id)) {
            high = mid - 1;
        } else {
            return item.count;
        }
    }
    return 0;
}

__global__ static void threshold_sweep_kernel(
    const ThresholdSweepQuery *queries, int query_count,
    const RelationalWordStats *word_stats, int vocab_size,
    const RelationalTransition *transitions, int transition_count,
    const double *position_thresholds, int position_threshold_count,
    const uint64_t *transition_thresholds, int transition_threshold_count,
    DeviceThresholdResult *results) {
    int work = blockIdx.x * blockDim.x + threadIdx.x;
    int config_count = position_threshold_count * transition_threshold_count;
    int total_work = query_count * config_count;
    if (work >= total_work) return;

    int query_index = work / config_count;
    int config = work % config_count;
    int position_index = config / transition_threshold_count;
    int transition_index = config % transition_threshold_count;
    const ThresholdSweepQuery query = queries[query_index];
    double position_threshold = position_thresholds[position_index];
    uint64_t transition_threshold = transition_thresholds[transition_index];

    unsigned long long emitted = 0;
    unsigned long long recovered = 0;
    for (int word_id = 1; word_id <= vocab_size; word_id++) {
        const RelationalWordStats stats = word_stats[word_id];
        double total = (double)(stats.left_count + stats.center_count + stats.right_count);
        if (total <= 0.0) continue;

        uint64_t position_count = query.target_position == 0 ? stats.left_count :
                                   query.target_position == 1 ? stats.center_count :
                                                                stats.right_count;
        double position_ratio = (double)position_count / total;
        if (position_ratio < position_threshold) continue;

        uint64_t forward = 0;
        uint64_t backward = 0;
        if (query.target_position == 2) {
            forward = transition_count_device(transitions, transition_count,
                                              query.second_id, word_id, 1);
            backward = transition_count_device(transitions, transition_count,
                                               word_id, query.second_id, 2);
        } else if (query.target_position == 0) {
            forward = transition_count_device(transitions, transition_count,
                                              word_id, query.first_id, 0);
            backward = transition_count_device(transitions, transition_count,
                                               query.first_id, word_id, 3);
        } else {
            forward = transition_count_device(transitions, transition_count,
                                              query.second_id, word_id, 0);
            backward = transition_count_device(transitions, transition_count,
                                               word_id, query.second_id, 3);
        }

        if (forward + backward < transition_threshold) continue;
        emitted++;
        if (word_id == query.target_id) recovered++;
    }

    if (emitted > 0) atomicAdd(&results[config].candidates_emitted, emitted);
    if (recovered > 0) {
        atomicAdd(&results[config].gold_recovered, recovered);

        const RelationalWordStats target = word_stats[query.target_id];
        double target_total = (double)(target.left_count + target.center_count + target.right_count);
        uint64_t target_position_count = query.target_position == 0 ? target.left_count :
                                         query.target_position == 1 ? target.center_count :
                                                                      target.right_count;
        double target_position_ratio = target_total > 0.0 ?
            (double)target_position_count / target_total : 0.0;
        uint64_t target_forward = 0;
        uint64_t target_backward = 0;
        if (query.target_position == 2) {
            target_forward = transition_count_device(transitions, transition_count,
                                                     query.second_id, query.target_id, 1);
            target_backward = transition_count_device(transitions, transition_count,
                                                      query.target_id, query.second_id, 2);
        } else if (query.target_position == 0) {
            target_forward = transition_count_device(transitions, transition_count,
                                                     query.target_id, query.first_id, 0);
            target_backward = transition_count_device(transitions, transition_count,
                                                      query.first_id, query.target_id, 3);
        } else {
            target_forward = transition_count_device(transitions, transition_count,
                                                     query.second_id, query.target_id, 0);
            target_backward = transition_count_device(transitions, transition_count,
                                                      query.target_id, query.second_id, 3);
        }
        if (target_position_ratio >= 0.15 && target_forward + target_backward > 0)
            atomicAdd(&results[config].fully_supported_recovered, 1ULL);
    }
}

int run_threshold_sweep_cuda(const ThresholdSweepQuery *queries,
                             size_t query_count,
                             size_t vocab_size,
                             const RelationalWordStats *word_stats,
                             const RelationalTransition *transitions,
                             size_t transition_count,
                             const double *position_thresholds,
                             size_t position_threshold_count,
                             const uint64_t *transition_thresholds,
                             size_t transition_threshold_count,
                             ThresholdSweepResult *results) {
    if (!queries || query_count == 0 || !word_stats || !results ||
        vocab_size == 0 || transition_count > (size_t)INT_MAX ||
        query_count > (size_t)INT_MAX || vocab_size > (size_t)INT_MAX ||
        position_threshold_count == 0 || transition_threshold_count == 0)
        return -1;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1)
        return -1;

    RelationalTransition *sorted_transitions = (RelationalTransition *)malloc(
        transition_count * sizeof(RelationalTransition));
    if (!sorted_transitions) return -1;
    memcpy(sorted_transitions, transitions, transition_count * sizeof(RelationalTransition));
    qsort(sorted_transitions, transition_count, sizeof(RelationalTransition), transition_compare);

    ThresholdSweepQuery *d_queries = NULL;
    RelationalWordStats *d_stats = NULL;
    RelationalTransition *d_transitions = NULL;
    double *d_position_thresholds = NULL;
    uint64_t *d_transition_thresholds = NULL;
    DeviceThresholdResult *d_results = NULL;
    size_t config_count = position_threshold_count * transition_threshold_count;
    DeviceThresholdResult *host_results = (DeviceThresholdResult *)calloc(
        config_count, sizeof(DeviceThresholdResult));
    int status = -1;

    if (cudaMalloc((void **)&d_queries, query_count * sizeof(ThresholdSweepQuery)) != cudaSuccess ||
        cudaMalloc((void **)&d_stats, (vocab_size + 1) * sizeof(RelationalWordStats)) != cudaSuccess ||
        cudaMalloc((void **)&d_transitions, transition_count * sizeof(RelationalTransition)) != cudaSuccess ||
        cudaMalloc((void **)&d_position_thresholds, position_threshold_count * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void **)&d_transition_thresholds, transition_threshold_count * sizeof(uint64_t)) != cudaSuccess ||
        cudaMalloc((void **)&d_results, config_count * sizeof(DeviceThresholdResult)) != cudaSuccess ||
        !host_results)
        goto cleanup;

    if (cudaMemcpy(d_queries, queries, query_count * sizeof(ThresholdSweepQuery), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_stats, word_stats, (vocab_size + 1) * sizeof(RelationalWordStats), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_transitions, sorted_transitions, transition_count * sizeof(RelationalTransition), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_position_thresholds, position_thresholds, position_threshold_count * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_transition_thresholds, transition_thresholds, transition_threshold_count * sizeof(uint64_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(d_results, 0, config_count * sizeof(DeviceThresholdResult)) != cudaSuccess)
        goto cleanup;

    {
        int total_work = (int)(query_count * config_count);
        int blocks = (total_work + 127) / 128;
        threshold_sweep_kernel<<<blocks, 128>>>(
            d_queries, (int)query_count, d_stats, (int)vocab_size,
            d_transitions, (int)transition_count, d_position_thresholds,
            (int)position_threshold_count, d_transition_thresholds,
            (int)transition_threshold_count, d_results);
    }
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(host_results, d_results, config_count * sizeof(DeviceThresholdResult),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;

    for (size_t i = 0; i < config_count; i++) {
        results[i].gold_recovered = host_results[i].gold_recovered;
        results[i].fully_supported_recovered = host_results[i].fully_supported_recovered;
        results[i].candidates_emitted = host_results[i].candidates_emitted;
    }
    status = 0;

cleanup:
    cudaFree(d_queries);
    cudaFree(d_stats);
    cudaFree(d_transitions);
    cudaFree(d_position_thresholds);
    cudaFree(d_transition_thresholds);
    cudaFree(d_results);
    free(host_results);
    free(sorted_transitions);
    return status;
}
