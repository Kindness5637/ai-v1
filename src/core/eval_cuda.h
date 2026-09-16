#ifndef EVAL_CUDA_H
#define EVAL_CUDA_H

#include <stddef.h>
#include <stdint.h>
#include "triangle.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int first_id;
    int second_id;
    int target_id;
    int target_position;
} ThresholdSweepQuery;

typedef struct {
    uint64_t gold_recovered;
    uint64_t fully_supported_recovered;
    uint64_t candidates_emitted;
} ThresholdSweepResult;

int eval_cuda_device_count(void);

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
                             ThresholdSweepResult *results);

#ifdef __cplusplus
}
#endif

#endif
