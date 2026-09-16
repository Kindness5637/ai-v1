#include "eval_cuda.h"

int eval_cuda_device_count(void) {
    return 0;
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
    (void)queries;
    (void)query_count;
    (void)vocab_size;
    (void)word_stats;
    (void)transitions;
    (void)transition_count;
    (void)position_thresholds;
    (void)position_threshold_count;
    (void)transition_thresholds;
    (void)transition_threshold_count;
    (void)results;
    return -1;
}
