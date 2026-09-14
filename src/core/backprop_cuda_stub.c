#include "backprop.h"
#include <stdio.h>

int backprop_train_cuda(BackpropTrainer *trainer, const TriangleChain *chain,
                        int requested_gpus) {
    (void)trainer;
    (void)chain;
    (void)requested_gpus;
    fprintf(stderr, "CUDA support is not included in this build. Use 'make cuda'.\n");
    return -1;
}
