#ifndef RECONSTRUCT_H
#define RECONSTRUCT_H

#include "triangle.h"

typedef struct {
    char **words;
    size_t count;
    int found;
    int total;
} ReconstructedText;

ReconstructedText *reconstruct(const TriangleChain *chain);
void reconstruct_free(ReconstructedText *text);
void reconstruct_print(const ReconstructedText *text);
void reconstruct_compare(const ReconstructedText *text, const char *original);

#endif
