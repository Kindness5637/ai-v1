#ifndef MATCHER_H
#define MATCHER_H

#include "triangle.h"
#include "probability.h"

typedef struct {
    int triangle_id;
    double score;
    int matched_words;
    int total_words;
    int *matched_positions;
} MatchResult;

typedef struct {
    MatchResult *results;
    size_t count;
    double accuracy;
    double deviation;
    char *reconstructed;
} MatcherOutput;

MatcherOutput *matcher_find(const TriangleChain *chain, const ProbabilityMatrix *prob, const char *input);
void matcher_free(MatcherOutput *output);
void matcher_print(const MatcherOutput *output);

#endif
