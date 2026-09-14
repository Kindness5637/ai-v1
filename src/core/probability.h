#ifndef PROBABILITY_H
#define PROBABILITY_H

#include "triangle.h"

typedef struct {
    int word_id;
    int neighbor_id;
    int position;
    int count;
} CoOccurrence;

typedef struct {
    CoOccurrence *entries;
    size_t count;
    size_t capacity;
} CoOccurrenceMatrix;

typedef struct {
    int word_id;
    int *neighbor_ids;
    int *positions;
    double *probabilities;
    size_t count;
} WordProbability;

typedef struct {
    int word_id;
    int *before_ids;
    int *after_ids;
    double *before_probs;
    double *after_probs;
    size_t before_count;
    size_t after_count;
} SequenceProbability;

typedef struct {
    WordProbability *words;
    size_t count;
    SequenceProbability *sequences;
    size_t seq_count;
} ProbabilityMatrix;

ProbabilityMatrix *prob_create(const TriangleChain *chain);
void prob_free(ProbabilityMatrix *prob);
void prob_print(const ProbabilityMatrix *prob);
void prob_print_word(const ProbabilityMatrix *prob, int word_id);
void prob_print_sequences(const ProbabilityMatrix *prob);

#endif
