#ifndef LEARN_H
#define LEARN_H

#include "triangle.h"

typedef struct {
    int word_id;
    int old_context_count;
    int new_context_count;
    double deviation;
} WordLearning;

typedef struct {
    WordLearning *words;
    size_t count;
    TriangleChain *new_triangles;
    double total_deviation;
    int new_words_found;
    int context_changes;
} LearningOutput;

LearningOutput *learn_compare(const TriangleChain *old_chain, const TriangleChain *new_chain);
void learning_free(LearningOutput *output);
void learning_print(const LearningOutput *output);
TriangleChain *learn_merge(const TriangleChain *old_chain, const TriangleChain *new_chain);

#endif
