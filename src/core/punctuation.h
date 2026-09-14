#ifndef PUNCTUATION_H
#define PUNCTUATION_H

#include "triangle.h"

typedef struct {
    int id;
    char symbol;
    int position;
    int triangle_id;
    int word_id;
} PunctuationMark;

typedef struct {
    PunctuationMark *marks;
    size_t count;
    size_t capacity;
} PunctuationChain;

PunctuationChain *punctuation_create(const TriangleChain *chain, const char *text);
void punctuation_free(PunctuationChain *chain);
void punctuation_print(const PunctuationChain *chain);
int punctuation_count_symbol(const PunctuationChain *chain, char symbol);

#endif
