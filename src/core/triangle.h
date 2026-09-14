#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <stddef.h>
#include "word.h"

typedef struct {
    int id;
    int word_ids[3];
    char *words[3];
} Triangle;

typedef struct {
    Triangle *triangles;
    size_t count;
    Vocabulary *vocab;
    WordRegistry *registry;
    int owns_vocab;
} TriangleChain;

TriangleChain *create_triangles(const char *sentence);
TriangleChain *create_triangles_with_vocab(const char *sentence, Vocabulary *vocab);
void free_triangles(TriangleChain *chain);
void print_triangles(const TriangleChain *chain);
void print_vocabulary(const TriangleChain *chain);
void print_registry(const TriangleChain *chain);

#endif
