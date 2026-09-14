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
} TriangleChain;

TriangleChain *create_triangles(const char *sentence);
void free_triangles(TriangleChain *chain);
void print_triangles(const TriangleChain *chain);
void print_vocabulary(const TriangleChain *chain);
void print_registry(const TriangleChain *chain);

#endif
