#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <stddef.h>
#include "word.h"

#define TRIANGLE_ROLE_FEATURE_DIM 32

typedef struct {
    int id;
    int word_ids[3];
    int role_ids[3];
    char *words[3];
    char upos[3][16];
    char deprel[3][32];
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
TriangleChain *create_triangles_from_conllu(const char *content);
TriangleChain *create_triangles_from_conllu_with_vocab(const char *content,
                                                       Vocabulary *vocab);
int triangle_role_id(const char *upos);
void free_triangles(TriangleChain *chain);
void print_triangles(const TriangleChain *chain);
void print_vocabulary(const TriangleChain *chain);
void print_registry(const TriangleChain *chain);

#endif
