#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <stddef.h>
#include <stdint.h>
#include "word.h"


#define TRIANGLE_ROLE_FEATURE_DIM 32

#define DISCOVERED_ROLE_LEFT   101
#define DISCOVERED_ROLE_CENTER 102
#define DISCOVERED_ROLE_RIGHT  103

typedef struct {
    uint64_t left_count;
    uint64_t right_count;
    uint64_t center_count;
    uint64_t total_count;
    double asymmetry;
} RelationalWordStats;

typedef struct {
    int from_word_id;
    int to_word_id;
    int transition_type; /* 0: L->C, 1: C->R, 2: R->C, 3: C->L */
    uint64_t count;
} RelationalTransition;

typedef struct {
    RelationalWordStats *word_stats;
    size_t vocab_size;
    RelationalTransition *transitions;
    size_t transition_count;
    size_t transition_capacity;
} RelationalRegistry;

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
    RelationalRegistry *relational_reg;
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

RelationalRegistry *relational_registry_create(size_t initial_vocab_size);
void relational_registry_free(RelationalRegistry *reg);
void relational_registry_ensure_vocab(RelationalRegistry *reg, size_t vocab_size);
void relational_registry_ingest_chain(RelationalRegistry *reg, const TriangleChain *chain);
uint64_t relational_registry_get_transition_count(const RelationalRegistry *reg,
                                                  int from_id, int to_id,
                                                  int transition_type);
void relational_registry_report(const RelationalRegistry *reg, const Vocabulary *vocab, size_t top_n);

#endif

