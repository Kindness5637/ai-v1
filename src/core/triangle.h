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

/* Diagnostic: how much does the role (UPOS) channel tell us beyond the word
   identity? Pure corpus statistic over the chain; no model involved.

   H(role|word) is the token-weighted conditional entropy in bits. If it is
   near zero, a token's role is derivable from its word alone, so any role
   feature -- a one-hot bias shift, or a role-by-word interaction term -- is
   largely redundant with the word embedding already present in the input.

   How each number is computed (all base-2, probabilities from token counts):
     tokens            = tokens carrying a NONZERO role (0 = holding/unknown)
     marginal_entropy  = H(role) = -sum_r p(r) log2 p(r) over all such tokens
     cond_entropy      = sum_w (n_w / tokens) * H(role | w)   [token-weighted]
     cond_entropy_word = (1/distinct_words) * sum_w H(role | w) [word-weighted]
     mutual_info       = marginal_entropy - cond_entropy

   What this does NOT account for:
     - Context: all occurrences of an ambiguous word are pooled, so a word that
       is disambiguated by its neighbours still counts as fully ambiguous here.
     - Coverage: holding tokens (role 0) are excluded, which biases H(role)
       upward when many tokens are unknown.
     - Usefulness: high entropy means roles are not redundant with the word;
       it does NOT establish that a role feature would improve ranking. */
typedef struct {
    size_t tokens;
    size_t distinct_words;
    size_t deterministic_words;   /* words observed with exactly one role */
    size_t ambiguous_tokens;      /* tokens whose word took more than one role */
    double marginal_entropy_bits;
    double cond_entropy_bits;
    double cond_entropy_per_word_bits;
    double mutual_info_bits;
    int top_ambiguous_word_id;
    char top_ambiguous_word[64];
    double top_ambiguous_entropy_bits;
} RoleEntropyStats;

int triangle_role_entropy(const TriangleChain *chain, RoleEntropyStats *out);
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

