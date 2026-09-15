#ifndef CONTEXT_GRAPH_H
#define CONTEXT_GRAPH_H

#include <stdint.h>
#include "triangle.h"

typedef enum {
    CONTEXT_TRIANGLE_NODE,
    CONTEXT_WORD_NODE
} ContextNodeType;

typedef enum {
    BOND_OCCURRENCE,
    BOND_ROTATION,
    BOND_NEIGHBOR
} ContextBondType;

typedef struct {
    int id;
    ContextNodeType type;
    int triangle_id;
    int rotation;
    int word_ids[3];
    int role_ids[3];
    int word_id;
    uint64_t signature;
} ContextNode;

typedef struct {
    int from_id;
    int to_id;
    double weight;
    ContextBondType type;
} ContextBond;

typedef struct {
    ContextNode *nodes;
    size_t node_count;
    ContextBond *bonds;
    size_t bond_count;
} ContextGraph;

typedef struct {
    int word_id;
    int occurrence_count;
    int neighbor_count;
    double match_score;
} ContextCandidate;

ContextGraph *context_graph_create(const TriangleChain *chain);
void context_graph_free(ContextGraph *graph);
void context_graph_print(const ContextGraph *graph);
void context_graph_save_dot(const ContextGraph *graph, const char *filename);
void context_graph_query(const ContextGraph *graph, const TriangleChain *chain,
                         int first_word_id, int second_word_id, size_t limit);
size_t context_graph_collect_candidates(const ContextGraph *graph,
                                        int first_word_id, int second_word_id,
                                        int *candidate_ids, size_t max_candidates);
size_t context_graph_collect_candidates_scoped(const ContextGraph *graph,
                                               int first_word_id, int first_role_id,
                                               int second_word_id, int second_role_id,
                                               int rotation,
                                               int *candidate_ids,
                                               size_t max_candidates);
size_t context_graph_collect_candidates_fallback(const ContextGraph *graph,
                                                 int first_word_id, int first_role_id,
                                                 int second_word_id, int second_role_id,
                                                 int rotation,
                                                 int *candidate_ids,
                                                 size_t max_candidates);
size_t context_graph_collect_candidate_evidence(const ContextGraph *graph,
                                                int first_word_id, int second_word_id,
                                                ContextCandidate *candidates,
                                                size_t max_candidates);
size_t context_graph_collect_candidate_evidence_scoped(const ContextGraph *graph,
                                                       int first_word_id, int first_role_id,
                                                       int second_word_id, int second_role_id,
                                                       int rotation,
                                                       ContextCandidate *candidates,
                                                       size_t max_candidates);
size_t context_graph_collect_candidate_evidence_fallback(const ContextGraph *graph,
                                                         int first_word_id, int first_role_id,
                                                         int second_word_id, int second_role_id,
                                                         int rotation,
                                                         ContextCandidate *candidates,
                                                         size_t max_candidates);
size_t context_graph_collect_candidate_evidence_relational(const ContextGraph *graph,
                                                           const RelationalRegistry *rel_reg,
                                                           int first_word_id,
                                                           int second_word_id,
                                                           int rotation,
                                                           ContextCandidate *candidates,
                                                           size_t max_candidates);

#endif

