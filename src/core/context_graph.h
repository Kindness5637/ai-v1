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

ContextGraph *context_graph_create(const TriangleChain *chain);
void context_graph_free(ContextGraph *graph);
void context_graph_print(const ContextGraph *graph);
void context_graph_save_dot(const ContextGraph *graph, const char *filename);
void context_graph_query(const ContextGraph *graph, const TriangleChain *chain,
                         int first_word_id, int second_word_id, size_t limit);

#endif
