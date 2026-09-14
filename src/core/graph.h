#ifndef GRAPH_H
#define GRAPH_H

#include "triangle.h"
#include "circle.h"

typedef enum {
    NODE_TRIANGLE,
    NODE_CIRCLE
} NodeType;

typedef struct {
    int id;
    NodeType type;
    int word_ids[3];
    int circle_word_id;
} GraphNode;

typedef struct {
    int from_id;
    int to_id;
    double weight;
} GraphEdge;

typedef struct {
    GraphNode *nodes;
    size_t node_count;
    GraphEdge *edges;
    size_t edge_count;
} Graph;

Graph *graph_create(const TriangleChain *triangles, const CircleChain *circles);
void graph_free(Graph *graph);
void graph_print(const Graph *graph);
void graph_save_dot(const Graph *graph, const char *filename);

#endif
