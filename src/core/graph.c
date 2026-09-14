#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Graph *graph_create(const TriangleChain *triangles, const CircleChain *circles) {
    if (!triangles) return NULL;

    Graph *graph = malloc(sizeof(Graph));
    if (!graph) return NULL;

    graph->node_count = triangles->count + (circles ? circles->count : 0);
    graph->nodes = malloc(graph->node_count * sizeof(GraphNode));
    if (!graph->nodes) {
        free(graph);
        return NULL;
    }

    size_t node_idx = 0;
    for (size_t i = 0; i < triangles->count; i++) {
        graph->nodes[node_idx].id = node_idx + 1;
        graph->nodes[node_idx].type = NODE_TRIANGLE;
        for (int p = 0; p < 3; p++) {
            graph->nodes[node_idx].word_ids[p] = triangles->triangles[i].word_ids[p];
        }
        graph->nodes[node_idx].circle_word_id = 0;
        node_idx++;
    }

    if (circles) {
        for (size_t i = 0; i < circles->count; i++) {
            graph->nodes[node_idx].id = node_idx + 1;
            graph->nodes[node_idx].type = NODE_CIRCLE;
            graph->nodes[node_idx].word_ids[0] = 0;
            graph->nodes[node_idx].word_ids[1] = 0;
            graph->nodes[node_idx].word_ids[2] = 0;
            graph->nodes[node_idx].circle_word_id = circles->circles[i].word_id;
            node_idx++;
        }
    }

    graph->edge_count = 0;
    graph->edges = NULL;

    if (circles) {
        for (size_t i = 0; i < circles->count; i++) {
            int circle_node = triangles->count + i + 1;

            if (circles->circles[i].before_triangle_id > 0) {
                int tri_node = circles->circles[i].before_triangle_id;
                graph->edge_count++;
                GraphEdge *temp = realloc(graph->edges, graph->edge_count * sizeof(GraphEdge));
                if (!temp) {
                    graph_free(graph);
                    return NULL;
                }
                graph->edges = temp;
                graph->edges[graph->edge_count - 1].from_id = tri_node;
                graph->edges[graph->edge_count - 1].to_id = circle_node;
                graph->edges[graph->edge_count - 1].weight = circles->circles[i].value;
            }

            if (circles->circles[i].after_triangle_id > 0) {
                int tri_node = circles->circles[i].after_triangle_id;
                graph->edge_count++;
                GraphEdge *temp = realloc(graph->edges, graph->edge_count * sizeof(GraphEdge));
                if (!temp) {
                    graph_free(graph);
                    return NULL;
                }
                graph->edges = temp;
                graph->edges[graph->edge_count - 1].from_id = circle_node;
                graph->edges[graph->edge_count - 1].to_id = tri_node;
                graph->edges[graph->edge_count - 1].weight = circles->circles[i].value;
            }
        }
    }

    for (size_t i = 0; i < triangles->count - 1; i++) {
        graph->edge_count++;
        GraphEdge *temp = realloc(graph->edges, graph->edge_count * sizeof(GraphEdge));
        if (!temp) {
            graph_free(graph);
            return NULL;
        }
        graph->edges = temp;
        graph->edges[graph->edge_count - 1].from_id = i + 1;
        graph->edges[graph->edge_count - 1].to_id = i + 2;
        graph->edges[graph->edge_count - 1].weight = 1.0;
    }

    return graph;
}

void graph_free(Graph *graph) {
    if (!graph) return;
    free(graph->nodes);
    free(graph->edges);
    free(graph);
}

void graph_print(const Graph *graph) {
    if (!graph) return;

    printf("\n=== Graph Structure ===\n");
    printf("Nodes: %zu\n", graph->node_count);
    printf("Edges: %zu\n\n", graph->edge_count);

    int tri_count = 0;
    int circle_count = 0;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].type == NODE_TRIANGLE) tri_count++;
        else circle_count++;
    }
    printf("Triangles: %d, Circles: %d\n\n", tri_count, circle_count);

    printf("Edges (first 30):\n");
    size_t show = graph->edge_count < 30 ? graph->edge_count : 30;
    for (size_t i = 0; i < show; i++) {
        const char *from_type = graph->nodes[graph->edges[i].from_id - 1].type == NODE_TRIANGLE ? "T" : "C";
        const char *to_type = graph->nodes[graph->edges[i].to_id - 1].type == NODE_TRIANGLE ? "T" : "C";
        printf("  %s%d -> %s%d (weight: %.2f)\n",
               from_type, graph->edges[i].from_id,
               to_type, graph->edges[i].to_id,
               graph->edges[i].weight);
    }
}

void graph_save_dot(const Graph *graph, const char *filename) {
    if (!graph || !filename) return;

    FILE *f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "graph G {\n");
    fprintf(f, "  rankdir=LR;\n");
    fprintf(f, "  node [shape=box];\n\n");

    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].type == NODE_TRIANGLE) {
            fprintf(f, "  T%d [label=\"T%d\\n%d %d %d\"];\n",
                   graph->nodes[i].id,
                   graph->nodes[i].id,
                   graph->nodes[i].word_ids[0],
                   graph->nodes[i].word_ids[1],
                   graph->nodes[i].word_ids[2]);
        } else {
            fprintf(f, "  C%d [label=\"C%d\\nword:%d\", shape=circle];\n",
                   graph->nodes[i].id,
                   graph->nodes[i].id,
                   graph->nodes[i].circle_word_id);
        }
    }

    fprintf(f, "\n");
    for (size_t i = 0; i < graph->edge_count; i++) {
        fprintf(f, "  %d -- %d [label=\"%.1f\"];\n",
               graph->edges[i].from_id,
               graph->edges[i].to_id,
               graph->edges[i].weight);
    }

    fprintf(f, "}\n");
    fclose(f);
    printf("Saved graph to %s\n", filename);
}
