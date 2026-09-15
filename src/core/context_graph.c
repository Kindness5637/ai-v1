#include "context_graph.h"
#include <stdio.h>
#include <stdlib.h>

static uint64_t signature_for(const int words[3], int rotation) {
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < 3; i++) {
        hash ^= (uint64_t)(unsigned int)words[(rotation + i) % 3];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static int add_bond(ContextGraph *graph, int from, int to, double weight,
                    ContextBondType type) {
    ContextBond *grown = realloc(graph->bonds,
                                 (graph->bond_count + 1) * sizeof(ContextBond));
    if (!grown) return 0;
    graph->bonds = grown;
    graph->bonds[graph->bond_count++] = (ContextBond){from, to, weight, type};
    return 1;
}

ContextGraph *context_graph_create(const TriangleChain *chain) {
    if (!chain || !chain->vocab) return NULL;

    ContextGraph *graph = calloc(1, sizeof(ContextGraph));
    if (!graph) return NULL;

    size_t context_count = chain->count * 3;
    graph->node_count = context_count + chain->vocab->count;
    graph->nodes = calloc(graph->node_count, sizeof(ContextNode));
    if (!graph->nodes) {
        free(graph);
        return NULL;
    }

    for (size_t t = 0; t < chain->count; t++) {
        const Triangle *triangle = &chain->triangles[t];
        for (int rotation = 0; rotation < 3; rotation++) {
            size_t index = t * 3 + rotation;
            ContextNode *node = &graph->nodes[index];
            node->id = (int)index + 1;
            node->type = CONTEXT_TRIANGLE_NODE;
            node->triangle_id = triangle->id;
            node->rotation = rotation;
            for (int i = 0; i < 3; i++) {
                node->word_ids[i] = triangle->word_ids[(rotation + i) % 3];
                node->role_ids[i] = triangle->role_ids[(rotation + i) % 3];
            }
            node->signature = signature_for(triangle->word_ids, rotation);

            int word_node_base = (int)context_count;
            for (int i = 0; i < 3; i++) {
                int word_id = node->word_ids[i];
                if (word_id <= 0 || word_id > (int)chain->vocab->count) continue;
                if (!add_bond(graph, node->id, word_node_base + word_id,
                              1.0, BOND_OCCURRENCE)) {
                    context_graph_free(graph);
                    return NULL;
                }
            }

            int next_rotation = (rotation + 1) % 3;
            if (!add_bond(graph, node->id, (int)(t * 3 + next_rotation) + 1,
                          2.0, BOND_ROTATION)) {
                context_graph_free(graph);
                return NULL;
            }

            if (t + 1 < chain->count) {
                int next_id = (int)((t + 1) * 3 + rotation) + 1;
                if (!add_bond(graph, node->id, next_id, 1.5, BOND_NEIGHBOR)) {
                    context_graph_free(graph);
                    return NULL;
                }
            }
        }
    }

    for (size_t i = 0; i < chain->vocab->count; i++) {
        ContextNode *node = &graph->nodes[context_count + i];
        node->id = (int)context_count + (int)i + 1;
        node->type = CONTEXT_WORD_NODE;
        node->word_id = (int)i + 1;
    }

    return graph;
}

void context_graph_free(ContextGraph *graph) {
    if (!graph) return;
    free(graph->nodes);
    free(graph->bonds);
    free(graph);
}

static const char *bond_name(ContextBondType type) {
    switch (type) {
        case BOND_OCCURRENCE: return "occurrence";
        case BOND_ROTATION: return "rotation";
        case BOND_NEIGHBOR: return "neighbor";
        default: return "bond";
    }
}

void context_graph_print(const ContextGraph *graph) {
    if (!graph) return;
    size_t triangles = 0, words = 0;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].type == CONTEXT_TRIANGLE_NODE) triangles++;
        else words++;
    }
    printf("\n=== Context Graph ===\n");
    printf("Context nodes: %zu\nWord hubs: %zu\nBonds: %zu\n",
           triangles, words, graph->bond_count);
    printf("Sample context nodes:\n");
    size_t shown = 0;
    for (size_t i = 0; i < graph->node_count && shown < 10; i++) {
        const ContextNode *node = &graph->nodes[i];
        if (node->type != CONTEXT_TRIANGLE_NODE) continue;
        printf("  C%d triangle=%d rotation=%d [%d,%d,%d] signature=%016llx\n",
               node->id, node->triangle_id, node->rotation,
               node->word_ids[0], node->word_ids[1], node->word_ids[2],
               (unsigned long long)node->signature);
        shown++;
    }
    printf("Sample bonds:\n");
    shown = 0;
    for (size_t i = 0; i < graph->bond_count && shown < 20; i++) {
        const ContextBond *bond = &graph->bonds[i];
        printf("  %d -> %d type=%s weight=%.1f\n",
               bond->from_id, bond->to_id, bond_name(bond->type), bond->weight);
        shown++;
    }
}

void context_graph_save_dot(const ContextGraph *graph, const char *filename) {
    if (!graph || !filename) return;
    FILE *file = fopen(filename, "w");
    if (!file) return;
    fprintf(file, "digraph ContextGraph {\n  rankdir=LR;\n");
    for (size_t i = 0; i < graph->node_count; i++) {
        const ContextNode *node = &graph->nodes[i];
        if (node->type == CONTEXT_TRIANGLE_NODE) {
            fprintf(file, "  n%d [shape=box,label=\"T%d r%d\\n%d %d %d\"];\n",
                    node->id, node->triangle_id, node->rotation,
                    node->word_ids[0], node->word_ids[1], node->word_ids[2]);
        } else {
            fprintf(file, "  n%d [shape=circle,label=\"W%d\"];\n",
                    node->id, node->word_id);
        }
    }
    for (size_t i = 0; i < graph->bond_count; i++) {
        const ContextBond *bond = &graph->bonds[i];
        fprintf(file, "  n%d -> n%d [label=\"%s %.1f\"];\n",
                bond->from_id, bond->to_id, bond_name(bond->type), bond->weight);
    }
    fprintf(file, "}\n");
    fclose(file);
    printf("Saved context graph to %s\n", filename);
}

void context_graph_query(const ContextGraph *graph, const TriangleChain *chain,
                         int first_word_id, int second_word_id, size_t limit) {
    if (!graph || !chain || limit == 0) return;

    size_t matches = 0;
    printf("\n=== Context Query: [%d, %d] ===\n", first_word_id, second_word_id);
    for (size_t i = 0; i < graph->node_count && matches < limit; i++) {
        const ContextNode *node = &graph->nodes[i];
        if (node->type != CONTEXT_TRIANGLE_NODE ||
            node->word_ids[0] != first_word_id ||
            node->word_ids[1] != second_word_id) {
            continue;
        }

        int next_id = node->word_ids[2];
        const char *first = vocab_get_word(chain->vocab, node->word_ids[0]);
        const char *second = vocab_get_word(chain->vocab, node->word_ids[1]);
        const char *next = vocab_get_word(chain->vocab, next_id);
        printf("Match %zu: triangle=%d rotation=%d [%s, %s] -> %s "
               "signature=%016llx\n",
               matches + 1, node->triangle_id, node->rotation,
               first ? first : "?", second ? second : "?",
               next ? next : "?", (unsigned long long)node->signature);

        for (size_t b = 0; b < graph->bond_count; b++) {
            const ContextBond *bond = &graph->bonds[b];
            if (bond->from_id != node->id || bond->type != BOND_NEIGHBOR) continue;
            const ContextNode *neighbor = &graph->nodes[bond->to_id - 1];
            const char *neighbor_next = vocab_get_word(chain->vocab,
                                                       neighbor->word_ids[2]);
            printf("  Neighbor: triangle=%d rotation=%d -> %s weight=%.1f\n",
                   neighbor->triangle_id, neighbor->rotation,
                   neighbor_next ? neighbor_next : "?", bond->weight);
        }
        matches++;
    }

    if (matches == 0) {
        printf("No exact ordered context found.\n");
    } else {
        printf("Found %zu matching context(s).\n", matches);
    }
}

static int context_node_matches_pair(const ContextNode *node,
                                     int first_word_id, int first_role_id,
                                     int second_word_id, int second_role_id,
                                     int rotation) {
    if (!node || node->type != CONTEXT_TRIANGLE_NODE) return 0;
    if (node->word_ids[0] != first_word_id ||
        node->word_ids[1] != second_word_id) return 0;
    if (first_role_id > 0 && node->role_ids[0] != first_role_id) return 0;
    if (second_role_id > 0 && node->role_ids[1] != second_role_id) return 0;
    if (rotation >= 0 && node->rotation != rotation) return 0;
    return 1;
}

static void add_unique_candidate(int candidate_id, int *candidate_ids,
                                 size_t *count, size_t max_candidates) {
    if (candidate_id <= 0 || *count >= max_candidates) return;
    for (size_t j = 0; j < *count; j++) {
        if (candidate_ids[j] == candidate_id) return;
    }
    candidate_ids[(*count)++] = candidate_id;
}

static void add_candidate_evidence(int candidate_id, int kind,
                                   double match_score,
                                   ContextCandidate *candidates,
                                   size_t *count, size_t max_candidates) {
    if (candidate_id <= 0) return;
    size_t index = 0;
    while (index < *count && candidates[index].word_id != candidate_id) index++;
    if (index == *count) {
        if (*count >= max_candidates) return;
        candidates[index] = (ContextCandidate){candidate_id, 0, 0, 0.0};
        (*count)++;
    }
    if (kind == 0) candidates[index].occurrence_count++;
    else candidates[index].neighbor_count++;
    candidates[index].match_score += match_score;
}

static size_t collect_candidates_into(const ContextGraph *graph,
                                      int first_word_id, int first_role_id,
                                      int second_word_id, int second_role_id,
                                      int rotation,
                                      int *candidate_ids,
                                      size_t count, size_t max_candidates) {
    if (!graph || !candidate_ids || max_candidates == 0) return count;
    for (size_t i = 0; i < graph->node_count && count < max_candidates; i++) {
        const ContextNode *node = &graph->nodes[i];
        if (!context_node_matches_pair(node, first_word_id, first_role_id,
                                       second_word_id, second_role_id,
                                       rotation)) continue;

        add_unique_candidate(node->word_ids[2], candidate_ids, &count,
                             max_candidates);
        /* Neighbor bonds preserve the same rotation three context nodes later. */
        if (node->id + 3 <= (int)graph->node_count &&
            graph->nodes[node->id + 2].type == CONTEXT_TRIANGLE_NODE) {
            const ContextNode *neighbor = &graph->nodes[node->id + 2];
            add_unique_candidate(neighbor->word_ids[2], candidate_ids, &count,
                                 max_candidates);
        }
    }
    return count;
}

static size_t collect_evidence_into(const ContextGraph *graph,
                                    int first_word_id, int first_role_id,
                                    int second_word_id, int second_role_id,
                                    int rotation,
                                    double match_score,
                                    ContextCandidate *candidates,
                                    size_t count, size_t max_candidates) {
    if (!graph || !candidates || max_candidates == 0) return count;
    for (size_t i = 0; i < graph->node_count; i++) {
        const ContextNode *node = &graph->nodes[i];
        if (!context_node_matches_pair(node, first_word_id, first_role_id,
                                       second_word_id, second_role_id,
                                       rotation)) continue;

        int next_ids[2] = {node->word_ids[2], 0};
        if (node->id + 3 <= (int)graph->node_count &&
            graph->nodes[node->id + 2].type == CONTEXT_TRIANGLE_NODE) {
            next_ids[1] = graph->nodes[node->id + 2].word_ids[2];
        }
        for (int kind = 0; kind < 2; kind++) {
            add_candidate_evidence(next_ids[kind], kind, match_score,
                                   candidates, &count, max_candidates);
        }
    }
    return count;
}

size_t context_graph_collect_candidates(const ContextGraph *graph,
                                        int first_word_id, int second_word_id,
                                        int *candidate_ids, size_t max_candidates) {
    return context_graph_collect_candidates_scoped(
        graph, first_word_id, 0, second_word_id, 0, -1,
        candidate_ids, max_candidates);
}

size_t context_graph_collect_candidates_scoped(const ContextGraph *graph,
                                               int first_word_id, int first_role_id,
                                               int second_word_id, int second_role_id,
                                               int rotation,
                                               int *candidate_ids,
                                               size_t max_candidates) {
    return collect_candidates_into(graph, first_word_id, first_role_id,
                                   second_word_id, second_role_id, rotation,
                                   candidate_ids, 0, max_candidates);
}

size_t context_graph_collect_candidates_fallback(const ContextGraph *graph,
                                                 int first_word_id,
                                                 int first_role_id,
                                                 int second_word_id,
                                                 int second_role_id,
                                                 int rotation,
                                                 int *candidate_ids,
                                                 size_t max_candidates) {
    size_t count = 0;
    count = collect_candidates_into(graph, first_word_id, first_role_id,
                                    second_word_id, second_role_id, rotation,
                                    candidate_ids, count, max_candidates);
    count = collect_candidates_into(graph, first_word_id, first_role_id,
                                    second_word_id, second_role_id, -1,
                                    candidate_ids, count, max_candidates);
    count = collect_candidates_into(graph, first_word_id, 0,
                                    second_word_id, 0, rotation,
                                    candidate_ids, count, max_candidates);
    return collect_candidates_into(graph, first_word_id, 0,
                                   second_word_id, 0, -1,
                                   candidate_ids, count, max_candidates);
}

size_t context_graph_collect_candidate_evidence(const ContextGraph *graph,
                                                int first_word_id, int second_word_id,
                                                ContextCandidate *candidates,
                                                size_t max_candidates) {
    return context_graph_collect_candidate_evidence_scoped(
        graph, first_word_id, 0, second_word_id, 0, -1,
        candidates, max_candidates);
}

size_t context_graph_collect_candidate_evidence_scoped(const ContextGraph *graph,
                                                       int first_word_id,
                                                       int first_role_id,
                                                       int second_word_id,
                                                       int second_role_id,
                                                       int rotation,
                                                       ContextCandidate *candidates,
                                                       size_t max_candidates) {
    size_t count = 0;
    return collect_evidence_into(graph, first_word_id, first_role_id,
                                 second_word_id, second_role_id, rotation,
                                 4.0, candidates, count, max_candidates);
}

size_t context_graph_collect_candidate_evidence_fallback(const ContextGraph *graph,
                                                         int first_word_id,
                                                         int first_role_id,
                                                         int second_word_id,
                                                         int second_role_id,
                                                         int rotation,
                                                         ContextCandidate *candidates,
                                                         size_t max_candidates) {
    size_t count = 0;
    count = collect_evidence_into(graph, first_word_id, first_role_id,
                                  second_word_id, second_role_id, rotation,
                                  4.0, candidates, count, max_candidates);
    count = collect_evidence_into(graph, first_word_id, first_role_id,
                                  second_word_id, second_role_id, -1,
                                  2.0, candidates, count, max_candidates);
    count = collect_evidence_into(graph, first_word_id, 0,
                                  second_word_id, 0, rotation,
                                  1.5, candidates, count, max_candidates);
    return collect_evidence_into(graph, first_word_id, 0,
                                 second_word_id, 0, -1,
                                 1.0, candidates, count, max_candidates);
}

size_t context_graph_collect_candidate_evidence_relational(const ContextGraph *graph,
                                                           const RelationalRegistry *rel_reg,
                                                           int first_word_id,
                                                           int second_word_id,
                                                           int rotation,
                                                           ContextCandidate *candidates,
                                                           size_t max_candidates) {
    /* Collect candidates using purely positional & directional evidence without UPOS/DEPREL */
    size_t count = 0;
    count = collect_evidence_into(graph, first_word_id, 0, second_word_id, 0, rotation,
                                  2.0, candidates, count, max_candidates);
    count = collect_evidence_into(graph, first_word_id, 0, second_word_id, 0, -1,
                                  1.0, candidates, count, max_candidates);

    if (!rel_reg) return count;

    /* Re-weight candidates using positional asymmetry and transition statistics */
    for (size_t i = 0; i < count; i++) {
        int cand_id = candidates[i].word_id;
        if (cand_id <= 0 || (size_t)cand_id > rel_reg->vocab_size) continue;

        const RelationalWordStats *stats = &rel_reg->word_stats[cand_id];
        uint64_t support = stats->left_count + stats->right_count + stats->center_count;

        /* Positional compatibility score based on rotation target position */
        double pos_compat = 0.0;
        int target_position = (rotation >= 0) ? (rotation + 2) % 3 : 2;
        if (target_position == 0) {
            pos_compat = (stats->total_count > 0) ? (double)stats->left_count / stats->total_count : 0.0;
        } else if (target_position == 1) {
            pos_compat = (stats->total_count > 0) ? (double)stats->center_count / stats->total_count : 0.0;
        } else {
            pos_compat = (stats->total_count > 0) ? (double)stats->right_count / stats->total_count : 0.0;
        }

        /* Transition compatibility with preceding word */
        uint64_t trans_count = relational_registry_get_transition_count(
            rel_reg, second_word_id, cand_id, (target_position == 2) ? 1 : 0);

        double transition_compat = (trans_count > 0) ? log1p((double)trans_count) : 0.0;
        double support_weight = log1p((double)support);

        candidates[i].match_score += pos_compat * 2.0 + transition_compat * 1.5 + support_weight * 0.5;
    }

    return count;
}

