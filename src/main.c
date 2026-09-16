#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core/triangle.h"
#include "core/probability.h"
#include "core/neural.h"
#include "core/formula.h"
#include "core/matrix.h"
#include "core/reconstruct.h"
#include "core/circle.h"
#include "core/graph.h"
#include "core/context_graph.h"
#include "core/learn.h"
#include "core/punctuation.h"
#include "core/backprop.h"
#include "core/eval_cuda.h"

char *read_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) return NULL;

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    char *content = malloc(size + 1);
    if (!content) {
        fclose(file);
        return NULL;
    }

    fread(content, 1, size, file);
    content[size] = '\0';
    fclose(file);

    return content;
}

void print_usage(void) {
    printf("Usage:\n");
    printf("  ./triangle.out <file>              - Process file (first pass)\n");
    printf("  ./triangle.out <file> -train       - Train neural network\n");
    printf("  ./triangle.out <file> -eval        - Evaluate predictions\n");
    printf("  ./triangle.out <file> -formula <p> - Calculate formula with power p\n");
    printf("  ./triangle.out <file> -matrix      - Create matrix table\n");
    printf("  ./triangle.out <file> -reconstruct - Reconstruct text from IDs\n");
    printf("  ./triangle.out <file> -circle <m>  - Analyze conjunctions with multiplier m\n");
    printf("  ./triangle.out <file> -graph <m>   - Build graph with circle multiplier m\n");
    printf("  ./triangle.out <file> -context-graph - Build rotated context graph\n");
    printf("  ./triangle.out <file> -context-query <a> <b> - Query ordered context\n");
    printf("  ./triangle.out <file> -predict <a> <b> - Rank graph candidates neurally\n");
    printf("  ./triangle.out <file> -evaluate-context [n] - Evaluate rotations\n");
    printf("  ./triangle.out <train> -heldout <test> [train2 ...] - Multi-document held-out experiment\n");
    printf("  ./triangle.out <old> -learn <new>  - Learn from new text using old as base\n");
    printf("  ./triangle.out <file> -backprop    - Train with backpropagation\n");
    printf("  .conllu files use FORM, UPOS, and DEPREL role metadata\n");
}

static int nearest_word_id(const BackpropNetwork *network, const double *vector) {
    int best = 0;
    double best_distance = INFINITY;

    for (int word = 0; word < network->vocab_size; word++) {
        double distance = 0.0;
        for (int d = 0; d < network->embed_dim; d++) {
            double delta = vector[d] - network->embeddings[word * network->embed_dim + d];
            distance += delta * delta;
        }
        if (distance < best_distance) {
            best_distance = distance;
            best = word;
        }
    }

    return best + 1;
}

static void fill_model_input(double *input, int slot, int word_id, int role_id,
                             const BackpropNetwork *network) {
    int feature_dim = network->embed_dim + TRIANGLE_ROLE_FEATURE_DIM;
    int word_index = word_id - 1;
    if (word_index < 0) word_index = 0;
    if (word_index >= network->vocab_size) word_index = network->vocab_size - 1;
    for (int d = 0; d < network->embed_dim; d++) {
        input[slot * feature_dim + d] =
            network->embeddings[word_index * network->embed_dim + d];
    }
    for (int r = 0; r < TRIANGLE_ROLE_FEATURE_DIM; r++) {
        input[slot * feature_dim + network->embed_dim + r] =
            role_id == r + 1 ? 1.0 : 0.0;
    }
}

static int has_conllu_suffix(const char *filename) {
    size_t length = strlen(filename);
    return length >= 7 && strcmp(filename + length - 7, ".conllu") == 0;
}

static uint64_t threshold_transition_count(const RelationalRegistry *rel_reg,
                                           const ThresholdSweepQuery *query,
                                           int candidate_id) {
    int target_position = query->target_position;
    uint64_t forward = 0;
    uint64_t backward = 0;
    if (target_position == 2) {
        forward = relational_registry_get_transition_count(
            rel_reg, query->second_id, candidate_id, 1);
        backward = relational_registry_get_transition_count(
            rel_reg, candidate_id, query->second_id, 2);
    } else if (target_position == 0) {
        forward = relational_registry_get_transition_count(
            rel_reg, candidate_id, query->first_id, 0);
        backward = relational_registry_get_transition_count(
            rel_reg, query->first_id, candidate_id, 3);
    } else {
        forward = relational_registry_get_transition_count(
            rel_reg, query->second_id, candidate_id, 0);
        backward = relational_registry_get_transition_count(
            rel_reg, candidate_id, query->second_id, 3);
    }
    return forward + backward;
}

static void run_threshold_sweep_cpu(const ThresholdSweepQuery *queries,
                                    size_t query_count,
                                    const RelationalRegistry *rel_reg,
                                    const double *position_thresholds,
                                    size_t position_threshold_count,
                                    const uint64_t *transition_thresholds,
                                    size_t transition_threshold_count,
                                    ThresholdSweepResult *results) {
    size_t config_count = position_threshold_count * transition_threshold_count;
    memset(results, 0, config_count * sizeof(*results));
    for (size_t q = 0; q < query_count; q++) {
        const ThresholdSweepQuery *query = &queries[q];
        for (size_t p = 0; p < position_threshold_count; p++) {
            for (size_t tr = 0; tr < transition_threshold_count; tr++) {
                size_t index = p * transition_threshold_count + tr;
                for (size_t word = 1; word <= rel_reg->vocab_size; word++) {
                    const RelationalWordStats *stats = &rel_reg->word_stats[word];
                    double total = (double)(stats->left_count + stats->center_count + stats->right_count);
                    if (total <= 0.0) continue;
                    uint64_t position_count = query->target_position == 0 ? stats->left_count :
                                               query->target_position == 1 ? stats->center_count :
                                                                            stats->right_count;
                    if ((double)position_count / total < position_thresholds[p]) continue;
                    if (threshold_transition_count(rel_reg, query, (int)word) < transition_thresholds[tr]) continue;
                    results[index].candidates_emitted++;
                    if ((int)word == query->target_id) {
                        results[index].gold_recovered++;
                        const RelationalWordStats *target = &rel_reg->word_stats[query->target_id];
                        double target_total = (double)(target->left_count + target->center_count + target->right_count);
                        uint64_t target_position_count = query->target_position == 0 ? target->left_count :
                                                         query->target_position == 1 ? target->center_count :
                                                                                      target->right_count;
                        if (target_total > 0.0 &&
                            (double)target_position_count / target_total >= 0.15 &&
                            threshold_transition_count(rel_reg, query, query->target_id) > 0)
                            results[index].fully_supported_recovered++;
                    }
                }
            }
        }
    }
}

static int run_gpu_evaluation_only(const TriangleChain *chain,
                                   const ContextGraph *graph,
                                   const RelationalRegistry *rel_reg,
                                   size_t limit) {
    if (!chain || !graph || !rel_reg) return -1;
    if (limit > chain->count) limit = chain->count;

    size_t capacity = limit * 3;
    ThresholdSweepQuery *queries = calloc(capacity ? capacity : 1,
                                          sizeof(*queries));
    if (!queries) return -1;
    size_t query_count = 0;

    for (size_t t = 0; t < limit; t++) {
        for (int rotation = 0; rotation < 3; rotation++) {
            int first_id = chain->triangles[t].word_ids[rotation];
            int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
            int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];
            if (target_id <= 0) continue;

            ContextCandidate evidence[256];
            size_t count = context_graph_collect_candidate_evidence_relational(
                graph, rel_reg, first_id, second_id, rotation, evidence, 256);
            if (count == 0) continue;

            int target_present = 0;
            for (size_t c = 0; c < count; c++) {
                if (evidence[c].word_id == target_id) {
                    target_present = 1;
                    break;
                }
            }
            if (target_present) continue;

            int exact_graph_match = 0;
            for (size_t i = 0; i < graph->node_count; i++) {
                const ContextNode *node = &graph->nodes[i];
                if (node->type == CONTEXT_TRIANGLE_NODE &&
                    node->word_ids[0] == first_id &&
                    node->word_ids[1] == second_id &&
                    node->rotation == rotation &&
                    node->word_ids[2] == target_id) {
                    exact_graph_match = 1;
                    break;
                }
            }
            if (!exact_graph_match && query_count < capacity) {
                queries[query_count++] = (ThresholdSweepQuery){
                    first_id, second_id, target_id, (rotation + 2) % 3};
            }
        }
    }

    double position_thresholds[3] = {0.15, 0.25, 0.35};
    uint64_t transition_thresholds[4] = {1, 2, 3, 6};
    ThresholdSweepResult results[12];
    int status = run_threshold_sweep_cuda(
        queries, query_count, rel_reg->vocab_size, rel_reg->word_stats,
        rel_reg->transitions, rel_reg->transition_count,
        position_thresholds, 3, transition_thresholds, 4, results);
    if (status != 0) {
        free(queries);
        return -1;
    }

    printf("\n=== GPU Evaluation Only: Threshold Sweep (%zu queries) ===\n",
           query_count);
    printf("%-10s %-12s %-20s %-20s %-15s\n",
           "Pos Thresh", "Trans Cutoff", "Gold Recovered",
           "Fully-Supp", "Avg Candidates");
    for (int p = 0; p < 3; p++) {
        for (int tr = 0; tr < 4; tr++) {
            int index = p * 4 + tr;
            printf("%-10.2f >= %-9llu %llu / %zu             %llu             %.2f\n",
                   position_thresholds[p],
                   (unsigned long long)transition_thresholds[tr],
                   (unsigned long long)results[index].gold_recovered,
                   query_count,
                   (unsigned long long)results[index].fully_supported_recovered,
                   query_count > 0 ? (double)results[index].candidates_emitted / query_count : 0.0);
        }
    }
    fflush(stdout);
    free(queries);
    return 0;
}

static void run_neural_evaluation_only(const BackpropTrainer *trainer,
                                       const TriangleChain *chain,
                                       size_t limit) {
    if (!trainer || !trainer->network || !chain) return;
    if (limit > chain->count) limit = chain->count;

    BackpropTrainer *mutable_trainer = (BackpropTrainer *)trainer;
    double *input = calloc(trainer->network->input_size, sizeof(double));
    double *output = calloc(trainer->network->output_size, sizeof(double));
    if (!input || !output) {
        free(input);
        free(output);
        return;
    }

    int total = 0, top1 = 0, top5 = 0;
    int vocab_size = trainer->network->vocab_size;
    int embed_dim = trainer->network->embed_dim;
    for (size_t t = 0; t < limit; t++) {
        for (int rotation = 0; rotation < 3; rotation++) {
            int first_id = chain->triangles[t].word_ids[rotation];
            int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
            int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];
            if (target_id <= 0 || target_id > vocab_size) continue;

            memset(input, 0, trainer->network->input_size * sizeof(double));
            fill_model_input(input, 0, first_id, 0, trainer->network);
            fill_model_input(input, 1, second_id, 0, trainer->network);
            backprop_predict(mutable_trainer, input, output);

            int best_ids[5] = {-1, -1, -1, -1, -1};
            double best_distances[5] = {INFINITY, INFINITY, INFINITY, INFINITY, INFINITY};
            int target_start = ((rotation + 2) % 3) * embed_dim;
            for (int word = 1; word <= vocab_size; word++) {
                double distance = 0.0;
                for (int d = 0; d < embed_dim; d++) {
                    double delta = output[target_start + d] -
                        trainer->network->embeddings[(word - 1) * embed_dim + d];
                    distance += delta * delta;
                }
                for (int rank = 0; rank < 5; rank++) {
                    if (distance >= best_distances[rank]) continue;
                    for (int shift = 4; shift > rank; shift--) {
                        best_distances[shift] = best_distances[shift - 1];
                        best_ids[shift] = best_ids[shift - 1];
                    }
                    best_distances[rank] = distance;
                    best_ids[rank] = word;
                    break;
                }
            }
            total++;
            if (best_ids[0] == target_id) top1++;
            for (int rank = 0; rank < 5; rank++) {
                if (best_ids[rank] == target_id) {
                    top5++;
                    break;
                }
            }
        }
    }
    printf("\n=== Neural Held-out Evaluation (%d queries) ===\n", total);
    printf("Neural top-1: %.1f%%\n", total > 0 ? 100.0 * top1 / total : 0.0);
    printf("Neural top-5: %.1f%%\n", total > 0 ? 100.0 * top5 / total : 0.0);
    fflush(stdout);
    free(input);
    free(output);
}

static void evaluate_context_model(BackpropTrainer *trainer,
                                   const TriangleChain *chain,
                                   const ContextGraph *graph,
                                   const RelationalRegistry *rel_reg,
                                   int use_mode_b,
                                   size_t limit) {
    if (!graph) {
        printf("Unable to create evaluation context graph.\n");
        return;
    }
    if (limit > chain->count) limit = chain->count;
    int graph_hits = 0, neural_top1 = 0, neural_top5 = 0, total = 0;
    int pair_queries = 0, ambiguous_pairs = 0, max_candidates = 0;
    size_t candidate_total = 0;
    int candidate_ids[256];
    int embed_dim = trainer->network->embed_dim;
    double *input = calloc(trainer->network->input_size, sizeof(double));
    double *output = calloc(trainer->network->output_size, sizeof(double));

    for (size_t t = 0; t < limit; t++) {
        for (int rotation = 0; rotation < 3; rotation++) {
            int first_id = chain->triangles[t].word_ids[rotation];
            int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
            int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];
            int first_role = use_mode_b ? 0 : chain->triangles[t].role_ids[rotation];
            int second_role = use_mode_b ? 0 : chain->triangles[t].role_ids[(rotation + 1) % 3];

            ContextCandidate evidence[256];
            size_t count = 0;
            if (use_mode_b) {
                count = context_graph_collect_candidate_evidence_relational(
                    graph, rel_reg, first_id, second_id, rotation, evidence, 256);
            } else {
                count = context_graph_collect_candidate_evidence_fallback(
                    graph, first_id, first_role, second_id, second_role, rotation,
                    evidence, 256);
            }
            total++;
            if (count == 0) continue;
            pair_queries++;
            candidate_total += count;
            if (count > 1) ambiguous_pairs++;
            if ((int)count > max_candidates) max_candidates = (int)count;

            fill_model_input(input, 0, first_id, first_role, trainer->network);
            fill_model_input(input, 1, second_id, second_role, trainer->network);

            backprop_predict(trainer, input, output);
            double scores[256];
            double target_score = -INFINITY;
            int found = 0;
            for (size_t c = 0; c < count; c++) {
                int cand_word_id = evidence[c].word_id;
                candidate_ids[c] = cand_word_id;
                double distance = 0.0;
                for (int d = 0; d < embed_dim; d++) {
                    double delta = output[2 * embed_dim + d] -
                        trainer->network->embeddings[(cand_word_id - 1) * embed_dim + d];
                    distance += delta * delta;
                }
                int occurrence = evidence[c].occurrence_count;
                int neighbors = evidence[c].neighbor_count;
                double match_score = evidence[c].match_score;
                double base_score = -distance +
                    0.75 * log(1.0 + occurrence) +
                    0.50 * log(1.0 + neighbors);
                if (use_mode_b) {
                    scores[c] = base_score * (0.5 + match_score);
                } else {
                    scores[c] = base_score + match_score;
                }
                if (cand_word_id == target_id) target_score = scores[c];
            }

            for (size_t c = 0; c < count; c++) {
                if (candidate_ids[c] == target_id) found = 1;
            }
            if (found) {

                int rank = 0;
                for (size_t c = 0; c < count; c++) if (scores[c] > target_score) rank++;
                graph_hits++;
                neural_top1 += rank == 0;
                neural_top5 += rank < 5;
            }
        }
    }

    if (use_mode_b && rel_reg && pair_queries > 0) {
        double thresholds[7] = {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30};
        printf("\n=== Phase 2B Positional Filter Diagnostic Sweep ===\n");
        printf("%-10s %-18s %-24s %-14s\n", "Threshold", "Avg Remaining Cand", "Cond Gold Retention", "Reduction");
        printf("-----------------------------------------------------------------------\n");
        for (int t_idx = 0; t_idx < 7; t_idx++) {
            double th = thresholds[t_idx];
            size_t total_retained = 0;
            int gold_retained_count = 0;

            for (size_t t = 0; t < limit; t++) {
                for (int rotation = 0; rotation < 3; rotation++) {
                    int first_id = chain->triangles[t].word_ids[rotation];
                    int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
                    int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];

                    ContextCandidate evidence[256];
                    size_t count = context_graph_collect_candidate_evidence_relational(
                        graph, rel_reg, first_id, second_id, rotation, evidence, 256);
                    if (count == 0) continue;

                    int target_position = (rotation >= 0) ? (rotation + 2) % 3 : 2;
                    size_t retained = 0;
                    int gold_kept = 0;

                    for (size_t c = 0; c < count; c++) {
                        int cand_id = evidence[c].word_id;
                        if (cand_id <= 0 || (size_t)cand_id > rel_reg->vocab_size) continue;
                        const RelationalWordStats *stats = &rel_reg->word_stats[cand_id];
                        double total_cnt = (double)(stats->left_count + stats->center_count + stats->right_count);
                        double target_ratio = 0.0;
                        if (total_cnt > 0.0) {
                            if (target_position == 0) target_ratio = (double)stats->left_count / total_cnt;
                            else if (target_position == 1) target_ratio = (double)stats->center_count / total_cnt;
                            else target_ratio = (double)stats->right_count / total_cnt;
                        }

                        if (target_ratio >= th) {
                            retained++;
                            if (cand_id == target_id) gold_kept = 1;
                        }
                    }
                    total_retained += retained;
                    if (gold_kept) gold_retained_count++;
                }
            }
            double avg_remaining = (double)total_retained / pair_queries;
            double gold_retention = 100.0 * gold_retained_count / pair_queries;
            double reduction = 100.0 * (1.0 - (double)total_retained / candidate_total);
            printf("%-10.2f %-18.2f %-23.1f%% %-13.1f%%\n", th, avg_remaining, gold_retention, reduction);
        }
        printf("================================================-----------------------\n");
    }

    printf("\n=== Context Evaluation (%zu triangles) ===\n", limit);
    printf("Rotation queries: %d\n", total);
    if (total > 0) {
        printf("Graph recall: %.1f%%\n", 100.0 * graph_hits / total);
        printf("Neural top-1: %.1f%%\n", 100.0 * neural_top1 / total);
        printf("Neural top-5: %.1f%%\n", 100.0 * neural_top5 / total);
        printf("Pair coverage: %.1f%% (%d/%d)\n",
               100.0 * pair_queries / total, pair_queries, total);
        if (pair_queries > 0) {
            printf("Average candidates per known pair: %.2f\n",
                   (double)candidate_total / pair_queries);
            printf("Ambiguous pairs: %.1f%% (%d/%d), max candidates: %d\n",
                   100.0 * ambiguous_pairs / pair_queries,
                   ambiguous_pairs, pair_queries, max_candidates);
            printf("Conditional graph recall: %.1f%%\n",
                   100.0 * graph_hits / pair_queries);
            printf("Conditional neural top-1: %.1f%%\n",
                   100.0 * neural_top1 / pair_queries);
            printf("Conditional neural top-5: %.1f%%\n",
                   100.0 * neural_top5 / pair_queries);

            printf("\n=== Candidate Discovery & Graph Recall Loss Analysis ===\n");
            printf("Total Rotation Queries: %d\n", total);
            printf("  ├─ Known Context Pair Queries (Covered): %d (%.1f%%)\n",
                   pair_queries, 100.0 * pair_queries / total);
            printf("  │   ├─ Graph Recall Hits (Gold in Graph): %d (%.1f%% of covered)\n",
                   graph_hits, 100.0 * graph_hits / pair_queries);
            printf("  │   └─ Graph Recall Misses (Gold absent in covered graph): %d (%.1f%% of covered)\n",
                   pair_queries - graph_hits, 100.0 * (pair_queries - graph_hits) / pair_queries);
            printf("  └─ Unseen Context Pair Queries (Uncovered): %d (%.1f%%)\n",
                   total - pair_queries, 100.0 * (total - pair_queries) / total);

            /* Diagnostic Provenance Audit for the 106 Covered Misses */
            int gold_seen_in_train_count = 0;
            int gold_never_seen_in_train_count = 0;
            int gold_truncated_by_max_count = 0;

            for (size_t t = 0; t < limit; t++) {
                for (int rotation = 0; rotation < 3; rotation++) {
                    int first_id = chain->triangles[t].word_ids[rotation];
                    int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
                    int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];

                    ContextCandidate evidence[256];
                    size_t count = context_graph_collect_candidate_evidence_relational(
                        graph, rel_reg, first_id, second_id, rotation, evidence, 256);
                    if (count == 0) continue;

                    int in_candidates = 0;
                    for (size_t c = 0; c < count; c++) {
                        if (evidence[c].word_id == target_id) { in_candidates = 1; break; }
                    }

                    if (!in_candidates) {
                        /* Check if (first_id, second_id, target_id) actually exists anywhere in graph nodes */
                        int seen_in_graph_nodes = 0;
                        for (size_t i = 0; i < graph->node_count; i++) {
                            const ContextNode *node = &graph->nodes[i];
                            if (node->type == CONTEXT_TRIANGLE_NODE &&
                                node->word_ids[0] == first_id &&
                                node->word_ids[1] == second_id &&
                                node->rotation == rotation) {
                                if (node->word_ids[2] == target_id) {
                                    seen_in_graph_nodes = 1;
                                    break;
                                }
                            }
                        }
                        if (seen_in_graph_nodes) {
                            gold_seen_in_train_count++;
                            if (count >= 256) gold_truncated_by_max_count++;
                        } else {
                            gold_never_seen_in_train_count++;
                        }
                    }
                }
            }

            printf("\n--- Breakdown of the %d Known-Context Misses ---\n", pair_queries - graph_hits);
            printf("  ├─ Gold Target ACTUALLY in Training Graph (Information Loss / Truncation): %d\n",
                   gold_seen_in_train_count);
            if (gold_seen_in_train_count > 0) {
                printf("  │   └─ Of which were truncated by candidate capacity cap (256): %d\n",
                       gold_truncated_by_max_count);
            }
            printf("  └─ Gold Target NEVER seen with this exact pair in Training (Generalization gap): %d\n",
                   gold_never_seen_in_train_count);
            printf("=========================================================\n");

            /* Diagnostic Audit for the 84 Exact-Pair Misses */
            if (gold_never_seen_in_train_count > 0) {
                int recoverable = 0;
                int pos_ok_trans_weak = 0;
                int trans_ok_pos_weak = 0;
                int no_relational_support = 0;

                for (size_t t = 0; t < limit; t++) {
                    for (int rotation = 0; rotation < 3; rotation++) {
                        int first_id = chain->triangles[t].word_ids[rotation];
                        int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
                        int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];
                        if (target_id <= 0) continue;

                        ContextCandidate evidence[256];
                        size_t count = context_graph_collect_candidate_evidence_relational(
                            graph, rel_reg, first_id, second_id, rotation, evidence, 256);
                        if (count == 0) continue;

                        int in_candidates = 0;
                        for (size_t c = 0; c < count; c++) {
                            if (evidence[c].word_id == target_id) { in_candidates = 1; break; }
                        }

                        if (!in_candidates) {
                            int seen_in_graph = 0;
                            for (size_t i = 0; i < graph->node_count; i++) {
                                const ContextNode *node = &graph->nodes[i];
                                if (node->type == CONTEXT_TRIANGLE_NODE &&
                                    node->word_ids[0] == first_id &&
                                    node->word_ids[1] == second_id &&
                                    node->rotation == rotation &&
                                    node->word_ids[2] == target_id) {
                                    seen_in_graph = 1;
                                    break;
                                }
                            }
                            if (!seen_in_graph && target_id > 0 && (size_t)target_id <= rel_reg->vocab_size) {
                                const RelationalWordStats *stats = &rel_reg->word_stats[target_id];
                                double total_cnt = (double)(stats->left_count + stats->center_count + stats->right_count);
                                int target_position = (rotation >= 0) ? (rotation + 2) % 3 : 2;
                                double pos_ratio = 0.0;
                                if (total_cnt > 0.0) {
                                    if (target_position == 0) pos_ratio = (double)stats->left_count / total_cnt;
                                    else if (target_position == 1) pos_ratio = (double)stats->center_count / total_cnt;
                                    else pos_ratio = (double)stats->right_count / total_cnt;
                                }

                                uint64_t fwd = 0, bwd = 0;
                                if (target_position == 2) {
                                    fwd = relational_registry_get_transition_count(rel_reg, second_id, target_id, 1);
                                    bwd = relational_registry_get_transition_count(rel_reg, target_id, second_id, 2);
                                } else if (target_position == 0) {
                                    fwd = relational_registry_get_transition_count(rel_reg, target_id, first_id, 0);
                                    bwd = relational_registry_get_transition_count(rel_reg, first_id, target_id, 3);
                                } else if (target_position == 1) {
                                    fwd = relational_registry_get_transition_count(rel_reg, second_id, target_id, 0);
                                    bwd = relational_registry_get_transition_count(rel_reg, target_id, second_id, 3);
                                }

                                int pos_ok = (pos_ratio >= 0.15);
                                int trans_ok = (fwd + bwd > 0);

                                if (pos_ok && trans_ok) recoverable++;
                                else if (pos_ok && !trans_ok) pos_ok_trans_weak++;
                                else if (!pos_ok && trans_ok) trans_ok_pos_weak++;
                                else no_relational_support++;
                            }
                        }
                    }
                }

                printf("\n=== Audit of the 84 Exact-Pair Misses (Relational Support Breakdown) ===\n");
                printf("  ├─ Fully Recoverable (Pos Ratio >= 0.15 AND Transition > 0): %d\n", recoverable);
                printf("  ├─ Positional Compatible (Pos Ratio >= 0.15), Weak Transition: %d\n", pos_ok_trans_weak);
                printf("  ├─ Transition Supported (Transition > 0), Weak Positional Ratio: %d\n", trans_ok_pos_weak);
                printf("  └─ No Relational Support (Neither Position nor Transition): %d\n", no_relational_support);
                printf("=========================================================================\n");

                /* Read-Only Relational Candidate Generation Audit */
                int gen_total_meaningful = 0;
                int gen_gold_recovered = 0;
                int gen_fully_supported_recovered = 0;
                int gen_pos_only_recovered = 0;
                int gen_no_support_recovered = 0;
                size_t gen_total_candidates_emitted = 0;

                for (size_t t = 0; t < limit; t++) {
                    for (int rotation = 0; rotation < 3; rotation++) {
                        int first_id = chain->triangles[t].word_ids[rotation];
                        int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
                        int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];
                        if (target_id <= 0) continue;

                        ContextCandidate evidence[256];
                        size_t count = context_graph_collect_candidate_evidence_relational(
                            graph, rel_reg, first_id, second_id, rotation, evidence, 256);
                        if (count == 0) continue;

                        int in_candidates = 0;
                        for (size_t c = 0; c < count; c++) {
                            if (evidence[c].word_id == target_id) { in_candidates = 1; break; }
                        }

                        if (!in_candidates) {
                            int seen_in_graph = 0;
                            for (size_t i = 0; i < graph->node_count; i++) {
                                const ContextNode *node = &graph->nodes[i];
                                if (node->type == CONTEXT_TRIANGLE_NODE &&
                                    node->word_ids[0] == first_id &&
                                    node->word_ids[1] == second_id &&
                                    node->rotation == rotation &&
                                    node->word_ids[2] == target_id) {
                                    seen_in_graph = 1;
                                    break;
                                }
                            }
                            if (!seen_in_graph && target_id > 0 && (size_t)target_id <= rel_reg->vocab_size) {
                                gen_total_meaningful++;
                                int target_position = (rotation >= 0) ? (rotation + 2) % 3 : 2;

                                /* Relational Candidate Generator: Collect words matching position ratio >= 0.15 & transition > 0 */
                                size_t generated_count = 0;
                                int gold_in_generated = 0;

                                for (size_t w = 1; w <= rel_reg->vocab_size; w++) {
                                    const RelationalWordStats *w_stats = &rel_reg->word_stats[w];
                                    double total_w = (double)(w_stats->left_count + w_stats->center_count + w_stats->right_count);
                                    if (total_w <= 0.0) continue;

                                    double pos_ratio = 0.0;
                                    if (target_position == 0) pos_ratio = (double)w_stats->left_count / total_w;
                                    else if (target_position == 1) pos_ratio = (double)w_stats->center_count / total_w;
                                    else pos_ratio = (double)w_stats->right_count / total_w;

                                    if (pos_ratio < 0.15) continue;

                                    uint64_t fwd = 0, bwd = 0;
                                    if (target_position == 2) {
                                        fwd = relational_registry_get_transition_count(rel_reg, second_id, (int)w, 1);
                                        bwd = relational_registry_get_transition_count(rel_reg, (int)w, second_id, 2);
                                    } else if (target_position == 0) {
                                        fwd = relational_registry_get_transition_count(rel_reg, (int)w, first_id, 0);
                                        bwd = relational_registry_get_transition_count(rel_reg, first_id, (int)w, 3);
                                    } else if (target_position == 1) {
                                        fwd = relational_registry_get_transition_count(rel_reg, second_id, (int)w, 0);
                                        bwd = relational_registry_get_transition_count(rel_reg, (int)w, second_id, 3);
                                    }

                                    if (fwd + bwd > 0) {
                                        generated_count++;
                                        if ((int)w == target_id) gold_in_generated = 1;
                                    }
                                }

                                gen_total_candidates_emitted += generated_count;
                                if (gold_in_generated) {
                                    gen_gold_recovered++;

                                    /* Categorize recovery group */
                                    const RelationalWordStats *g_stats = &rel_reg->word_stats[target_id];
                                    double g_total = (double)(g_stats->left_count + g_stats->center_count + g_stats->right_count);
                                    double g_pos = 0.0;
                                    if (g_total > 0.0) {
                                        if (target_position == 0) g_pos = (double)g_stats->left_count / g_total;
                                        else if (target_position == 1) g_pos = (double)g_stats->center_count / g_total;
                                        else g_pos = (double)g_stats->right_count / g_total;
                                    }
                                    uint64_t g_fwd = 0, g_bwd = 0;
                                    if (target_position == 2) {
                                        g_fwd = relational_registry_get_transition_count(rel_reg, second_id, target_id, 1);
                                        g_bwd = relational_registry_get_transition_count(rel_reg, target_id, second_id, 2);
                                    } else if (target_position == 0) {
                                        g_fwd = relational_registry_get_transition_count(rel_reg, target_id, first_id, 0);
                                        g_bwd = relational_registry_get_transition_count(rel_reg, first_id, target_id, 3);
                                    } else if (target_position == 1) {
                                        g_fwd = relational_registry_get_transition_count(rel_reg, second_id, target_id, 0);
                                        g_bwd = relational_registry_get_transition_count(rel_reg, target_id, second_id, 3);
                                    }

                                    if (g_pos >= 0.15 && (g_fwd + g_bwd > 0)) gen_fully_supported_recovered++;
                                    else if (g_pos >= 0.15) gen_pos_only_recovered++;
                                    else gen_no_support_recovered++;
                                }
                            }
                        }
                    }
                }

                printf("\n=== Read-Only Relational Candidate Generation Audit ===\n");
                printf("Meaningful exact-pair misses: %d\n", gen_total_meaningful);
                printf("Position Threshold: 0.15 | Transition Threshold: > 0\n");
                printf("Gold Target Recovered: %d / %d (%.1f%%)\n",
                       gen_gold_recovered, gen_total_meaningful,
                       gen_total_meaningful > 0 ? 100.0 * gen_gold_recovered / gen_total_meaningful : 0.0);
                printf("Average Candidates Generated per Query: %.2f\n",
                       gen_total_meaningful > 0 ? (double)gen_total_candidates_emitted / gen_total_meaningful : 0.0);
                printf("  ├─ Fully-Supported Group (33): %d / 33 (%.1f%%)\n",
                       gen_fully_supported_recovered,
                       33 > 0 ? 100.0 * gen_fully_supported_recovered / 33.0 : 0.0);
                printf("  ├─ Position-Only Group (38): %d / 38 (%.1f%%)\n",
                       gen_pos_only_recovered,
                       38 > 0 ? 100.0 * gen_pos_only_recovered / 38.0 : 0.0);
                printf("  └─ No-Support Group (10): %d / 10 (%.1f%%)\n",
                       gen_no_support_recovered,
                       10 > 0 ? 100.0 * gen_no_support_recovered / 10.0 : 0.0);
                printf("=======================================================\n");

                /* 2D Threshold Sweep: Position Ratio x Transition Count Cutoffs */
                double pos_ths[3] = {0.15, 0.25, 0.35};
                uint64_t trans_ths[4] = {1, 2, 3, 6}; /* >0, >1, >2, >5 */

                /* Build the exact same fallback-query set used by the
                 * original CPU sweep. Only these misses need vocabulary-wide
                 * generation; all other queries are already represented by
                 * graph candidates. */
                size_t query_capacity = limit * 3;
                ThresholdSweepQuery *sweep_queries = calloc(
                    query_capacity ? query_capacity : 1, sizeof(*sweep_queries));
                size_t sweep_query_count = 0;
                if (sweep_queries) {
                    for (size_t t = 0; t < limit; t++) {
                        for (int rotation = 0; rotation < 3; rotation++) {
                            int first_id = chain->triangles[t].word_ids[rotation];
                            int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
                            int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];
                            if (target_id <= 0) continue;

                            ContextCandidate evidence[256];
                            size_t count = context_graph_collect_candidate_evidence_relational(
                                graph, rel_reg, first_id, second_id, rotation, evidence, 256);
                            if (count == 0) continue;

                            int in_candidates = 0;
                            for (size_t c = 0; c < count; c++) {
                                if (evidence[c].word_id == target_id) {
                                    in_candidates = 1;
                                    break;
                                }
                            }
                            if (in_candidates) continue;

                            int seen_in_graph = 0;
                            for (size_t i = 0; i < graph->node_count; i++) {
                                const ContextNode *node = &graph->nodes[i];
                                if (node->type == CONTEXT_TRIANGLE_NODE &&
                                    node->word_ids[0] == first_id &&
                                    node->word_ids[1] == second_id &&
                                    node->rotation == rotation &&
                                    node->word_ids[2] == target_id) {
                                    seen_in_graph = 1;
                                    break;
                                }
                            }
                            if (!seen_in_graph && (size_t)target_id <= rel_reg->vocab_size &&
                                sweep_query_count < query_capacity) {
                                sweep_queries[sweep_query_count++] = (ThresholdSweepQuery){
                                    first_id, second_id, target_id, (rotation + 2) % 3};
                            }
                        }
                    }
                }

                printf("\n=== 2D Threshold Sweep: Candidate Reduction vs Gold Recovery (81 Misses) ===\n");
                printf("%-10s %-12s %-20s %-20s %-15s\n", "Pos Thresh", "Trans Cutoff", "Gold Recovered", "Fully-Supp (33)", "Avg Candidates");
                printf("-----------------------------------------------------------------------------------------\n");

                ThresholdSweepResult sweep_results[12];
                int sweep_gpu = sweep_query_count > 0 &&
                    run_threshold_sweep_cuda(
                        sweep_queries, sweep_query_count, rel_reg->vocab_size,
                        rel_reg->word_stats, rel_reg->transitions,
                        rel_reg->transition_count, pos_ths, 3, trans_ths, 4,
                        sweep_results) == 0;
                if (!sweep_gpu) {
                    run_threshold_sweep_cpu(
                        sweep_queries, sweep_query_count, rel_reg, pos_ths, 3,
                        trans_ths, 4, sweep_results);
                    printf("[Threshold sweep: CPU fallback]\n");
                } else {
                    printf("[Threshold sweep: CUDA]\n");
                }
                for (int p_idx = 0; p_idx < 3; p_idx++) {
                    for (int t_idx = 0; t_idx < 4; t_idx++) {
                        int index = p_idx * 4 + t_idx;
                        char cutoff_str[16];
                        snprintf(cutoff_str, sizeof(cutoff_str), ">= %llu", (unsigned long long)trans_ths[t_idx]);
                        printf("%-10.2f %-12s %llu / 81 (%.1f%%)   %llu / 33 (%.1f%%)   %.2f\n",
                               pos_ths[p_idx], cutoff_str,
                               (unsigned long long)sweep_results[index].gold_recovered,
                               gen_total_meaningful > 0 ? 100.0 * sweep_results[index].gold_recovered / gen_total_meaningful : 0.0,
                               (unsigned long long)sweep_results[index].fully_supported_recovered,
                               33.0 > 0.0 ? 100.0 * sweep_results[index].fully_supported_recovered / 33.0 : 0.0,
                               gen_total_meaningful > 0 ? (double)sweep_results[index].candidates_emitted / gen_total_meaningful : 0.0);
                }
                printf("================================================-----------------------------------------\n");
                free(sweep_queries);
            }

            if (gold_seen_in_train_count > 0) {
                printf("\n=== Experiment A: Detailed Trace of the %d Confirmed Information-Loss Queries ===\n",
                       gold_seen_in_train_count);
                printf("%-5s %-15s %-15s %-15s %-10s %-12s %-18s\n",
                       "Rot", "Word1 (W0)", "Word2 (W1)", "Gold (W2)", "Cap(256)", "UncappedTotal", "Reason");
                printf("---------------------------------------------------------------------------------------------\n");
                for (size_t t = 0; t < limit; t++) {
                    for (int rotation = 0; rotation < 3; rotation++) {
                        int first_id = chain->triangles[t].word_ids[rotation];
                        int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
                        int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];

                        ContextCandidate evidence[256];
                        size_t count = context_graph_collect_candidate_evidence_relational(
                            graph, rel_reg, first_id, second_id, rotation, evidence, 256);
                        if (count == 0) continue;

                        int in_candidates = 0;
                        for (size_t c = 0; c < count; c++) {
                            if (evidence[c].word_id == target_id) { in_candidates = 1; break; }
                        }

                        if (!in_candidates) {
                            int seen_in_graph = 0;
                            for (size_t i = 0; i < graph->node_count; i++) {
                                const ContextNode *node = &graph->nodes[i];
                                if (node->type == CONTEXT_TRIANGLE_NODE &&
                                    node->word_ids[0] == first_id &&
                                    node->word_ids[1] == second_id &&
                                    node->rotation == rotation &&
                                    node->word_ids[2] == target_id) {
                                    seen_in_graph = 1;
                                    break;
                                }
                            }
                            if (seen_in_graph) {
                                ContextCandidate uncapped_evidence[4096];
                                size_t uncapped_count = context_graph_collect_candidate_evidence_relational(
                                    graph, rel_reg, first_id, second_id, rotation, uncapped_evidence, 4096);
                                const char *w1 = vocab_get_word(chain->vocab, first_id);
                                const char *w2 = vocab_get_word(chain->vocab, second_id);
                                const char *gold = vocab_get_word(chain->vocab, target_id);
                                const char *reason = (count >= 256) ? "256 Cap Truncation" : "Non-Cap Retrieval Miss";
                                printf("%-5d %-15s %-15s %-15s %-10zu %-12zu %-18s\n",
                                       rotation, w1 ? w1 : "?", w2 ? w2 : "?", gold ? gold : "?",
                                       count, uncapped_count, reason);

                                if (count < 256) {
                                    printf("  └── [Forensic Audit] Rot=%d Pair=[%s, %s] Target=%s (TargetID=%d)\n",
                                           rotation, w1 ? w1 : "?", w2 ? w2 : "?", gold ? gold : "?", target_id);
                                    size_t node_match_count = 0;
                                    for (size_t i = 0; i < graph->node_count; i++) {
                                        const ContextNode *node = &graph->nodes[i];
                                        if (node->type == CONTEXT_TRIANGLE_NODE &&
                                            node->word_ids[0] == first_id &&
                                            node->word_ids[1] == second_id) {
                                            node_match_count++;
                                            printf("        ├─ Node #%d rot=%d: w0=%d w1=%d w2=%d (Gold %s)\n",
                                                   node->id, node->rotation, node->word_ids[0], node->word_ids[1], node->word_ids[2],
                                                   (node->word_ids[2] == target_id) ? "MATCH" : "diff");
                                        }
                                    }
                                    printf("        └─ Total matching nodes found for pair: %zu\n", node_match_count);
                                }
                            }
                        }
                    }
                }
                printf("================================================---------------------------------------------\n");
            }
        }
    }
    free(input);
    free(output);
}
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const char *filename = argv[1];
    char *file_content = NULL;

    FILE *f = fopen(filename, "r");
    if (f) {
        fclose(f);
        file_content = read_file(filename);
        if (file_content) {
            printf("Loading file: %s\n", filename);
        } else {
            fprintf(stderr, "Error reading file: %s\n", filename);
            return 1;
        }
    } else {
        file_content = strdup(filename);
        if (!file_content) {
            fprintf(stderr, "Error allocating memory\n");
            return 1;
        }
        printf("Input: %s\n", filename);
    }

    int is_conllu = has_conllu_suffix(filename);
    TriangleChain *chain = is_conllu
        ? create_triangles_from_conllu(file_content)
        : create_triangles(file_content);
    if (!chain) {
        fprintf(stderr, "Failed to create triangles\n");
        free(file_content);
        return 1;
    }

    printf("Created %zu triangle(s)\n", chain->count);
    printf("Vocabulary size: %zu\n", chain->vocab->count);
    if (argc <= 2 || (strcmp(argv[2], "-backprop") != 0 &&
                      strcmp(argv[2], "-backprop-cuda") != 0 &&
                      strcmp(argv[2], "-train") != 0 &&
                      strcmp(argv[2], "-context-graph") != 0 &&
                      strcmp(argv[2], "-context-query") != 0 &&
                      strcmp(argv[2], "-predict") != 0 &&
                      strcmp(argv[2], "-evaluate-context") != 0 &&
                      strcmp(argv[2], "-heldout") != 0)) {
        print_triangles(chain);
    }

    if (argc > 2 && strcmp(argv[2], "-train") == 0) {
        printf("\nCreating training set...\n");
        TrainingSet *training = training_create(chain);
        if (!training) {
            fprintf(stderr, "Failed to create training set\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        printf("Training samples: %zu\n", training->count);

        printf("\nCreating neural network...\n");
        int max_vocab = chain->vocab->count > 100 ? 100 : chain->vocab->count;
        int hidden = 32;
        NeuralNetwork *nn = nn_create(4, hidden, max_vocab);
        if (!nn) {
            fprintf(stderr, "Failed to create neural network\n");
            training_free(training);
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        printf("Training (5 epochs)...\n");
        nn_train(nn, training, 5, 0.1);

        printf("\nEvaluating...\n");
        EvaluationOutput *eval = evaluate(nn, chain);
        if (eval) {
            evaluation_print(eval);
            evaluation_free(eval);
        }

        nn_save(nn, "model.bin");
        printf("\nModel saved to model.bin\n");

        nn_free(nn);
        training_free(training);
    } else if (argc > 2 && strcmp(argv[2], "-eval") == 0) {
        printf("\nLoading model...\n");
        NeuralNetwork *nn = nn_load("model.bin");
        if (!nn) {
            fprintf(stderr, "Failed to load model\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        printf("\nEvaluating...\n");
        EvaluationOutput *eval = evaluate(nn, chain);
        if (eval) {
            evaluation_print(eval);
            evaluation_free(eval);
        }

        nn_free(nn);
    } else if (argc > 2 && strcmp(argv[2], "-formula") == 0) {
        int power = 1;
        if (argc > 3) {
            power = atoi(argv[3]);
        }

        printf("Calculating formula with power %d...\n", power);

        FormulaResult *formula = formula_calculate(chain, power);
        if (formula) {
            formula_print(formula);
            formula_free(formula);
        }
    } else if (argc > 2 && strcmp(argv[2], "-matrix") == 0) {
        printf("Creating matrix table...\n");

        MatrixTable *matrix = matrix_create(chain);
        if (matrix) {
            matrix_print(matrix);
            matrix_save_csv(matrix, "matrix.csv");
            matrix_free(matrix);
        }
    } else if (argc > 2 && strcmp(argv[2], "-reconstruct") == 0) {
        printf("Reconstructing text...\n");

        ReconstructedText *text = reconstruct(chain);
        if (text) {
            reconstruct_print(text);
            reconstruct_compare(text, file_content);
            reconstruct_free(text);
        }
    } else if (argc > 2 && strcmp(argv[2], "-circle") == 0) {
        double multiplier = 1.0;
        if (argc > 3) {
            multiplier = atof(argv[3]);
        }

        printf("Analyzing conjunctions with multiplier %.2f...\n", multiplier);

        CircleChain *circles = circle_create(chain, multiplier);
        if (circles) {
            circle_print(circles);

            double impact = circle_calculate_impact(circles, chain);
            printf("\nTotal conjunction impact: %.0f\n", impact);

            circle_free(circles);
        }
    } else if (argc > 2 && strcmp(argv[2], "-graph") == 0) {
        double multiplier = 1.0;
        if (argc > 3) {
            multiplier = atof(argv[3]);
        }

        printf("Building graph with multiplier %.2f...\n", multiplier);

        CircleChain *circles = circle_create(chain, multiplier);
        Graph *graph = graph_create(chain, circles);
        if (graph) {
            graph_print(graph);
            graph_save_dot(graph, "graph.dot");
            graph_free(graph);
        }
    } else if (argc > 2 && strcmp(argv[2], "-context-graph") == 0) {
        ContextGraph *context_graph = context_graph_create(chain);
        if (!context_graph) {
            fprintf(stderr, "Failed to create context graph\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }
        context_graph_print(context_graph);
        context_graph_save_dot(context_graph, "context_graph.dot");
        context_graph_free(context_graph);
    } else if (argc > 4 && strcmp(argv[2], "-context-query") == 0) {
        int first_id = 0;
        int second_id = 0;
        for (size_t i = 0; i < chain->vocab->count; i++) {
            if (strcmp(chain->vocab->words[i].text, argv[3]) == 0) first_id = (int)i + 1;
            if (strcmp(chain->vocab->words[i].text, argv[4]) == 0) second_id = (int)i + 1;
        }
        if (first_id == 0 || second_id == 0) {
            fprintf(stderr, "One or both query words are not in the vocabulary\n");
        } else {
            ContextGraph *context_graph = context_graph_create(chain);
            if (!context_graph) {
                fprintf(stderr, "Failed to create context graph\n");
                free_triangles(chain);
                free(file_content);
                return 1;
            }
            context_graph_query(context_graph, chain, first_id, second_id, 20);
            context_graph_free(context_graph);
        }
    } else if (argc > 4 && strcmp(argv[2], "-predict") == 0) {
        int first_id = 0;
        int second_id = 0;
        for (size_t i = 0; i < chain->vocab->count; i++) {
            if (strcmp(chain->vocab->words[i].text, argv[3]) == 0) first_id = (int)i + 1;
            if (strcmp(chain->vocab->words[i].text, argv[4]) == 0) second_id = (int)i + 1;
        }
        if (first_id == 0 || second_id == 0) {
            fprintf(stderr, "One or both prediction words are not in the vocabulary\n");
        } else {
            BackpropTrainer *trainer = backprop_load_model("backprop_model.bin");
            ContextGraph *context_graph = context_graph_create(chain);
            if (!trainer || !context_graph) {
                fprintf(stderr, "Need backprop_model.bin and a context graph\n");
                backprop_free(trainer);
                context_graph_free(context_graph);
                free_triangles(chain);
                free(file_content);
                return 1;
            }

            int candidate_ids[256];
            size_t candidate_count = 0;
            for (size_t i = 0; i < context_graph->node_count; i++) {
                const ContextNode *node = &context_graph->nodes[i];
                if (node->type != CONTEXT_TRIANGLE_NODE ||
                    node->word_ids[0] != first_id ||
                    node->word_ids[1] != second_id) continue;

                int candidates_to_add[2] = {node->word_ids[2], 0};
                for (size_t b = 0; b < context_graph->bond_count; b++) {
                    const ContextBond *bond = &context_graph->bonds[b];
                    if (bond->from_id == node->id && bond->type == BOND_NEIGHBOR) {
                        candidates_to_add[1] = context_graph->nodes[bond->to_id - 1].word_ids[2];
                        break;
                    }
                }
                for (int c = 0; c < 2; c++) {
                    if (candidates_to_add[c] <= 0) continue;
                    int exists = 0;
                    for (size_t j = 0; j < candidate_count; j++) {
                        if (candidate_ids[j] == candidates_to_add[c]) exists = 1;
                    }
                    if (!exists && candidate_count < 256) {
                        candidate_ids[candidate_count++] = candidates_to_add[c];
                    }
                }
            } 

            if (candidate_count == 0) {
                printf("No graph candidates found for [%s, %s].\n", argv[3], argv[4]);
            } else {
                int embed_dim = trainer->network->embed_dim;
                double *input = calloc(trainer->network->input_size, sizeof(double));
                double *output = calloc(trainer->network->output_size, sizeof(double));
                fill_model_input(input, 0, first_id, 0, trainer->network);
                fill_model_input(input, 1, second_id, 0, trainer->network);
                backprop_predict(trainer, input, output);

                double candidate_distances[256];
                ContextCandidate evidence[256];
                size_t evidence_count = context_graph_collect_candidate_evidence(
                    context_graph, first_id, second_id, evidence, 256);
                for (size_t i = 0; i < candidate_count; i++) {
                    int candidate_index = candidate_ids[i] - 1;
                    candidate_distances[i] = 0.0;
                    for (int d = 0; d < embed_dim; d++) {
                        double delta = output[2 * embed_dim + d] -
                            trainer->network->embeddings[candidate_index * embed_dim + d];
                        candidate_distances[i] += delta * delta;
                    }
                }
                double candidate_scores[256];
                for (size_t i = 0; i < candidate_count; i++) {
                    int occurrence = 0;
                    int neighbors = 0;
                    for (size_t e = 0; e < evidence_count; e++) {
                        if (evidence[e].word_id == candidate_ids[i]) {
                            occurrence = evidence[e].occurrence_count;
                            neighbors = evidence[e].neighbor_count;
                            break;
                        }
                    }
                    candidate_scores[i] = -candidate_distances[i] +
                        0.75 * log(1.0 + occurrence) +
                        0.50 * log(1.0 + neighbors);
                }
                for (size_t i = 1; i < candidate_count; i++) {
                    int id = candidate_ids[i];
                    double distance = candidate_distances[i];
                    double score = candidate_scores[i];
                    size_t j = i;
                    while (j > 0 && candidate_scores[j - 1] < score) {
                        candidate_ids[j] = candidate_ids[j - 1];
                        candidate_distances[j] = candidate_distances[j - 1];
                        candidate_scores[j] = candidate_scores[j - 1];
                        j--;
                    }
                    candidate_ids[j] = id;
                    candidate_distances[j] = distance;
                    candidate_scores[j] = score;
                }

                printf("\n=== Neural Ranking for [%s, %s] ===\n", argv[3], argv[4]);
                size_t shown = candidate_count < 10 ? candidate_count : 10;
                printf("Graph candidates: %zu (showing top %zu)\n",
                       candidate_count, shown);
                for (size_t i = 0; i < shown; i++) {
                    const char *candidate_word = vocab_get_word(
                        chain->vocab, candidate_ids[i]);
                    int occurrence = 0;
                    int neighbors = 0;
                    for (size_t e = 0; e < evidence_count; e++) {
                        if (evidence[e].word_id == candidate_ids[i]) {
                            occurrence = evidence[e].occurrence_count;
                            neighbors = evidence[e].neighbor_count;
                            break;
                        }
                    }
                    printf("  candidate=%s (id=%d) score=%.6f distance=%.6f "
                           "occurrences=%d neighbors=%d\n",
                           candidate_word ? candidate_word : "<unknown>",
                           candidate_ids[i], candidate_scores[i],
                           candidate_distances[i], occurrence, neighbors);
                }
                free(input);
                free(output);
            }
            context_graph_free(context_graph);
            backprop_free(trainer);
        }
    } else if (argc > 2 && strcmp(argv[2], "-evaluate-context") == 0) {
        size_t limit = 100;
        if (argc > 3) {
            int requested = atoi(argv[3]);
            if (requested > 0) limit = (size_t)requested;
        }
        if (limit > chain->count) limit = chain->count;

        BackpropTrainer *trainer = backprop_load_model("backprop_model.bin");
        ContextGraph *context_graph = context_graph_create(chain);
        if (!trainer || !context_graph) {
            fprintf(stderr, "Need backprop_model.bin and a context graph\n");
            backprop_free(trainer);
            context_graph_free(context_graph);
        } else {
            int graph_hits = 0;
            int neural_top1 = 0;
            int neural_top5 = 0;
            int total = 0;
            int candidate_ids[256];
            int embed_dim = trainer->network->embed_dim;
            double *input = calloc(trainer->network->input_size, sizeof(double));
            double *output = calloc(trainer->network->output_size, sizeof(double));

            for (size_t t = 0; t < limit; t++) {
                for (int rotation = 0; rotation < 3; rotation++) {
                    int first_id = chain->triangles[t].word_ids[rotation];
                    int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
                    int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];
                    size_t candidate_count = context_graph_collect_candidates(
                        context_graph, first_id, second_id, candidate_ids, 256);
                    if (candidate_count == 0) continue;
                    total++;
                    ContextCandidate evidence[256];
                    size_t evidence_count = context_graph_collect_candidate_evidence(
                        context_graph, first_id, second_id, evidence, 256);
                    double candidate_scores[256];
                    double target_score = -INFINITY;
                    fill_model_input(input, 0, first_id,
                                     chain->triangles[t].role_ids[rotation], trainer->network);
                    fill_model_input(input, 1, second_id,
                                     chain->triangles[t].role_ids[(rotation + 1) % 3], trainer->network);
                    backprop_predict(trainer, input, output);
                    for (size_t c = 0; c < candidate_count; c++) {
                        double distance = 0.0;
                        for (int d = 0; d < embed_dim; d++) {
                            double delta = output[2 * embed_dim + d] -
                                trainer->network->embeddings[(candidate_ids[c] - 1) * embed_dim + d];
                            distance += delta * delta;
                        }
                        int occurrence = 0;
                        int neighbors = 0;
                        for (size_t e = 0; e < evidence_count; e++) {
                            if (evidence[e].word_id == candidate_ids[c]) {
                                occurrence = evidence[e].occurrence_count;
                                neighbors = evidence[e].neighbor_count;
                                break;
                            }
                        }
                        candidate_scores[c] = -distance +
                            0.75 * log(1.0 + occurrence) +
                            0.50 * log(1.0 + neighbors);
                        if (candidate_ids[c] == target_id) target_score = candidate_scores[c];
                    }
                    int found = 0;
                    for (size_t c = 0; c < candidate_count; c++) {
                        if (candidate_ids[c] == target_id) found = 1;
                    }
                    int target_rank = 0;
                    if (found) {
                        for (size_t c = 0; c < candidate_count; c++) {
                            if (candidate_scores[c] > target_score) target_rank++;
                        }
                    }
                    graph_hits += found;
                    neural_top1 += found && target_rank == 0;
                    neural_top5 += found && target_rank < 5;
                }
            }
            printf("\n=== Context Evaluation (%zu triangles) ===\n", limit);
            printf("Rotation queries: %d\n", total);
            if (total > 0) {
                printf("Graph recall: %.1f%%\n", 100.0 * graph_hits / total);
                printf("Neural top-1: %.1f%%\n", 100.0 * neural_top1 / total);
                printf("Neural top-5: %.1f%%\n", 100.0 * neural_top5 / total);
            }
            free(input);
            free(output);
            context_graph_free(context_graph);
            backprop_free(trainer);
        }
    } else if (argc > 3 && strcmp(argv[2], "-heldout") == 0) {
        char *all_content = read_file("data/combined_all.txt");
        char *test_content = read_file(argv[3]);
        TriangleChain *global_chain = all_content ? create_triangles(all_content) : NULL;
        if (!global_chain || !test_content) {
            fprintf(stderr, "Held-out experiment requires data/combined_all.txt and a test file\n");
            free(test_content);
            free(all_content);
            free_triangles(global_chain);
        } else {
            Vocabulary *shared_vocab = global_chain->vocab;
            global_chain->vocab = NULL;
            global_chain->owns_vocab = 0;
            free_triangles(global_chain);

            /* Combine the primary training document with any additional
             * training documents listed after the held-out test file. */
            size_t train_size = strlen(file_content);
            char *combined_train = malloc(train_size + 1);
            if (combined_train) {
                memcpy(combined_train, file_content, train_size);
                combined_train[train_size] = '\0';
            }
            for (int arg = 4; combined_train && arg < argc; arg++) {
                if (argv[arg][0] == '-') {
                    if (strcmp(argv[arg], "-mode") == 0 && arg + 1 < argc) arg++;
                    continue;
                }
                char *extra_content = read_file(argv[arg]);

                if (!extra_content) {
                    fprintf(stderr, "Unable to read additional training file: %s\n", argv[arg]);
                    free(combined_train);
                    combined_train = NULL;
                    break;
                }

                size_t extra_size = strlen(extra_content);
                char *expanded = realloc(combined_train,
                                         train_size + 1 + extra_size + 1);
                if (!expanded) {
                    free(extra_content);
                    free(combined_train);
                    combined_train = NULL;
                    break;
                }
                combined_train = expanded;
                combined_train[train_size] = ' ';
                memcpy(combined_train + train_size + 1, extra_content, extra_size + 1);
                train_size += 1 + extra_size;
                free(extra_content);
            }

            TriangleChain *train_chain = NULL;
            if (combined_train) {
                train_chain = has_conllu_suffix(filename)
                    ? create_triangles_from_conllu_with_vocab(combined_train, shared_vocab)
                    : create_triangles_with_vocab(combined_train, shared_vocab);
            }
            free(combined_train);
            TriangleChain *test_chain = has_conllu_suffix(argv[3])
                ? create_triangles_from_conllu_with_vocab(test_content, shared_vocab)
                : create_triangles_with_vocab(test_content, shared_vocab);
            if (!train_chain || !test_chain) {
                fprintf(stderr, "Failed to create shared-vocabulary train/test chains\n");
                free_triangles(train_chain);
                free_triangles(test_chain);
                vocab_free(shared_vocab);
            } else {
                RelationalRegistry *rel_reg = relational_registry_create(shared_vocab->count);
                if (rel_reg) {
                    relational_registry_ingest_chain(rel_reg, train_chain);
                }

                int use_mode_b = 0;
                for (int arg = 1; arg < argc; arg++) {
                    if (strcmp(argv[arg], "-mode") == 0 && arg + 1 < argc && strcmp(argv[arg + 1], "B") == 0) {
                        use_mode_b = 1;
                        break;
                    }
                    if (strcmp(argv[arg], "-modeB") == 0 || strcmp(argv[arg], "--modeB") == 0) {
                        use_mode_b = 1;
                        break;
                    }
                }

                printf("\n=== Held-out Experiment ===\n");
                printf("Training triangles: %zu\n", train_chain->count);
                printf("Held-out triangles: %zu\n", test_chain->count);
                printf("Shared vocabulary: %zu words\n", shared_vocab->count);
                printf("Training documents: %d\n", argc - 3);

                if (train_chain->count == 0) {
                    fprintf(stderr,
                            "No training triangles were created. Check that the training file exists and is valid CoNLL-U.\n");
                    free_triangles(train_chain);
                    free_triangles(test_chain);
                    vocab_free(shared_vocab);
                    free(file_content);
                    return 1;
                }

                if (use_mode_b && rel_reg) {
                    printf("\n=== Running in Unsupervised Discovery Mode (Mode B) ===\n");
                    relational_registry_report(rel_reg, shared_vocab, 15);
                } else {
                    printf("\n=== Running in Supervised Baseline Mode (Mode A) ===\n");
                }


                int training_epochs = 50;
                const char *epochs_env = getenv("TRAIN_EPOCHS");
                if (epochs_env && *epochs_env) {
                    char *end = NULL;
                    long parsed = strtol(epochs_env, &end, 10);
                    if (end != epochs_env && *end == '\0' && parsed > 0 && parsed <= 10000)
                        training_epochs = (int)parsed;
                }
                printf("Training epochs: %d\n", training_epochs);
                BackpropTrainer *trainer = backprop_create(
                    (int)shared_vocab->count, 32, 128, 96, training_epochs, 0.1);
                if (!trainer) {
                    fprintf(stderr, "Failed to create backprop trainer\n");
                } else {
                    if (backprop_train_cuda(trainer, train_chain, 2) != 0) {
                        printf("CUDA training unavailable or failed. Falling back to CPU backprop_train...\n");
                        backprop_train(trainer, train_chain);
                    }
                    fprintf(stderr, "[main] training returned\n");
                    backprop_save_model(trainer, "backprop_model_heldout.bin");
                    fprintf(stderr, "[main] model save returned\n");
                    ContextGraph *train_graph = context_graph_create(train_chain);
                    fprintf(stderr, "[main] context graph construction returned\n");
                    const char *gpu_eval_only = getenv("GPU_EVAL_ONLY");
                    if (gpu_eval_only && strcmp(gpu_eval_only, "1") == 0) {
                        fprintf(stderr, "[main] GPU_EVAL_ONLY enabled\n");
                        if (run_gpu_evaluation_only(test_chain, train_graph, rel_reg, 100) != 0)
                            fprintf(stderr, "[main] GPU evaluation failed\n");
                        run_neural_evaluation_only(trainer, test_chain, 100);
                    } else {
                        evaluate_context_model(trainer, test_chain, train_graph, rel_reg, use_mode_b, 100);
                        fprintf(stderr, "[main] context evaluation returned\n");
                    }
                    context_graph_free(train_graph);
                    fprintf(stderr, "[main] context graph free returned\n");
                    backprop_free(trainer);
                    fprintf(stderr, "[main] trainer free returned\n");
                }
                if (rel_reg) relational_registry_free(rel_reg);
                fprintf(stderr, "[main] relational registry free returned\n");
                free_triangles(train_chain);
                fprintf(stderr, "[main] train chain free returned\n");
                free_triangles(test_chain);
                fprintf(stderr, "[main] test chain free returned\n");
                vocab_free(shared_vocab);
                fprintf(stderr, "[main] shared vocabulary free returned\n");
            }
        }

    } else if (argc > 2 && strcmp(argv[2], "-learn") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Error: -learn requires new text/file argument\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        const char *new_input = argv[3];
        char *new_content = NULL;

        FILE *nf = fopen(new_input, "r");
        if (nf) {
            fclose(nf);
            new_content = read_file(new_input);
            printf("Learning from file: %s\n", new_input);
        } else {
            new_content = strdup(new_input);
            printf("Learning from: %s\n", new_input);
        }

        if (!new_content) {
            fprintf(stderr, "Error reading new input\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        TriangleChain *new_chain = create_triangles(new_content);
        if (!new_chain) {
            fprintf(stderr, "Failed to create triangles for new input\n");
            free(new_content);
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        printf("\nOld: %zu triangles, %zu words\n", chain->count, chain->vocab->count);
        printf("New: %zu triangles, %zu words\n", new_chain->count, new_chain->vocab->count);

        LearningOutput *learning = learn_compare(chain, new_chain);
        if (learning) {
            learning_print(learning);

            TriangleChain *merged = learn_merge(chain, new_chain);
            if (merged) {
                printf("\nMerged: %zu triangles, %zu words\n", merged->count, merged->vocab->count);

                FormulaResult *formula_old = formula_calculate(chain, 1);
                FormulaResult *formula_new = formula_calculate(new_chain, 1);
                FormulaResult *formula_merged = formula_calculate(merged, 1);

                if (formula_old && formula_new && formula_merged) {
                    printf("\nFormula comparison:\n");
                    printf("  Old total:   %.0f\n", formula_old->total);
                    printf("  New total:   %.0f\n", formula_new->total);
                    printf("  Merged total: %.0f\n", formula_merged->total);
                    printf("  Old + New:   %.0f\n", formula_old->total + formula_new->total);
                    formula_free(formula_old);
                    formula_free(formula_new);
                    formula_free(formula_merged);
                }

                for (size_t i = 0; i < merged->count; i++) {
                    for (int p = 0; p < 3; p++) {
                        free(merged->triangles[i].words[p]);
                    }
                }
                free(merged->triangles);
                free(merged->vocab);
                free(merged);
            }

            learning_free(learning);
        }

        free(new_content);
        free_triangles(new_chain);
    } else if (argc > 2 &&
               (strcmp(argv[2], "-backprop") == 0 ||
                strcmp(argv[2], "-backprop-cuda") == 0)) {
        int use_cuda = strcmp(argv[2], "-backprop-cuda") == 0;
        printf("Starting %s backpropagation training on full vocabulary...\n\n",
               use_cuda ? "CUDA" : "CPU");

        int vocab_size = (int)chain->vocab->count;
        int embed_dim = 32;
        BackpropTrainer *trainer = backprop_create(vocab_size, embed_dim, 128,
                                                    3 * embed_dim, 50, 0.1);
        if (!trainer) {
            fprintf(stderr, "Failed to create trainer\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        if (use_cuda) {
            if (backprop_train_cuda(trainer, chain, 2) != 0) {
                backprop_free(trainer);
                free_triangles(chain);
                free(file_content);
                return 1;
            }
        } else {
            backprop_train(trainer, chain);
        }
        backprop_save_model(trainer, "backprop_model.bin");

        printf("\n=== Final Results ===\n");
        printf("Vocabulary: %zu words\n", chain->vocab->count);
        printf("Triangles: %zu\n", chain->count);

        if (trainer->log_count > 0) {
            printf("Final loss: %.6f\n", trainer->logs[trainer->log_count - 1].loss);
        }

        printf("\nTesting predictions on first 10 triangles:\n");
        int evaluation_correct = 0;
        int evaluation_total = 0;
        for (size_t t = 0; t < 10 && t < chain->count; t++) {
            int pred_ids[3];
            for (int target_position = 0; target_position < 3; target_position++) {
                double input[2 * (32 + TRIANGLE_ROLE_FEATURE_DIM)];
                for (int q = 0; q < 2; q++) {
                    int source_position = (target_position + q + 1) % 3;
                    fill_model_input(input, q,
                                     chain->triangles[t].word_ids[source_position],
                                     chain->triangles[t].role_ids[source_position],
                                     trainer->network);
                }

                double output[3 * 32];
                backprop_predict(trainer, input, output);
                pred_ids[target_position] = nearest_word_id(
                    trainer->network, output + target_position * embed_dim);
            }

            int pred_id0 = pred_ids[0];
            int pred_id1 = pred_ids[1];
            int pred_id2 = pred_ids[2];
            int predictions[3] = {pred_id0, pred_id1, pred_id2};
            for (int p = 0; p < 3; p++) {
                evaluation_correct += predictions[p] == chain->triangles[t].word_ids[p];
                evaluation_total++;
            }

            printf("  Triangle %zu: actual=[%d,%d,%d] predicted_ids=[%d,%d,%d]\n",
                   t + 1,
                   chain->triangles[t].word_ids[0],
                   chain->triangles[t].word_ids[1],
                   chain->triangles[t].word_ids[2],
                   pred_id0, pred_id1, pred_id2);
        }
        if (evaluation_total > 0) {
            printf("Evaluation exact word accuracy: %.1f%% (%d/%d)\n",
                   100.0 * evaluation_correct / evaluation_total,
                   evaluation_correct, evaluation_total);
        }

        backprop_free(trainer);
    } else {
        ProbabilityMatrix *prob = prob_create(chain);
        if (prob) {
            prob_print_sequences(prob);
            prob_free(prob);
        }
    }

    free_triangles(chain);
    free(file_content);
    return 0;
}
