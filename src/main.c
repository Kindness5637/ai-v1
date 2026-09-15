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
        printf("%-10s %-18s %-16s %-14s\n", "Threshold", "Avg Remaining Cand", "Gold Retention", "Reduction");
        printf("-------------------------------------------------------------\n");
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
            printf("%-10.2f %-18.2f %-15.1f%% %-13.1f%%\n", th, avg_remaining, gold_retention, reduction);
        }
        printf("=============================================================\n");
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
        }
    }
    free(input);
    free(output);
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

                if (use_mode_b && rel_reg) {
                    printf("\n=== Running in Unsupervised Discovery Mode (Mode B) ===\n");
                    relational_registry_report(rel_reg, shared_vocab, 15);
                } else {
                    printf("\n=== Running in Supervised Baseline Mode (Mode A) ===\n");
                }


                BackpropTrainer *trainer = backprop_create(
                    (int)shared_vocab->count, 32, 128, 96, 50, 0.1);
                if (!trainer) {
                    fprintf(stderr, "Failed to create backprop trainer\n");
                } else {
                    if (backprop_train_cuda(trainer, train_chain, 2) != 0) {
                        printf("CUDA training unavailable or failed. Falling back to CPU backprop_train...\n");
                        backprop_train(trainer, train_chain);
                    }
                    backprop_save_model(trainer, "backprop_model_heldout.bin");
                    ContextGraph *train_graph = context_graph_create(train_chain);
                    evaluate_context_model(trainer, test_chain, train_graph, rel_reg, use_mode_b, 100);
                    context_graph_free(train_graph);
                    backprop_free(trainer);
                }
                if (rel_reg) relational_registry_free(rel_reg);
                free_triangles(train_chain);
                free_triangles(test_chain);
                vocab_free(shared_vocab);
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
