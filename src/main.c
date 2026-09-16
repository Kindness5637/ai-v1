#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/stat.h>
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

typedef struct {
    const char *dataset_path;
    long training_triangles;
    long heldout_triangles;
    int vocab_size;
    int training_vocab;
    int embed_dim;
    int hidden_dim;
    int negative_samples;
    int epochs;
    int seed;
    int gpu_count;
    const char *evaluation_mode;
    double exact_recall;
    double neural_top1;
    double neural_top5;
    double mrr;
    double avg_candidates;
} BaselineMetrics;

typedef struct {
    int first_id;
    int second_id;
    int target_id;
    int rotation;
    int target_position;
    int first_role;   /* UPOS role of first_id, as used at training time */
    int second_role;  /* UPOS role of second_id, as used at training time */
} EvaluationQuery;

typedef struct {
    const char *name;
    int query_count;
    int candidate_hits;
    int top1;
    int top5;
    double mrr_sum;
    size_t total_candidates;
    int nonempty_queries;
    int max_candidates;
    int cap_hit_queries;      /* queries whose candidate count hit the cap */
    double median_candidates; /* median per-query candidate count (all queries) */
    size_t *cand_counts;      /* per-query candidate counts, for median */
} UnifiedEvalMetrics;

static int cmp_size_t(const void *a, const void *b) {
    size_t x = *(const size_t *)a, y = *(const size_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* Per-query diagnostic record for the unified evaluator.
   Modes: 0=EXACT, 1=STRUCT, 2=UNION. rank[] is the 0-based rank of the
   gold target among that mode's candidates by the current scorer, or -1
   when the gold is absent from the candidate set. */
typedef struct {
    int first_id;
    int second_id;
    int rotation;
    int target_id;
    int count[3];
    int gold_in[3];
    int rank[3];
    int struct_gold_eligible; /* gold passed pos+trans thresholds (may be truncated) */
    int struct_gold_past_cap; /* eligible but dropped by the candidate buffer cap */
} UnifiedQueryRecord;

typedef struct {
    int found;
    int top1;
    int top5;
    double mrr_sum;
} BucketModeStats;

typedef struct {
    int queries;
    BucketModeStats mode[3];
} BucketStats;

typedef struct {
    int from_word_id;
    int to_word_id;
    int transition_type;
    uint64_t count;
    int used;
} TransitionLookupEntry;

typedef struct {
    TransitionLookupEntry *entries;
    size_t capacity;
} TransitionLookup;

static void write_baseline_report(const BaselineMetrics *m) {
    mkdir("results", 0755);
    FILE *f = fopen("results/baseline_report.txt", "w");
    if (!f) return;

    fprintf(f, "=== Frozen Baseline Measurement Snapshot ===\n\n");
    fprintf(f, "Configuration & Dataset:\n");
    fprintf(f, "  Dataset Path: %s\n", m->dataset_path ? m->dataset_path : "unknown");
    fprintf(f, "  Training Triangles: %ld\n", m->training_triangles);
    fprintf(f, "  Held-out Triangles: %ld\n", m->heldout_triangles);
    fprintf(f, "  Vocabulary Size: %d\n", m->vocab_size);
    fprintf(f, "  Training Vocabulary: %d\n", m->training_vocab);
    fprintf(f, "  Embedding Dim: %d\n", m->embed_dim);
    fprintf(f, "  Hidden Dim: %d\n", m->hidden_dim);
    fprintf(f, "  Negative Samples: %d\n", m->negative_samples);
    fprintf(f, "  Epochs: %d\n", m->epochs);
    fprintf(f, "  Seed: %d\n", m->seed);
    fprintf(f, "  GPU Count: %d\n\n", m->gpu_count);
    fprintf(f, "Evaluation Metrics:\n");
    fprintf(f, "  Evaluation Mode: %s\n", m->evaluation_mode ? m->evaluation_mode : "EXACT");
    fprintf(f, "  Exact Candidate Recall: %.2f%%\n", m->exact_recall);
    fprintf(f, "  Neural Top-1 Accuracy: %.2f%%\n", m->neural_top1);
    fprintf(f, "  Neural Top-5 Accuracy: %.2f%%\n", m->neural_top5);
    fprintf(f, "  Mean Reciprocal Rank (MRR): %.4f\n", m->mrr);
    fprintf(f, "  Avg Candidates / Known Pair: %.2f\n", m->avg_candidates);
    fclose(f);
}

static void write_metrics_csv(const BaselineMetrics *m) {
    mkdir("results", 0755);
    FILE *f = fopen("results/metrics.csv", "w");
    if (!f) return;

    fprintf(f, "dataset_path,training_triangles,heldout_triangles,vocab_size,training_vocab,embed_dim,hidden_dim,negative_samples,epochs,seed,gpu_count,evaluation_mode,exact_recall,neural_top1,neural_top5,mrr,avg_candidates\n");
    fprintf(f, "\"%s\",%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,\"%s\",%.2f,%.2f,%.2f,%.4f,%.2f\n",
            m->dataset_path ? m->dataset_path : "",
            m->training_triangles,
            m->heldout_triangles,
            m->vocab_size,
            m->training_vocab,
            m->embed_dim,
            m->hidden_dim,
            m->negative_samples,
            m->epochs,
            m->seed,
            m->gpu_count,
            m->evaluation_mode ? m->evaluation_mode : "EXACT",
            m->exact_recall,
            m->neural_top1,
            m->neural_top5,
            m->mrr,
            m->avg_candidates);
    fclose(f);
}


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
    printf("      --cand-mode exact|struct|union --pos-thresh <x> --trans-thresh <n> --seed <n>\n");
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

/* Role checker: resolve the UPOS role the chain associates with a word id.
   Returns the most frequently observed NONZERO role for that word across the
   chain's triangles, or 0 when the word never carried a tag.

   Why this exists: training reads role_ids from the annotated chain, so any
   inference path that invents a role (or silently passes 0) feeds the network
   a different input distribution than it trained on. CLI paths receive bare
   words with no annotation, so they look up what the chain already knows and
   leave the rest in holding (0) rather than guessing.
   Limitation: this is a frequency lookup over observed tags, not a predicted
   tag; an unknown word stays 0. It does not account for the word's current
   context. */
static int lookup_word_role(const TriangleChain *chain, int word_id) {
    if (!chain || word_id <= 0) return 0;
    int counts[TRIANGLE_ROLE_FEATURE_DIM];
    for (int r = 0; r < TRIANGLE_ROLE_FEATURE_DIM; r++) counts[r] = 0;
    for (size_t t = 0; t < chain->count; t++) {
        for (int p = 0; p < 3; p++) {
            if (chain->triangles[t].word_ids[p] != word_id) continue;
            int role = chain->triangles[t].role_ids[p];
            if (role > 0 && role < TRIANGLE_ROLE_FEATURE_DIM) counts[role]++;
        }
    }
    int best = 0;
    int best_count = 0;
    for (int r = 1; r < TRIANGLE_ROLE_FEATURE_DIM; r++) {
        if (counts[r] > best_count) {
            best_count = counts[r];
            best = r;
        }
    }
    return best;
}

static int has_conllu_suffix(const char *filename) {
    size_t length = strlen(filename);
    return length >= 7 && strcmp(filename + length - 7, ".conllu") == 0;
}

static uint64_t transition_hash(int from_id, int to_id, int transition_type) {
    uint64_t x = (uint64_t)(unsigned int)from_id;
    x = x * 1000003ULL ^ (uint64_t)(unsigned int)to_id;
    x = x * 1000033ULL ^ (uint64_t)(unsigned int)transition_type;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static TransitionLookup *transition_lookup_create(const RelationalRegistry *rel_reg) {
    if (!rel_reg) return NULL;

    size_t capacity = 1;
    size_t needed = rel_reg->transition_count * 2 + 1;
    while (capacity < needed) capacity <<= 1;

    TransitionLookup *lookup = calloc(1, sizeof(*lookup));
    if (!lookup) return NULL;
    lookup->entries = calloc(capacity, sizeof(*lookup->entries));
    if (!lookup->entries) {
        free(lookup);
        return NULL;
    }
    lookup->capacity = capacity;

    for (size_t i = 0; i < rel_reg->transition_count; i++) {
        const RelationalTransition *tr = &rel_reg->transitions[i];
        size_t mask = lookup->capacity - 1;
        size_t slot = (size_t)transition_hash(
            tr->from_word_id, tr->to_word_id, tr->transition_type) & mask;
        while (lookup->entries[slot].used) {
            slot = (slot + 1) & mask;
        }
        lookup->entries[slot] = (TransitionLookupEntry){
            tr->from_word_id,
            tr->to_word_id,
            tr->transition_type,
            tr->count,
            1
        };
    }

    return lookup;
}

static void transition_lookup_free(TransitionLookup *lookup) {
    if (!lookup) return;
    free(lookup->entries);
    free(lookup);
}

static uint64_t transition_lookup_get(const TransitionLookup *lookup,
                                      int from_id,
                                      int to_id,
                                      int transition_type) {
    if (!lookup || !lookup->entries || lookup->capacity == 0) return 0;

    size_t mask = lookup->capacity - 1;
    size_t slot = (size_t)transition_hash(from_id, to_id, transition_type) & mask;
    while (lookup->entries[slot].used) {
        const TransitionLookupEntry *entry = &lookup->entries[slot];
        if (entry->from_word_id == from_id &&
            entry->to_word_id == to_id &&
            entry->transition_type == transition_type) {
            return entry->count;
        }
        slot = (slot + 1) & mask;
    }
    return 0;
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

static uint64_t threshold_transition_count_lookup(const RelationalRegistry *rel_reg,
                                                  const TransitionLookup *lookup,
                                                  const ThresholdSweepQuery *query,
                                                  int candidate_id) {
    if (!lookup) return threshold_transition_count(rel_reg, query, candidate_id);

    int target_position = query->target_position;
    uint64_t forward = 0;
    uint64_t backward = 0;
    if (target_position == 2) {
        forward = transition_lookup_get(lookup, query->second_id, candidate_id, 1);
        backward = transition_lookup_get(lookup, candidate_id, query->second_id, 2);
    } else if (target_position == 0) {
        forward = transition_lookup_get(lookup, candidate_id, query->first_id, 0);
        backward = transition_lookup_get(lookup, query->first_id, candidate_id, 3);
    } else {
        forward = transition_lookup_get(lookup, query->second_id, candidate_id, 0);
        backward = transition_lookup_get(lookup, candidate_id, query->second_id, 3);
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

static size_t collect_structural_vocab_candidates(const RelationalRegistry *rel_reg,
                                                  const TransitionLookup *lookup,
                                                  int first_id,
                                                  int second_id,
                                                  int target_position,
                                                  double position_threshold,
                                                  uint64_t transition_threshold,
                                                  ContextCandidate *candidates,
                                                  size_t max_candidates,
                                                  int gold_target_id,
                                                  int *out_gold_eligible,
                                                  int *out_gold_past_cap) {
    if (!rel_reg || !candidates || max_candidates == 0) return 0;
    if (out_gold_eligible) *out_gold_eligible = 0;
    if (out_gold_past_cap) *out_gold_past_cap = 0;

    size_t count = 0;
    for (size_t word = 1; word <= rel_reg->vocab_size; word++) {
        const RelationalWordStats *stats = &rel_reg->word_stats[word];
        double total = (double)(stats->left_count +
                                stats->center_count +
                                stats->right_count);
        if (total <= 0.0) continue;

        uint64_t position_count = target_position == 0 ? stats->left_count :
                                  target_position == 1 ? stats->center_count :
                                                         stats->right_count;
        double position_ratio = (double)position_count / total;
        if (position_ratio < position_threshold) continue;

        ThresholdSweepQuery query = {
            first_id, second_id, (int)word, target_position
        };
        uint64_t transition_count = threshold_transition_count_lookup(
            rel_reg, lookup, &query, (int)word);
        if (transition_count < transition_threshold) continue;

        /* Eligibility diagnostic: gold passed both thresholds even if the
           candidate buffer later truncates it. Distinguishes "generator
           cannot identify gold" from "generator found too many options
           and the cap discarded gold". */
        if (gold_target_id > 0 && (int)word == gold_target_id) {
            if (out_gold_eligible) *out_gold_eligible = 1;
            if (count >= max_candidates && out_gold_past_cap)
                *out_gold_past_cap = 1;
        }

        if (count >= max_candidates) break;

        double transition_strength = log1p((double)transition_count);
        double transition_compat = transition_strength / (1.0 + transition_strength);
        double relational_compat = position_ratio * (0.5 + 0.5 * transition_compat);

        candidates[count++] = (ContextCandidate){
            (int)word,
            (int)position_count,
            transition_count > (uint64_t)INT_MAX ? INT_MAX : (int)transition_count,
            relational_compat
        };
        if (count == max_candidates) break;
    }
    return count;
}

static size_t merge_candidate_sets(ContextCandidate *base,
                                   size_t base_count,
                                   const ContextCandidate *extra,
                                   size_t extra_count,
                                   size_t max_candidates) {
    size_t count = base_count;
    for (size_t i = 0; i < extra_count; i++) {
        int duplicate = 0;
        for (size_t j = 0; j < count; j++) {
            if (base[j].word_id != extra[i].word_id) continue;
            if (extra[i].occurrence_count > base[j].occurrence_count)
                base[j].occurrence_count = extra[i].occurrence_count;
            if (extra[i].neighbor_count > base[j].neighbor_count)
                base[j].neighbor_count = extra[i].neighbor_count;
            if (extra[i].match_score > base[j].match_score)
                base[j].match_score = extra[i].match_score;
            duplicate = 1;
            break;
        }
        if (!duplicate && count < max_candidates)
            base[count++] = extra[i];
    }
    return count;
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
            fill_model_input(input, 0, first_id,
                             chain->triangles[t].role_ids[rotation],
                             trainer->network);
            fill_model_input(input, 1, second_id,
                             chain->triangles[t].role_ids[(rotation + 1) % 3],
                             trainer->network);
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

static size_t collect_unified_candidates(const ContextGraph *graph,
                                         const RelationalRegistry *rel_reg,
                                         const TransitionLookup *lookup,
                                         const EvaluationQuery *query,
                                         int mode,
                                         double position_threshold,
                                         uint64_t transition_threshold,
                                         ContextCandidate *candidates,
                                         size_t max_candidates,
                                         int *out_struct_gold_eligible,
                                         int *out_struct_gold_past_cap) {
    if (!graph || !query || !candidates || max_candidates == 0) return 0;

    if (mode == 0) {
        return context_graph_collect_candidate_evidence_fallback(
            graph, query->first_id, 0, query->second_id, 0,
            query->rotation, candidates, max_candidates);
    }

    if (mode == 1) {
        return collect_structural_vocab_candidates(
            rel_reg, lookup, query->first_id, query->second_id,
            query->target_position, position_threshold,
            transition_threshold, candidates, max_candidates,
            query->target_id, out_struct_gold_eligible,
            out_struct_gold_past_cap);
    }

    ContextCandidate structural[1024];
    size_t exact_count = context_graph_collect_candidate_evidence_fallback(
        graph, query->first_id, 0, query->second_id, 0,
        query->rotation, candidates, max_candidates);
    size_t structural_count = collect_structural_vocab_candidates(
        rel_reg, lookup, query->first_id, query->second_id,
        query->target_position, position_threshold,
        transition_threshold, structural, 1024,
        query->target_id, out_struct_gold_eligible,
        out_struct_gold_past_cap);
    return merge_candidate_sets(candidates, exact_count, structural,
                                structural_count, max_candidates);
}

static void write_unified_eval_csv(const UnifiedEvalMetrics *metrics,
                                   size_t mode_count) {
    mkdir("results", 0755);
    FILE *f = fopen("results/unified_eval.csv", "w");
    if (!f) return;
    fprintf(f, "mode,queries,candidate_recall,top1,top5,mrr,avg_candidates,nonempty_queries,max_candidates,median_candidates,cap_hit_queries,cap_hit_rate\n");
    for (size_t i = 0; i < mode_count; i++) {
        const UnifiedEvalMetrics *m = &metrics[i];
        fprintf(f, "%s,%d,%.4f,%.4f,%.4f,%.6f,%.2f,%d,%d,%.2f,%d,%.4f\n",
                m->name,
                m->query_count,
                m->query_count > 0 ? (double)m->candidate_hits / m->query_count : 0.0,
                m->query_count > 0 ? (double)m->top1 / m->query_count : 0.0,
                m->query_count > 0 ? (double)m->top5 / m->query_count : 0.0,
                m->query_count > 0 ? m->mrr_sum / m->query_count : 0.0,
                m->query_count > 0 ? (double)m->total_candidates / m->query_count : 0.0,
                m->nonempty_queries,
                m->max_candidates,
                m->median_candidates,
                m->cap_hit_queries,
                m->query_count > 0 ? (double)m->cap_hit_queries / m->query_count : 0.0);
    }
    fclose(f);
}

/* Per-query artifact so partitions can be re-sliced without re-training. */
static void write_unified_query_csv(const UnifiedQueryRecord *records,
                                    size_t query_count,
                                    double position_threshold,
                                    uint64_t transition_threshold) {
    mkdir("results", 0755);
    FILE *f = fopen("results/unified_eval_queries.csv", "w");
    if (!f) {
        fprintf(stderr, "[main] could not open results/unified_eval_queries.csv\n");
        return;
    }
    fprintf(f, "first_id,second_id,rotation,target_id,pos_threshold,trans_threshold,"
               "exact_count,struct_count,union_count,gold_in_exact,gold_in_struct,"
               "gold_in_union,exact_rank,struct_rank,union_rank,"
               "struct_gold_eligible,struct_gold_past_cap\n");
    for (size_t q = 0; q < query_count; q++) {
        const UnifiedQueryRecord *r = &records[q];
        fprintf(f, "%d,%d,%d,%d,%.2f,%llu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                r->first_id, r->second_id, r->rotation, r->target_id,
                position_threshold, (unsigned long long)transition_threshold,
                r->count[0], r->count[1], r->count[2],
                r->gold_in[0], r->gold_in[1], r->gold_in[2],
                r->rank[0], r->rank[1], r->rank[2],
                r->struct_gold_eligible, r->struct_gold_past_cap);
    }
    fclose(f);
}

static void run_unified_candidate_evaluation(const BackpropTrainer *trainer,
                                             const TriangleChain *chain,
                                             const ContextGraph *graph,
                                             const RelationalRegistry *rel_reg,
                                             size_t limit,
                                             double position_threshold,
                                             uint64_t transition_threshold) {
    if (!trainer || !trainer->network || !chain || !graph || !rel_reg) return;
    if (limit > chain->count) limit = chain->count;

    size_t query_capacity = limit * 3;
    EvaluationQuery *queries = calloc(query_capacity ? query_capacity : 1,
                                      sizeof(*queries));
    if (!queries) return;

    size_t query_count = 0;
    int vocab_size = trainer->network->vocab_size;
    for (size_t t = 0; t < limit; t++) {
        for (int rotation = 0; rotation < 3; rotation++) {
            int first_id = chain->triangles[t].word_ids[rotation];
            int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
            int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];
            if (first_id <= 0 || second_id <= 0 || target_id <= 0 ||
                first_id > vocab_size || second_id > vocab_size ||
                target_id > vocab_size)
                continue;
            queries[query_count++] = (EvaluationQuery){
                first_id, second_id, target_id, rotation, (rotation + 2) % 3,
                chain->triangles[t].role_ids[rotation],
                chain->triangles[t].role_ids[(rotation + 1) % 3]
            };
        }
    }

    UnifiedEvalMetrics metrics[3] = {
        {"EXACT", (int)query_count, 0, 0, 0, 0.0, 0, 0, 0, 0, 0.0, NULL},
        {"STRUCT", (int)query_count, 0, 0, 0, 0.0, 0, 0, 0, 0, 0.0, NULL},
        {"UNION", (int)query_count, 0, 0, 0, 0.0, 0, 0, 0, 0, 0.0, NULL}
    };
    for (int mode = 0; mode < 3; mode++) {
        metrics[mode].cand_counts = calloc(query_count ? query_count : 1,
                                           sizeof(size_t));
        if (!metrics[mode].cand_counts) {
            fprintf(stderr, "[main] unified eval: count buffer alloc failed\n");
            for (int m2 = 0; m2 < mode; m2++) free(metrics[m2].cand_counts);
            free(queries);
            return;
        }
    }

    UnifiedQueryRecord *records = calloc(query_count ? query_count : 1,
                                         sizeof(*records));
    BucketStats buckets[4];
    memset(buckets, 0, sizeof(buckets));
    int mv_total = 0, mv_from_exact = 0, mv_improved = 0, mv_unchanged = 0;
    int mv_worsened = 0, mv_out_top5 = 0, mv_lost_rank1 = 0;
    int elig_gold_eligible = 0, elig_gold_retained = 0, elig_gold_past_cap = 0;

    double *input = calloc(trainer->network->input_size, sizeof(double));
    double *output = calloc(trainer->network->output_size, sizeof(double));
    if (!input || !output || !records) {
        fprintf(stderr, "[main] unified eval: alloc failed (records/input/output)\n");
        free(records);
        free(input);
        free(output);
        free(queries);
        return;
    }

    const int max_candidates = 1024;
    int embed_dim = trainer->network->embed_dim;
    size_t role_known_queries = 0;
    size_t role_holding_queries = 0;
    BackpropTrainer *mutable_trainer = (BackpropTrainer *)trainer;
    TransitionLookup *lookup = transition_lookup_create(rel_reg);

    /* SCORER_DIAG=1: attribution instrumentation only. Scoring itself is
       untouched; default output is byte-identical when the flag is unset. */
    const char *sd_env = getenv("SCORER_DIAG");
    int scorer_diag = (sd_env && strcmp(sd_env, "1") == 0);
    int sd_hits = 0;
    int sd_hist[4] = {0, 0, 0, 0}; /* rank 1 / 2-5 / 6-20 / >20 */
    long sd_attr_dist = 0, sd_attr_bonus = 0, sd_attr_match = 0;
    int sd_worsened = 0;
    long sd_un_noise_exact = 0, sd_un_noise_struct = 0;
    FILE *sd_csv = NULL;
    if (scorer_diag) {
        mkdir("results", 0755);
        sd_csv = fopen("results/scorer_diag.csv", "w");
        if (sd_csv)
            fprintf(sd_csv, "first_id,second_id,rotation,target_id,pos_threshold,"
                            "trans_threshold,gold_rank,gold_dist,gold_bonus,"
                            "gold_match,n_outrankers,n_out_dist,n_out_bonus,n_out_match\n");
        else
            fprintf(stderr, "[main] SCORER_DIAG: could not open results/scorer_diag.csv\n");
    }

    fprintf(stderr,
            "[main] unified evaluation starting: %zu queries, %zu transitions\n",
            query_count, rel_reg->transition_count);
    fflush(stderr);

    for (size_t q = 0; q < query_count; q++) {
        if (q == 0 || q % 25 == 0) {
            fprintf(stderr, "[main] unified evaluation progress: %zu / %zu queries\n",
                    q, query_count);
            fflush(stderr);
        }

        const EvaluationQuery *query = &queries[q];
        if (query->first_role > 0 && query->second_role > 0) role_known_queries++;
        else role_holding_queries++;
        memset(input, 0, trainer->network->input_size * sizeof(double));
        /* Feed the SAME role one-hots the network saw during training.
           Hardcoding 0 here was a train/eval input mismatch: the model was
           trained with UPOS role features active, but evaluated with the
           whole 64-wide role block zeroed. Roles come from the held-out
           chain (gold UPOS from the annotated file; 0 = holding/unknown). */
        fill_model_input(input, 0, query->first_id, query->first_role,
                         trainer->network);
        fill_model_input(input, 1, query->second_id, query->second_role,
                         trainer->network);
        backprop_predict(mutable_trainer, input, output);

        UnifiedQueryRecord *rec = &records[q];
        rec->first_id = query->first_id;
        rec->second_id = query->second_id;
        rec->rotation = query->rotation;
        rec->target_id = query->target_id;
        int struct_gold_eligible = 0;
        int struct_gold_past_cap = 0;

        for (int mode = 0; mode < 3; mode++) {
            ContextCandidate candidates[1024];
            size_t count = collect_unified_candidates(
                graph, rel_reg, lookup, query, mode, position_threshold,
                transition_threshold, candidates, max_candidates,
                &struct_gold_eligible, &struct_gold_past_cap);
            rec->count[mode] = (int)count;
            rec->gold_in[mode] = 0;
            rec->rank[mode] = -1;

            metrics[mode].total_candidates += count;
            if (count > 0) metrics[mode].nonempty_queries++;
            if ((int)count > metrics[mode].max_candidates)
                metrics[mode].max_candidates = (int)count;
            if (metrics[mode].cand_counts && q < (size_t)metrics[mode].query_count)
                metrics[mode].cand_counts[q] = count;
            if ((int)count >= max_candidates) metrics[mode].cap_hit_queries++;

            double target_score = -INFINITY;
            int found = 0;
            double gold_dist = 0.0, gold_bonus = 0.0, gold_match = 0.0;
            double cand_dist[1024];
            double cand_match0[1024];
            for (size_t c = 0; c < count; c++) {
                int candidate_id = candidates[c].word_id;
                cand_dist[c] = 0.0;
                /* keep the ORIGINAL match_score: it is overwritten with the
                   full score below, but the attribution needs the raw term */
                cand_match0[c] = candidates[c].match_score;
                if (candidate_id <= 0 || candidate_id > vocab_size) continue;

                double distance = 0.0;
                int target_start = query->target_position * embed_dim;
                for (int d = 0; d < embed_dim; d++) {
                    double delta = output[target_start + d] -
                        trainer->network->embeddings[(candidate_id - 1) * embed_dim + d];
                    distance += delta * delta;
                }
                cand_dist[c] = distance;

                double score = -distance +
                    0.75 * log(1.0 + candidates[c].occurrence_count) +
                    0.50 * log(1.0 + candidates[c].neighbor_count) +
                    candidates[c].match_score;
                candidates[c].match_score = score;

                if (candidate_id == query->target_id) {
                    found = 1;
                    target_score = score;
                    gold_dist = distance;
                    gold_bonus = 0.75 * log(1.0 + candidates[c].occurrence_count) +
                                 0.50 * log(1.0 + candidates[c].neighbor_count);
                    gold_match = cand_match0[c];
                }
            }

            rec->gold_in[mode] = found ? 1 : 0;
            int rank = -1;
            if (found) {
                rank = 0;
                for (size_t c = 0; c < count; c++) {
                    int candidate_id = candidates[c].word_id;
                    if (candidate_id <= 0 || candidate_id > vocab_size) continue;
                    if (candidates[c].match_score > target_score) rank++;
                }
                rec->rank[mode] = rank;

                if (scorer_diag) {
                    if (mode == 1) {
                        if (rank == 0) sd_hist[0]++;
                        else if (rank < 5) sd_hist[1]++;
                        else if (rank < 20) sd_hist[2]++;
                        else sd_hist[3]++;

                        int n_out = 0, n_dist = 0, n_bonus = 0, n_match = 0;
                        for (size_t c = 0; c < count; c++) {
                            int candidate_id = candidates[c].word_id;
                            if (candidate_id <= 0 || candidate_id > vocab_size) continue;
                            if (candidates[c].match_score <= target_score) continue;
                            /* Lexicographic first-flip attribution: attribute the
                               win to the FIRST score component in the chain
                               distance -> bonus -> match whose partial score
                               already beats gold. NOTE: this does not model
                               interactions between components; it classifies the
                               minimal sufficient component. */
                            double s1c = -cand_dist[c], s1g = -gold_dist;
                            n_out++;
                            if (s1c > s1g) {
                                n_dist++;
                                sd_attr_dist++;
                            } else {
                                double s2c = s1c +
                                    0.75 * log(1.0 + candidates[c].occurrence_count) +
                                    0.50 * log(1.0 + candidates[c].neighbor_count);
                                double s2g = s1g + gold_bonus;
                                if (s2c > s2g) {
                                    n_bonus++;
                                    sd_attr_bonus++;
                                } else {
                                    n_match++;
                                    sd_attr_match++;
                                }
                            }
                        }
                        sd_hits++;
                        if (sd_csv)
                            fprintf(sd_csv, "%d,%d,%d,%d,%.2f,%llu,%d,%.6f,%.6f,%.6f,%d,%d,%d,%d\n",
                                    query->first_id, query->second_id, query->rotation,
                                    query->target_id, position_threshold,
                                    (unsigned long long)transition_threshold,
                                    rank, gold_dist, gold_bonus, gold_match,
                                    n_out, n_dist, n_bonus, n_match);
                    } else if (mode == 2 && rec->rank[1] >= 0 &&
                               rank > rec->rank[1]) {
                        /* Worsened by UNION: are the new outrankers of gold
                           EXACT-sourced or STRUCT-sourced? merge_candidate_sets
                           preserves the base (EXACT) entries at indices
                           [0, exact_n) and appends STRUCT entries after. */
                        ContextCandidate scratch[1024];
                        size_t exact_n = context_graph_collect_candidate_evidence_fallback(
                            graph, query->first_id, 0, query->second_id, 0,
                            query->rotation, scratch, (size_t)max_candidates);
                        for (size_t c = 0; c < count; c++) {
                            if (candidates[c].match_score <= target_score) continue;
                            if ((int)c < (int)exact_n) sd_un_noise_exact++;
                            else sd_un_noise_struct++;
                        }
                        sd_worsened++;
                    }
                }
            }
            if (!found) continue;
            metrics[mode].candidate_hits++;
            if (rank == 0) metrics[mode].top1++;
            if (rank < 5) metrics[mode].top5++;
            metrics[mode].mrr_sum += 1.0 / (rank + 1);
        }

        /* Partition by where gold lives (EXACT vs STRUCT coverage):
           A = gold only in EXACT, B = gold only in STRUCT,
           C = gold in both, D = gold in neither. */
        int bucket;
        if (rec->gold_in[0] && rec->gold_in[1]) bucket = 2;
        else if (rec->gold_in[0]) bucket = 0;
        else if (rec->gold_in[1]) bucket = 1;
        else bucket = 3;
        buckets[bucket].queries++;
        for (int mode = 0; mode < 3; mode++) {
            BucketModeStats *bs = &buckets[bucket].mode[mode];
            if (!rec->gold_in[mode]) continue;
            bs->found++;
            if (rec->rank[mode] == 0) bs->top1++;
            if (rec->rank[mode] >= 0 && rec->rank[mode] < 5) bs->top5++;
            if (rec->rank[mode] >= 0)
                bs->mrr_sum += 1.0 / (rec->rank[mode] + 1);
        }

        /* Rank movement STRUCT -> UNION (answers: does adding exact
           candidates push gold down the ranking?) */
        if (rec->gold_in[2]) {
            mv_total++;
            if (!rec->gold_in[1]) {
                mv_from_exact++;
            } else if (rec->rank[2] < rec->rank[1]) {
                mv_improved++;
            } else if (rec->rank[2] == rec->rank[1]) {
                mv_unchanged++;
            } else {
                mv_worsened++;
                if (rec->rank[1] < 5 && rec->rank[2] >= 5) mv_out_top5++;
                if (rec->rank[1] == 0 && rec->rank[2] > 0) mv_lost_rank1++;
            }
        }

        /* STRUCT eligibility vs retention (censoring audit) */
        if (struct_gold_eligible) elig_gold_eligible++;
        if (rec->gold_in[1]) elig_gold_retained++;
        if (struct_gold_past_cap) elig_gold_past_cap++;
    }

    fprintf(stderr, "[main] unified evaluation progress: %zu / %zu queries\n",
            query_count, query_count);
    fflush(stderr);

    for (int mode = 0; mode < 3; mode++) {
        UnifiedEvalMetrics *m = &metrics[mode];
        if (!m->cand_counts || query_count == 0) {
            m->median_candidates = 0.0;
            continue;
        }
        size_t *sorted = malloc(query_count * sizeof(*sorted));
        if (!sorted) {
            fprintf(stderr, "[main] median computation unavailable for %s (OOM)\n",
                    m->name);
            m->median_candidates = 0.0;
            continue;
        }
        memcpy(sorted, m->cand_counts, query_count * sizeof(*sorted));
        qsort(sorted, query_count, sizeof(*sorted), cmp_size_t);
        m->median_candidates = (query_count % 2)
            ? (double)sorted[query_count / 2]
            : 0.5 * ((double)sorted[query_count / 2 - 1] +
                     (double)sorted[query_count / 2]);
        free(sorted);
    }

    printf("\n=== Unified Held-out Candidate Evaluation (%zu shared queries) ===\n",
           query_count);
    printf("Thresholds: pos >= %.2f | transition >= %llu\n",
           position_threshold, (unsigned long long)transition_threshold);
    printf("Eval roles: %zu queries with gold UPOS roles, %zu holding (role 0)\n",
           role_known_queries, role_holding_queries);
    printf("%-8s %-9s %-12s %-8s %-8s %-8s %-12s %-10s %-10s %-10s\n",
           "Mode", "Queries", "CandRecall", "R@1", "R@5", "MRR",
           "AvgCand", "MaxCand", "MedianCand", "CapHit");
    printf("--------------------------------------------------------------------------------------\n");
    for (int mode = 0; mode < 3; mode++) {
        UnifiedEvalMetrics *m = &metrics[mode];
        printf("%-8s %-9d %-11.1f%% %-7.1f%% %-7.1f%% %-8.4f %-12.2f %-10d %-10.1f %d/%.1f%%\n",
               m->name,
               m->query_count,
               m->query_count > 0 ? 100.0 * m->candidate_hits / m->query_count : 0.0,
               m->query_count > 0 ? 100.0 * m->top1 / m->query_count : 0.0,
               m->query_count > 0 ? 100.0 * m->top5 / m->query_count : 0.0,
               m->query_count > 0 ? m->mrr_sum / m->query_count : 0.0,
               m->query_count > 0 ? (double)m->total_candidates / m->query_count : 0.0,
               m->max_candidates,
               m->median_candidates,
               m->cap_hit_queries,
               m->query_count > 0 ? 100.0 * m->cap_hit_queries / m->query_count : 0.0);
    }
    printf("==========================================================================\n");

    static const char *bucket_names[4] = {
        "A: EXACT-only", "B: STRUCT-only", "C: EXACT+STRUCT", "D: neither"
    };
    printf("\n=== Per-Query Gold Composition (%zu queries, pos>=%.2f trans>=%llu) ===\n",
           query_count, position_threshold, (unsigned long long)transition_threshold);
    printf("%-16s %-8s %-22s %-22s %-22s\n",
           "Bucket", "Queries", "EXACT h/t1/t5/mrr", "STRUCT h/t1/t5/mrr",
           "UNION h/t1/t5/mrr");
    printf("----------------------------------------------------------------------------------------------\n");
    for (int b = 0; b < 4; b++) {
        printf("%-16s %-8d", bucket_names[b], buckets[b].queries);
        for (int mode = 0; mode < 3; mode++) {
            BucketModeStats *bs = &buckets[b].mode[mode];
            int n = buckets[b].queries;
            printf(" %d/%.1f%%/%.1f%%/%.4f",
                   bs->found,
                   n > 0 ? 100.0 * bs->top1 / n : 0.0,
                   n > 0 ? 100.0 * bs->top5 / n : 0.0,
                   n > 0 ? bs->mrr_sum / n : 0.0);
        }
        printf("\n");
    }
    printf("  (h = gold present in that mode's candidates; rates over bucket size)\n");

    printf("\n=== Rank Movement: STRUCT -> UNION (gold in UNION: %d) ===\n", mv_total);
    printf("  STRUCT missed gold, added via EXACT side : %d\n", mv_from_exact);
    printf("  rank improved                            : %d\n", mv_improved);
    printf("  rank unchanged                           : %d\n", mv_unchanged);
    printf("  rank worsened                            : %d (out of top-5: %d, lost rank-1: %d)\n",
           mv_worsened, mv_out_top5, mv_lost_rank1);

    printf("\n=== STRUCT Eligibility vs Retention (censoring audit) ===\n");
    printf("  gold eligible under thresholds (incl. truncated): %d\n",
           elig_gold_eligible);
    printf("  gold retained in STRUCT candidates              : %d\n",
           elig_gold_retained);
    printf("  gold eligible but DROPPED by %d cap             : %d\n",
           max_candidates, elig_gold_past_cap);
    printf("  gold not eligible under thresholds              : %d\n",
           (int)query_count - elig_gold_eligible);
    printf("==========================================================================\n");
    fflush(stdout);

    if (scorer_diag) {
        printf("\n=== Scorer Failure Attribution (SCORER_DIAG, %d STRUCT hits) ===\n",
               sd_hits);
        printf("Gold rank among STRUCT candidates:\n");
        printf("  rank 1    : %d\n", sd_hist[0]);
        printf("  rank 2-5  : %d\n", sd_hist[1]);
        printf("  rank 6-20 : %d\n", sd_hist[2]);
        printf("  rank >20  : %d\n", sd_hist[3]);
        printf("Outranker attribution (first component whose partial score beats gold):\n");
        printf("  distance alone : %ld\n", sd_attr_dist);
        printf("  occ/neigh bonus: %ld\n", sd_attr_bonus);
        printf("  match_score    : %ld\n", sd_attr_match);
        printf("UNION noise attribution (%d worsened queries):\n", sd_worsened);
        printf("  outrankers of gold from EXACT side  : %ld\n", sd_un_noise_exact);
        printf("  outrankers of gold from STRUCT side : %ld\n", sd_un_noise_struct);
        printf("==========================================================================\n");
        fflush(stdout);
        if (sd_csv) fclose(sd_csv);
    }

    write_unified_eval_csv(metrics, 3);
    write_unified_query_csv(records, query_count, position_threshold,
                            transition_threshold);
    for (int mode = 0; mode < 3; mode++) free(metrics[mode].cand_counts);
    free(records);
    transition_lookup_free(lookup);
    free(input);
    free(output);
    free(queries);
}

static void evaluate_context_model(BackpropTrainer *trainer,
                                   const TriangleChain *chain,
                                   const ContextGraph *graph,
                                   const RelationalRegistry *rel_reg,
                                   int candidate_mode,
                                   size_t limit,
                                   double position_threshold,
                                   uint64_t transition_threshold,
                                   BaselineMetrics *out_metrics) {
    if (!graph) {
        printf("Unable to create evaluation context graph.\n");
        return;
    }
    if (limit > chain->count) limit = chain->count;
    int graph_hits = 0, neural_top1 = 0, neural_top5 = 0, total = 0;
    int pair_queries = 0, ambiguous_pairs = 0, max_candidates = 0;
    size_t candidate_total = 0;
    double mrr_sum = 0.0;

    int candidate_ids[256];
    int embed_dim = trainer->network->embed_dim;
    double *input = calloc(trainer->network->input_size, sizeof(double));
    double *output = calloc(trainer->network->output_size, sizeof(double));

    for (size_t t = 0; t < limit; t++) {
        for (int rotation = 0; rotation < 3; rotation++) {
            int first_id = chain->triangles[t].word_ids[rotation];
            int second_id = chain->triangles[t].word_ids[(rotation + 1) % 3];
            int target_id = chain->triangles[t].word_ids[(rotation + 2) % 3];
            /* Roles always on at eval: matches the training input
               distribution (training consumed UPOS roles from the chain). */
            int first_role = chain->triangles[t].role_ids[rotation];
            int second_role = chain->triangles[t].role_ids[(rotation + 1) % 3];

            ContextCandidate evidence[256];
            size_t count = 0;
            if (candidate_mode == 1) {
                count = context_graph_collect_candidate_evidence_relational_thresholded(
                    graph, rel_reg, first_id, second_id, rotation,
                    position_threshold, transition_threshold, evidence, 256);
            } else if (candidate_mode == 0) {
                count = context_graph_collect_candidate_evidence_fallback(
                    graph, first_id, first_role, second_id, second_role, rotation,
                    evidence, 256);
            } else {
                ContextCandidate structural[256];
                size_t exact_count = context_graph_collect_candidate_evidence_fallback(
                    graph, first_id, first_role, second_id, second_role, rotation,
                    evidence, 256);
                size_t structural_count = context_graph_collect_candidate_evidence_relational_thresholded(
                    graph, rel_reg, first_id, second_id, rotation,
                    position_threshold, transition_threshold, structural, 256);
                count = exact_count;
                for (size_t s = 0; s < structural_count && count < 256; s++) {
                    int duplicate = 0;
                    for (size_t e = 0; e < count; e++) {
                        if (evidence[e].word_id == structural[s].word_id) {
                            if (structural[s].match_score > evidence[e].match_score)
                                evidence[e].match_score = structural[s].match_score;
                            duplicate = 1;
                            break;
                        }
                    }
                    if (!duplicate) evidence[count++] = structural[s];
                }
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
                if (candidate_mode == 1) {
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
                mrr_sum += 1.0 / (rank + 1);
            }
        }
    }

    if (out_metrics) {
        out_metrics->evaluation_mode = candidate_mode == 1 ? "STRUCTURAL" :
                                       candidate_mode == 2 ? "UNION" : "EXACT";
        out_metrics->exact_recall = total > 0 ? 100.0 * graph_hits / total : 0.0;
        out_metrics->neural_top1 = total > 0 ? 100.0 * neural_top1 / total : 0.0;
        out_metrics->neural_top5 = total > 0 ? 100.0 * neural_top5 / total : 0.0;
        out_metrics->mrr = total > 0 ? mrr_sum / total : 0.0;
        out_metrics->avg_candidates = pair_queries > 0 ? (double)candidate_total / pair_queries : 0.0;
    }


    if (candidate_mode == 1 && rel_reg && pair_queries > 0) {
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

    // New command-line options
    int cand_mode = 2; // 0=exact,1=struct,2=union (default)
    double pos_thresh = 0.25;
    int trans_thresh = 3;
    unsigned int seed = 42;
    // Manual long option parsing
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--cand-mode") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "exact") == 0) cand_mode = 0;
            else if (strcmp(mode, "struct") == 0) cand_mode = 1;
            else if (strcmp(mode, "union") == 0) cand_mode = 2;
        } else if (strcmp(argv[i], "--pos-thresh") == 0 && i + 1 < argc) {
            pos_thresh = atof(argv[++i]);
        } else if (strcmp(argv[i], "--trans-thresh") == 0 && i + 1 < argc) {
            trans_thresh = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        }
    }
    srand(seed);

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
                /* CLI query words arrive without annotation; look up the role
                   the chain already knows for them (0 = holding) so this path
                   feeds the network the same role features training used. */
                int first_role = lookup_word_role(chain, first_id);
                int second_role = lookup_word_role(chain, second_id);
                fill_model_input(input, 0, first_id, first_role, trainer->network);
                fill_model_input(input, 1, second_id, second_role, trainer->network);
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
            RelationalRegistry *eval_reg = relational_registry_create(chain->vocab->count);
            if (eval_reg) relational_registry_ingest_chain(eval_reg, chain);
            BaselineMetrics metrics = {0};
            metrics.dataset_path = filename;
            metrics.training_triangles = (long)chain->count;
            metrics.heldout_triangles = 0;
            metrics.vocab_size = (int)chain->vocab->count;
            metrics.training_vocab = (int)chain->vocab->count;
            metrics.embed_dim = trainer->network->embed_dim;
            metrics.hidden_dim = trainer->network->hidden_size;

            metrics.negative_samples = 32;
            metrics.epochs = 50;
            metrics.seed = (int)seed;
            metrics.gpu_count = eval_cuda_device_count();

            if (cand_mode != 0 && eval_reg) {
                evaluate_context_model(trainer, chain, context_graph, eval_reg, cand_mode, limit,
                                       pos_thresh, (uint64_t)trans_thresh, &metrics);
            } else {
                evaluate_context_model(trainer, chain, context_graph, NULL, 0, limit,
                                       pos_thresh, (uint64_t)trans_thresh, &metrics);
            }
            write_baseline_report(&metrics);
            write_metrics_csv(&metrics);

            relational_registry_free(eval_reg);
            context_graph_free(context_graph);
            backprop_free(trainer);
        }

    } else if (argc > 3 && strcmp(argv[2], "-heldout") == 0) {
        char *test_content = read_file(argv[3]);
        TriangleChain *global_chain = has_conllu_suffix(filename)
            ? create_triangles_from_conllu(file_content)
            : create_triangles(file_content);
        if (!global_chain || !test_content) {
            fprintf(stderr, "Held-out experiment requires training data and a test file\n");
            free(test_content);
            free_triangles(global_chain);
        } else {


            Vocabulary *shared_vocab = global_chain->vocab;
            global_chain->vocab = NULL;
            global_chain->owns_vocab = 0;
            free_triangles(global_chain);

            /* Combine the primary training document with any additional
             * training documents listed after the held-out test file. */
            size_t train_size = strlen(file_content);
            int training_documents = 1;
            char *combined_train = malloc(train_size + 1);
            if (combined_train) {
                memcpy(combined_train, file_content, train_size);
                combined_train[train_size] = '\0';
            }
            for (int arg = 4; combined_train && arg < argc; arg++) {
                if (argv[arg][0] == '-') {
                    if (strcmp(argv[arg], "-mode") == 0 && arg + 1 < argc) arg++;
                    else if (strcmp(argv[arg], "--cand-mode") == 0 && arg + 1 < argc) arg++;
                    else if (strcmp(argv[arg], "--pos-thresh") == 0 && arg + 1 < argc) arg++;
                    else if (strcmp(argv[arg], "--trans-thresh") == 0 && arg + 1 < argc) arg++;
                    else if (strcmp(argv[arg], "--seed") == 0 && arg + 1 < argc) arg++;
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
                training_documents++;
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
                printf("Training documents: %d\n", training_documents);

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
                BackpropTrainer *trainer = backprop_create_seeded(
                    (int)shared_vocab->count, 32, 128, 96, training_epochs, 0.1, seed);
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
                        run_unified_candidate_evaluation(
                            trainer, test_chain, train_graph, rel_reg, 100,
                            pos_thresh, (uint64_t)trans_thresh);
                        const char *unified_sweep = getenv("UNIFIED_SWEEP");
                        if (unified_sweep && strcmp(unified_sweep, "1") == 0) {
                            static const struct { double pos; uint64_t trans; } grid[] = {
                                {0.20, 4}, {0.25, 4}, {0.30, 3}, {0.30, 4}, {0.35, 3}
                            };
                            printf("\n=== Unified Threshold Grid (UNIFIED_SWEEP) ===\n");
                            fflush(stdout);
                            for (size_t g = 0; g < sizeof(grid) / sizeof(grid[0]); g++) {
                                printf("\n--- grid pos=%.2f trans=%llu ---\n",
                                       grid[g].pos,
                                       (unsigned long long)grid[g].trans);
                                fflush(stdout);
                                run_unified_candidate_evaluation(
                                    trainer, test_chain, train_graph, rel_reg, 100,
                                    grid[g].pos, grid[g].trans);
                            }
                        }
                        if (run_gpu_evaluation_only(test_chain, train_graph, rel_reg, 100) != 0)
                            fprintf(stderr, "[main] GPU evaluation failed\n");
                        run_neural_evaluation_only(trainer, test_chain, 100);
                    } else {
                        BaselineMetrics metrics = {0};
                        metrics.dataset_path = filename;
                        metrics.training_triangles = (long)train_chain->count;
                        metrics.heldout_triangles = (long)test_chain->count;
                        metrics.vocab_size = (int)shared_vocab->count;
                        metrics.training_vocab = (int)shared_vocab->count;
                        metrics.embed_dim = trainer->network->embed_dim;
                        metrics.hidden_dim = trainer->network->hidden_size;

                        metrics.negative_samples = 32;
                        metrics.epochs = training_epochs;
                        metrics.seed = (int)seed;
                        metrics.gpu_count = eval_cuda_device_count();

                        run_unified_candidate_evaluation(
                            trainer, test_chain, train_graph, rel_reg, 100,
                            pos_thresh, (uint64_t)trans_thresh);

                        // Evaluate based on candidate mode selection
                        evaluate_context_model(trainer, test_chain, train_graph, rel_reg, cand_mode, 100,
                                               pos_thresh, (uint64_t)trans_thresh, &metrics);
                        write_baseline_report(&metrics);
                        write_metrics_csv(&metrics);
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
        BackpropTrainer *trainer = backprop_create_seeded(vocab_size, embed_dim, 128,
                                                          3 * embed_dim, 50, 0.1,
                                                          seed);
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
