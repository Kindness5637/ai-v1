#include "matcher.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static char **tokenize_input(const char *input, size_t *count) {
    char *copy = strdup(input);
    if (!copy) return NULL;

    size_t capacity = 16;
    char **words = malloc(capacity * sizeof(char *));
    if (!words) {
        free(copy);
        return NULL;
    }

    *count = 0;
    char *token = strtok(copy, " ,.!?;:\n\"—");
    while (token) {
        if (*count >= capacity) {
            capacity *= 2;
            char **temp = realloc(words, capacity * sizeof(char *));
            if (!temp) {
                for (size_t i = 0; i < *count; i++) free(words[i]);
                free(words);
                free(copy);
                return NULL;
            }
            words = temp;
        }
        char *lower = malloc(strlen(token) + 1);
        for (size_t i = 0; token[i]; i++) {
            lower[i] = tolower((unsigned char)token[i]);
        }
        lower[strlen(token)] = '\0';
        words[*count] = lower;
        (*count)++;
        token = strtok(NULL, " ,.!?;:\n\"—");
    }

    free(copy);
    return words;
}

static int find_word_id(const TriangleChain *chain, const char *word) {
    for (size_t i = 0; i < chain->vocab->count; i++) {
        if (strcmp(chain->vocab->words[i].text, word) == 0) {
            return chain->vocab->words[i].id;
        }
    }
    return -1;
}

static int check_sequential_match(const TriangleChain *chain, size_t start_tri, int *input_ids, size_t input_count) {
    int matched = 0;
    size_t tri_idx = start_tri;
    size_t input_idx = 0;

    while (tri_idx < chain->count && input_idx < input_count) {
        int found_in_tri = 0;
        for (int p = 0; p < 3; p++) {
            if (chain->triangles[tri_idx].word_ids[p] == input_ids[input_idx]) {
                found_in_tri = 1;
                break;
            }
        }

        if (found_in_tri) {
            matched++;
            input_idx++;
        }

        tri_idx++;
        if (tri_idx - start_tri > 10 && matched == 0) break;
    }

    return matched;
}

MatcherOutput *matcher_find(const TriangleChain *chain, const ProbabilityMatrix *prob, const char *input) {
    if (!chain || !prob || !input) return NULL;

    size_t input_count = 0;
    char **input_words = tokenize_input(input, &input_count);
    if (!input_words || input_count == 0) return NULL;

    int *input_ids = malloc(input_count * sizeof(int));
    if (!input_ids) {
        for (size_t i = 0; i < input_count; i++) free(input_words[i]);
        free(input_words);
        return NULL;
    }

    size_t valid_count = 0;
    for (size_t i = 0; i < input_count; i++) {
        int id = find_word_id(chain, input_words[i]);
        if (id > 0) {
            input_ids[valid_count] = id;
            valid_count++;
        }
    }

    MatcherOutput *output = malloc(sizeof(MatcherOutput));
    if (!output) {
        free(input_ids);
        for (size_t i = 0; i < input_count; i++) free(input_words[i]);
        free(input_words);
        return NULL;
    }

    output->results = malloc(chain->count * sizeof(MatchResult));
    if (!output->results) {
        free(output);
        free(input_ids);
        for (size_t i = 0; i < input_count; i++) free(input_words[i]);
        free(input_words);
        return NULL;
    }

    output->count = 0;
    double total_score = 0;

    for (size_t t = 0; t < chain->count; t++) {
        int matches = 0;
        int positions[3] = {0, 0, 0};

        for (int p = 0; p < 3; p++) {
            for (size_t w = 0; w < valid_count; w++) {
                if (chain->triangles[t].word_ids[p] == input_ids[w]) {
                    matches++;
                    positions[p] = 1;
                    break;
                }
            }
        }

        if (matches > 0) {
            double score = (double)matches / 3.0;
            output->results[output->count].triangle_id = chain->triangles[t].id;
            output->results[output->count].score = score;
            output->results[output->count].matched_words = matches;
            output->results[output->count].total_words = valid_count;
            output->results[output->count].matched_positions = malloc(3 * sizeof(int));
            memcpy(output->results[output->count].matched_positions, positions, 3 * sizeof(int));
            output->count++;
            total_score += score;
        }
    }

    for (size_t i = 0; i < output->count - 1; i++) {
        for (size_t j = i + 1; j < output->count; j++) {
            if (output->results[j].score > output->results[i].score) {
                MatchResult temp = output->results[i];
                output->results[i] = output->results[j];
                output->results[j] = temp;
            }
        }
    }

    int best_coverage = 0;
    for (size_t t = 0; t < chain->count; t++) {
        int seq_match = check_sequential_match(chain, t, input_ids, valid_count);
        if (seq_match > best_coverage) {
            best_coverage = seq_match;
        }
    }

    output->accuracy = valid_count > 0 ? (double)best_coverage / (double)valid_count : 0;
    output->deviation = 1.0 - output->accuracy;

    output->reconstructed = malloc(4096);
    if (output->reconstructed) {
        output->reconstructed[0] = '\0';
        int found_words[100] = {0};

        for (size_t i = 0; i < input_count && i < 100; i++) {
            for (size_t t = 0; t < chain->count && !found_words[i]; t++) {
                for (int p = 0; p < 3; p++) {
                    if (chain->triangles[t].word_ids[p] > 0) {
                        const char *w = chain->vocab->words[chain->triangles[t].word_ids[p] - 1].text;
                        if (strcmp(w, input_words[i]) == 0) {
                            found_words[i] = 1;
                            break;
                        }
                    }
                }
            }
        }

        for (size_t i = 0; i < input_count; i++) {
            if (i > 0) strcat(output->reconstructed, " ");
            if (found_words[i]) {
                strcat(output->reconstructed, input_words[i]);
            } else {
                strcat(output->reconstructed, "[?]");
            }
        }
    }

    free(input_ids);
    for (size_t i = 0; i < input_count; i++) free(input_words[i]);
    free(input_words);

    return output;
}

void matcher_free(MatcherOutput *output) {
    if (!output) return;
    for (size_t i = 0; i < output->count; i++) {
        free(output->results[i].matched_positions);
    }
    free(output->results);
    free(output->reconstructed);
    free(output);
}

void matcher_print(const MatcherOutput *output) {
    if (!output) return;

    printf("\n=== Matcher Results ===\n");
    printf("Found %zu matching triangles\n", output->count);
    printf("Accuracy: %.1f%%\n", output->accuracy * 100);
    printf("Deviation: %.1f%%\n", output->deviation * 100);

    if (output->reconstructed) {
        printf("\nReconstructed: %s\n", output->reconstructed);
    }

    printf("\nTop matches (by triangle fill):\n");
    size_t show = output->count < 10 ? output->count : 10;
    for (size_t i = 0; i < show; i++) {
        printf("  Triangle %d: %.0f%% filled [%s%s%s]\n",
               output->results[i].triangle_id,
               output->results[i].score * 100,
               output->results[i].matched_positions[0] ? "A" : "",
               output->results[i].matched_positions[1] ? "B" : "",
               output->results[i].matched_positions[2] ? "C" : "");
    }
}
