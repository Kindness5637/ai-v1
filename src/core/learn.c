#include "learn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int count_word_occurrences(const TriangleChain *chain, int word_id) {
    int count = 0;
    for (size_t i = 0; i < chain->count; i++) {
        for (int p = 0; p < 3; p++) {
            if (chain->triangles[i].word_ids[p] == word_id) {
                count++;
            }
        }
    }
    return count;
}

LearningOutput *learn_compare(const TriangleChain *old_chain, const TriangleChain *new_chain) {
    if (!old_chain || !new_chain) return NULL;

    LearningOutput *output = malloc(sizeof(LearningOutput));
    if (!output) return NULL;

    output->count = old_chain->vocab->count;
    output->words = malloc(output->count * sizeof(WordLearning));
    if (!output->words) {
        free(output);
        return NULL;
    }

    output->total_deviation = 0;
    output->new_words_found = 0;
    output->context_changes = 0;
    output->new_triangles = NULL;

    for (size_t i = 0; i < old_chain->vocab->count; i++) {
        int word_id = old_chain->vocab->words[i].id;
        output->words[i].word_id = word_id;
        output->words[i].old_context_count = count_word_occurrences(old_chain, word_id);
        output->words[i].new_context_count = count_word_occurrences(new_chain, word_id);

        int diff = output->words[i].new_context_count - output->words[i].old_context_count;
        if (diff < 0) diff = -diff;

        output->words[i].deviation = output->words[i].old_context_count > 0 ?
            (double)diff / output->words[i].old_context_count : 1.0;

        output->total_deviation += output->words[i].deviation;
        if (diff > 0) output->context_changes++;
    }

    for (size_t i = 0; i < new_chain->vocab->count; i++) {
        int found = 0;
        for (size_t j = 0; j < old_chain->vocab->count; j++) {
            if (strcmp(new_chain->vocab->words[i].text, old_chain->vocab->words[j].text) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) output->new_words_found++;
    }

    if (old_chain->vocab->count > 0) {
        output->total_deviation /= old_chain->vocab->count;
    }

    return output;
}

void learning_free(LearningOutput *output) {
    if (!output) return;
    free(output->words);
    if (output->new_triangles) {
        for (size_t i = 0; i < output->new_triangles->count; i++) {
            for (int p = 0; p < 3; p++) {
                free(output->new_triangles->triangles[i].words[p]);
            }
        }
        free(output->new_triangles->triangles);
        free(output->new_triangles->vocab);
        free(output->new_triangles);
    }
    free(output);
}

void learning_print(const LearningOutput *output) {
    if (!output) return;

    printf("\n=== Learning Analysis ===\n");
    printf("Total deviation: %.1f%%\n", output->total_deviation * 100);
    printf("Context changes: %d\n", output->context_changes);
    printf("New words found: %d\n\n", output->new_words_found);

    printf("Words with context changes:\n");
    int shown = 0;
    for (size_t i = 0; i < output->count && shown < 20; i++) {
        if (output->words[i].old_context_count != output->words[i].new_context_count) {
            printf("  WordID %d: old=%d new=%d deviation=%.1f%%\n",
                   output->words[i].word_id,
                   output->words[i].old_context_count,
                   output->words[i].new_context_count,
                   output->words[i].deviation * 100);
            shown++;
        }
    }
}

TriangleChain *learn_merge(const TriangleChain *old_chain, const TriangleChain *new_chain) {
    if (!old_chain || !new_chain) return NULL;

    TriangleChain *merged = malloc(sizeof(TriangleChain));
    if (!merged) return NULL;

    merged->count = old_chain->count + new_chain->count;
    merged->triangles = malloc(merged->count * sizeof(Triangle));
    if (!merged->triangles) {
        free(merged);
        return NULL;
    }

    for (size_t i = 0; i < old_chain->count; i++) {
        merged->triangles[i] = old_chain->triangles[i];
        for (int p = 0; p < 3; p++) {
            merged->triangles[i].words[p] = strdup(old_chain->triangles[i].words[p]);
        }
    }

    for (size_t i = 0; i < new_chain->count; i++) {
        size_t idx = old_chain->count + i;
        merged->triangles[idx] = new_chain->triangles[i];
        merged->triangles[idx].id = old_chain->count + i + 1;
        for (int p = 0; p < 3; p++) {
            merged->triangles[idx].words[p] = strdup(new_chain->triangles[i].words[p]);
        }
    }

    merged->vocab = malloc(sizeof(Vocabulary));
    if (!merged->vocab) {
        for (size_t i = 0; i < merged->count; i++) {
            for (int p = 0; p < 3; p++) {
                free(merged->triangles[i].words[p]);
            }
        }
        free(merged->triangles);
        free(merged);
        return NULL;
    }

    merged->vocab->count = old_chain->vocab->count + new_chain->vocab->count;
    merged->vocab->words = malloc(merged->vocab->count * sizeof(Word));
    if (!merged->vocab->words) {
        free(merged->vocab);
        for (size_t i = 0; i < merged->count; i++) {
            for (int p = 0; p < 3; p++) {
                free(merged->triangles[i].words[p]);
            }
        }
        free(merged->triangles);
        free(merged);
        return NULL;
    }

    for (size_t i = 0; i < old_chain->vocab->count; i++) {
        merged->vocab->words[i].id = old_chain->vocab->words[i].id;
        merged->vocab->words[i].text = strdup(old_chain->vocab->words[i].text);
    }

    for (size_t i = 0; i < new_chain->vocab->count; i++) {
        size_t idx = old_chain->vocab->count + i;
        merged->vocab->words[idx].id = old_chain->vocab->count + i + 1;
        merged->vocab->words[idx].text = strdup(new_chain->vocab->words[i].text);
    }

    merged->registry = NULL;

    return merged;
}
