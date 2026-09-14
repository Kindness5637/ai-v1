#include "reconstruct.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

ReconstructedText *reconstruct(const TriangleChain *chain) {
    if (!chain) return NULL;

    ReconstructedText *text = malloc(sizeof(ReconstructedText));
    if (!text) return NULL;

    text->count = chain->count * 3;
    text->words = malloc(text->count * sizeof(char *));
    if (!text->words) {
        free(text);
        return NULL;
    }

    text->found = 0;
    text->total = 0;
    size_t idx = 0;

    for (size_t i = 0; i < chain->count; i++) {
        for (int p = 0; p < 3; p++) {
            int word_id = chain->triangles[i].word_ids[p];
            if (word_id > 0 && word_id <= (int)chain->vocab->count) {
                text->words[idx] = strdup(chain->vocab->words[word_id - 1].text);
                text->found++;
            } else {
                text->words[idx] = strdup("[?]");
            }
            text->total++;
            idx++;
        }
    }

    text->count = idx;
    return text;
}

void reconstruct_free(ReconstructedText *text) {
    if (!text) return;
    for (size_t i = 0; i < text->count; i++) {
        free(text->words[i]);
    }
    free(text->words);
    free(text);
}

void reconstruct_print(const ReconstructedText *text) {
    if (!text) return;

    printf("\n=== Reconstructed Text ===\n");
    printf("Found %d/%d words\n\n", text->found, text->total);

    for (size_t i = 0; i < text->count; i++) {
        printf("%s ", text->words[i]);
        if ((i + 1) % 10 == 0) printf("\n");
    }
    printf("\n");
}

void reconstruct_compare(const ReconstructedText *text, const char *original) {
    if (!text || !original) return;

    printf("\n=== Comparison ===\n");

    char *copy = strdup(original);
    if (!copy) return;

    size_t orig_count = 0;
    char *orig_words[1000];
    char *token = strtok(copy, " ,.!?;:\n\"");
    while (token && orig_count < 1000) {
        orig_words[orig_count] = token;
        orig_count++;
        token = strtok(NULL, " ,.!?;:\n\"");
    }

    int matches = 0;
    size_t total = text->count < orig_count ? text->count : orig_count;

    for (size_t i = 0; i < total; i++) {
        char *lower = malloc(strlen(orig_words[i]) + 1);
        for (size_t j = 0; orig_words[i][j]; j++) {
            lower[j] = tolower((unsigned char)orig_words[i][j]);
        }
        lower[strlen(orig_words[i])] = '\0';

        if (strcmp(text->words[i], lower) == 0) {
            matches++;
        }
        free(lower);
    }

    printf("Matches: %d/%zu\n", matches, total);
    printf("Accuracy: %.1f%%\n", total > 0 ? (double)matches / total * 100 : 0);

    free(copy);
}
