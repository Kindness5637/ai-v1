#include "triangle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static char **tokenize(const char *sentence, size_t *word_count) {
    char *copy = strdup(sentence);
    if (!copy) return NULL;

    size_t capacity = 16;
    char **words = malloc(capacity * sizeof(char *));
    if (!words) {
        free(copy);
        return NULL;
    }

    *word_count = 0;
    char *token = strtok(copy, " ,.!?;:\n");
    while (token) {
        if (*word_count >= capacity) {
            capacity *= 2;
            char **temp = realloc(words, capacity * sizeof(char *));
            if (!temp) {
                for (size_t i = 0; i < *word_count; i++) free(words[i]);
                free(words);
                free(copy);
                return NULL;
            }
            words = temp;
        }
        words[*word_count] = strdup(token);
        (*word_count)++;
        token = strtok(NULL, " ,.!?;:\n");
    }

    free(copy);
    return words;
}

static char *str_to_lower(const char *str) {
    char *lower = malloc(strlen(str) + 1);
    if (!lower) return NULL;
    for (size_t i = 0; str[i]; i++) {
        lower[i] = tolower((unsigned char)str[i]);
    }
    lower[strlen(str)] = '\0';
    return lower;
}

TriangleChain *create_triangles(const char *sentence) {
    size_t word_count = 0;
    char **words = tokenize(sentence, &word_count);
    if (!words || word_count == 0) return NULL;

    TriangleChain *chain = malloc(sizeof(TriangleChain));
    if (!chain) {
        for (size_t i = 0; i < word_count; i++) free(words[i]);
        free(words);
        return NULL;
    }

    chain->vocab = vocab_create();
    chain->registry = registry_create();
    if (!chain->vocab || !chain->registry) {
        vocab_free(chain->vocab);
        registry_free(chain->registry);
        free(chain);
        for (size_t i = 0; i < word_count; i++) free(words[i]);
        free(words);
        return NULL;
    }

    size_t tri_count = (word_count + 2) / 3;
    chain->triangles = malloc(tri_count * sizeof(Triangle));
    if (!chain->triangles) {
        vocab_free(chain->vocab);
        registry_free(chain->registry);
        free(chain);
        for (size_t i = 0; i < word_count; i++) free(words[i]);
        free(words);
        return NULL;
    }

    chain->count = tri_count;
    size_t word_idx = 0;

    for (size_t i = 0; i < tri_count; i++) {
        chain->triangles[i].id = i + 1;
        for (size_t j = 0; j < 3; j++) {
            if (word_idx < word_count) {
                char *lower = str_to_lower(words[word_idx]);
                chain->triangles[i].words[j] = lower;
                int word_id = vocab_get_or_add(chain->vocab, lower);
                chain->triangles[i].word_ids[j] = word_id;
                registry_add(chain->registry, word_id, i + 1, j, lower);
                word_idx++;
            } else {
                chain->triangles[i].words[j] = strdup("-");
                chain->triangles[i].word_ids[j] = 0;
            }
        }
    }

    for (size_t i = 0; i < word_count; i++) free(words[i]);
    free(words);

    return chain;
}

void free_triangles(TriangleChain *chain) {
    if (!chain) return;
    for (size_t i = 0; i < chain->count; i++) {
        for (size_t j = 0; j < 3; j++) {
            free(chain->triangles[i].words[j]);
        }
    }
    free(chain->triangles);
    vocab_free(chain->vocab);
    registry_free(chain->registry);
    free(chain);
}

void print_triangles(const TriangleChain *chain) {
    if (!chain) return;

    for (size_t i = 0; i < chain->count; i++) {
        printf("\nTriangle [ID:%d]:\n", chain->triangles[i].id);
        printf("    %s (WordID:%d) (A)\n",
               chain->triangles[i].words[0],
               chain->triangles[i].word_ids[0]);
        printf("       /\\\n");
        printf("      /  \\\n");
        printf("     /    \\\n");
        printf("    /      \\\n");
        printf(" %s (WordID:%d) (B)---%s (WordID:%d) (C)\n",
               chain->triangles[i].words[1],
               chain->triangles[i].word_ids[1],
               chain->triangles[i].words[2],
               chain->triangles[i].word_ids[2]);
    }
}

void print_vocabulary(const TriangleChain *chain) {
    if (!chain || !chain->vocab) return;

    printf("\n=== Vocabulary ===\n");
    printf("%-8s %-15s\n", "WordID", "Word");
    printf("------------------------\n");
    for (size_t i = 0; i < chain->vocab->count; i++) {
        printf("%-8d %-15s\n",
               chain->vocab->words[i].id,
               chain->vocab->words[i].text);
    }
}

void print_registry(const TriangleChain *chain) {
    if (!chain || !chain->registry) return;
    registry_print(chain->registry);
}
