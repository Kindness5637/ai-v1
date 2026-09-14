#include "word.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Vocabulary *vocab_create(void) {
    Vocabulary *vocab = malloc(sizeof(Vocabulary));
    if (!vocab) return NULL;
    vocab->words = NULL;
    vocab->count = 0;
    vocab->capacity = 0;
    return vocab;
}

void vocab_free(Vocabulary *vocab) {
    if (!vocab) return;
    for (size_t i = 0; i < vocab->count; i++) {
        free(vocab->words[i].text);
    }
    free(vocab->words);
    free(vocab);
}

int vocab_get_or_add(Vocabulary *vocab, const char *word) {
    for (size_t i = 0; i < vocab->count; i++) {
        if (strcmp(vocab->words[i].text, word) == 0) {
            return vocab->words[i].id;
        }
    }

    if (vocab->count >= vocab->capacity) {
        vocab->capacity = vocab->capacity == 0 ? 16 : vocab->capacity * 2;
        Word *temp = realloc(vocab->words, vocab->capacity * sizeof(Word));
        if (!temp) return -1;
        vocab->words = temp;
    }

    int new_id = vocab->count + 1;
    vocab->words[vocab->count].id = new_id;
    vocab->words[vocab->count].text = strdup(word);
    vocab->count++;

    return new_id;
}

const char *vocab_get_word(const Vocabulary *vocab, int id) {
    for (size_t i = 0; i < vocab->count; i++) {
        if (vocab->words[i].id == id) {
            return vocab->words[i].text;
        }
    }
    return NULL;
}

WordRegistry *registry_create(void) {
    WordRegistry *reg = malloc(sizeof(WordRegistry));
    if (!reg) return NULL;
    reg->entries = NULL;
    reg->count = 0;
    reg->capacity = 0;
    return reg;
}

void registry_free(WordRegistry *reg) {
    if (!reg) return;
    for (size_t i = 0; i < reg->count; i++) {
        free(reg->entries[i].word);
    }
    free(reg->entries);
    free(reg);
}

void registry_add(WordRegistry *reg, int word_id, int triangle_id, int position, const char *word) {
    if (reg->count >= reg->capacity) {
        reg->capacity = reg->capacity == 0 ? 16 : reg->capacity * 2;
        WordEntry *temp = realloc(reg->entries, reg->capacity * sizeof(WordEntry));
        if (!temp) return;
        reg->entries = temp;
    }

    reg->entries[reg->count].word_id = word_id;
    reg->entries[reg->count].triangle_id = triangle_id;
    reg->entries[reg->count].position = position;
    reg->entries[reg->count].word = strdup(word);
    reg->count++;
}

void registry_print(const WordRegistry *registry) {
    printf("\n=== Word Registry ===\n");
    printf("%-6s %-10s %-12s %-10s %-10s\n", "ID", "Word", "Triangle", "Position", "Coords");
    printf("--------------------------------------------------\n");
    for (size_t i = 0; i < registry->count; i++) {
        const char *pos;
        switch (registry->entries[i].position) {
            case 0: pos = "A"; break;
            case 1: pos = "B"; break;
            case 2: pos = "C"; break;
            default: pos = "?"; break;
        }
        printf("%-6d %-10s T%-10d %-10s [%s]\n",
               registry->entries[i].word_id,
               registry->entries[i].word,
               registry->entries[i].triangle_id,
               pos,
               pos);
    }
}

void registry_print_by_word(const WordRegistry *registry, int word_id) {
    printf("\nWord ID %d appearances:\n", word_id);
    for (size_t i = 0; i < registry->count; i++) {
        if (registry->entries[i].word_id == word_id) {
            const char *pos;
            switch (registry->entries[i].position) {
                case 0: pos = "A"; break;
                case 1: pos = "B"; break;
                case 2: pos = "C"; break;
                default: pos = "?"; break;
            }
            printf("  - Triangle %d, Position %s\n",
                   registry->entries[i].triangle_id, pos);
        }
    }
}
