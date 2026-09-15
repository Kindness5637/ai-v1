#include "word.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long vocab_hash_word(const char *word) {
    unsigned long hash = 5381;
    int c;
    while ((c = (unsigned char)*word++) != '\0') {
        hash = ((hash << 5) + hash) + (unsigned long)c;
    }
    return hash;
}

static int vocab_index_resize(Vocabulary *vocab, size_t capacity) {
    VocabHashEntry *old_index = vocab->index;
    size_t old_capacity = vocab->index_capacity;

    vocab->index = calloc(capacity, sizeof(VocabHashEntry));
    if (!vocab->index) {
        vocab->index = old_index;
        return 0;
    }
    vocab->index_capacity = capacity;
    vocab->index_count = 0;

    for (size_t i = 0; i < old_capacity; i++) {
        if (!old_index[i].key) continue;
        size_t slot = vocab_hash_word(old_index[i].key) % vocab->index_capacity;
        while (vocab->index[slot].key) {
            slot = (slot + 1) % vocab->index_capacity;
        }
        vocab->index[slot] = old_index[i];
        vocab->index_count++;
    }

    free(old_index);
    return 1;
}

static int vocab_index_ensure_capacity(Vocabulary *vocab) {
    if (vocab->index_capacity == 0) {
        return vocab_index_resize(vocab, 1024);
    }
    if ((vocab->index_count + 1) * 10 >= vocab->index_capacity * 7) {
        return vocab_index_resize(vocab, vocab->index_capacity * 2);
    }
    return 1;
}

static int vocab_index_find(const Vocabulary *vocab, const char *word) {
    if (!vocab->index || vocab->index_capacity == 0) return -1;
    size_t slot = vocab_hash_word(word) % vocab->index_capacity;
    for (size_t probes = 0; probes < vocab->index_capacity; probes++) {
        const VocabHashEntry *entry = &vocab->index[slot];
        if (!entry->key) return -1;
        if (strcmp(entry->key, word) == 0) return entry->word_index;
        slot = (slot + 1) % vocab->index_capacity;
    }
    return -1;
}

static int vocab_index_insert(Vocabulary *vocab, char *word, int word_index) {
    if (!vocab_index_ensure_capacity(vocab)) return 0;
    size_t slot = vocab_hash_word(word) % vocab->index_capacity;
    while (vocab->index[slot].key) {
        slot = (slot + 1) % vocab->index_capacity;
    }
    vocab->index[slot] = (VocabHashEntry){word, word_index};
    vocab->index_count++;
    return 1;
}

Vocabulary *vocab_create(void) {
    Vocabulary *vocab = malloc(sizeof(Vocabulary));
    if (!vocab) return NULL;
    vocab->words = NULL;
    vocab->count = 0;
    vocab->capacity = 0;
    vocab->index = NULL;
    vocab->index_capacity = 0;
    vocab->index_count = 0;
    return vocab;
}

void vocab_free(Vocabulary *vocab) {
    if (!vocab) return;
    for (size_t i = 0; i < vocab->count; i++) {
        free(vocab->words[i].text);
    }
    free(vocab->words);
    free(vocab->index);
    free(vocab);
}

int vocab_get_or_add(Vocabulary *vocab, const char *word) {
    int existing_index = vocab_index_find(vocab, word);
    if (existing_index >= 0) {
        return vocab->words[existing_index].id;
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
    if (!vocab->words[vocab->count].text) return -1;
    if (!vocab_index_insert(vocab, vocab->words[vocab->count].text,
                            (int)vocab->count)) {
        free(vocab->words[vocab->count].text);
        return -1;
    }
    vocab->count++;

    return new_id;
}

const char *vocab_get_word(const Vocabulary *vocab, int id) {
    if (!vocab || id <= 0 || (size_t)id > vocab->count) return NULL;
    return vocab->words[id - 1].text;
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
