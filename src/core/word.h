#ifndef WORD_H
#define WORD_H

#include <stddef.h>

typedef struct {
    int word_id;
    int triangle_id;
    int position;
    char *word;
} WordEntry;

typedef struct {
    int id;
    char *text;
} Word;

typedef struct {
    char *key;
    int word_index;
} VocabHashEntry;

typedef struct {
    Word *words;
    size_t count;
    size_t capacity;
    VocabHashEntry *index;
    size_t index_capacity;
    size_t index_count;
} Vocabulary;

typedef struct {
    WordEntry *entries;
    size_t count;
    size_t capacity;
} WordRegistry;

Vocabulary *vocab_create(void);
void vocab_free(Vocabulary *vocab);
int vocab_get_or_add(Vocabulary *vocab, const char *word);
const char *vocab_get_word(const Vocabulary *vocab, int id);

WordRegistry *registry_create(void);
void registry_free(WordRegistry *registry);
void registry_add(WordRegistry *registry, int word_id, int triangle_id, int position, const char *word);
void registry_print(const WordRegistry *registry);
void registry_print_by_word(const WordRegistry *registry, int word_id);

#endif
