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

int triangle_role_id(const char *upos) {
    static const char *roles[] = {
        "", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
        "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ",
        "SYM", "VERB", "X"
    };
    if (!upos) return 0;
    for (int i = 1; i < (int)(sizeof(roles) / sizeof(roles[0])); i++) {
        if (strcmp(upos, roles[i]) == 0) return i;
    }
    return 0;
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
    chain->owns_vocab = 1;
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
                chain->triangles[i].role_ids[j] = 0;
                registry_add(chain->registry, word_id, i + 1, j, lower);
                word_idx++;
            } else {
                chain->triangles[i].words[j] = strdup("-");
                chain->triangles[i].word_ids[j] = 0;
                chain->triangles[i].role_ids[j] = 0;
            }
        }
    }

    for (size_t i = 0; i < word_count; i++) free(words[i]);
    free(words);

    return chain;
}

TriangleChain *create_triangles_with_vocab(const char *sentence, Vocabulary *vocab) {
    if (!vocab) return NULL;
    TriangleChain *chain = create_triangles(sentence);
    if (!chain) return NULL;

    Vocabulary *local_vocab = chain->vocab;
    for (size_t i = 0; i < chain->count; i++) {
        for (int p = 0; p < 3; p++) {
            if (chain->triangles[i].word_ids[p] <= 0) continue;
            chain->triangles[i].word_ids[p] =
                vocab_get_or_add(vocab, chain->triangles[i].words[p]);
        }
    }
    for (size_t i = 0; i < chain->registry->count; i++) {
        chain->registry->entries[i].word_id =
            vocab_get_or_add(vocab, chain->registry->entries[i].word);
    }
    vocab_free(local_vocab);
    chain->vocab = vocab;
    chain->owns_vocab = 0;
    return chain;
}

typedef struct {
    char *form;
    char upos[16];
    char deprel[32];
} ConlluToken;

static void free_conllu_tokens(ConlluToken *tokens, size_t count) {
    if (!tokens) return;
    for (size_t i = 0; i < count; i++) free(tokens[i].form);
    free(tokens);
}

static int append_conllu_sentence(TriangleChain *chain,
                                  ConlluToken *tokens, size_t token_count) {
    if (!chain || token_count == 0) return 1;

    size_t triangle_count = (token_count + 2) / 3;
    Triangle *grown = realloc(chain->triangles,
                              (chain->count + triangle_count) * sizeof(Triangle));
    if (!grown) return 0;
    chain->triangles = grown;

    for (size_t i = 0; i < triangle_count; i++) {
        Triangle *triangle = &chain->triangles[chain->count + i];
        triangle->id = (int)(chain->count + i + 1);
        for (int p = 0; p < 3; p++) {
            size_t token_index = i * 3 + (size_t)p;
            if (token_index < token_count) {
                char *lower = str_to_lower(tokens[token_index].form);
                if (!lower) return 0;
                triangle->words[p] = lower;
                triangle->word_ids[p] = vocab_get_or_add(chain->vocab, lower);
                triangle->role_ids[p] = triangle_role_id(tokens[token_index].upos);
                registry_add(chain->registry, triangle->word_ids[p],
                             (size_t)triangle->id, p, lower);
                strncpy(triangle->upos[p], tokens[token_index].upos,
                        sizeof(triangle->upos[p]) - 1);
                triangle->upos[p][sizeof(triangle->upos[p]) - 1] = '\0';
                strncpy(triangle->deprel[p], tokens[token_index].deprel,
                        sizeof(triangle->deprel[p]) - 1);
                triangle->deprel[p][sizeof(triangle->deprel[p]) - 1] = '\0';
            } else {
                triangle->words[p] = strdup("-");
                if (!triangle->words[p]) return 0;
                triangle->word_ids[p] = 0;
                triangle->role_ids[p] = 0;
                triangle->upos[p][0] = '\0';
                triangle->deprel[p][0] = '\0';
            }
        }
    }
    chain->count += triangle_count;
    return 1;
}

TriangleChain *create_triangles_from_conllu(const char *content) {
    if (!content) return NULL;

    TriangleChain *chain = calloc(1, sizeof(TriangleChain));
    if (!chain) return NULL;
    chain->vocab = vocab_create();
    chain->registry = registry_create();
    chain->owns_vocab = 1;
    if (!chain->vocab || !chain->registry) {
        vocab_free(chain->vocab);
        registry_free(chain->registry);
        free(chain);
        return NULL;
    }

    char *copy = strdup(content);
    if (!copy) {
        free_triangles(chain);
        return NULL;
    }

    ConlluToken *tokens = NULL;
    size_t token_count = 0, token_capacity = 0;
    char *line_save = NULL;
    char *line = strtok_r(copy, "\n", &line_save);
    while (line) {
        while (*line == '\r') line++;
        if (*line == '\0') {
            if (!append_conllu_sentence(chain, tokens, token_count)) {
                free_conllu_tokens(tokens, token_count);
                free(copy);
                free_triangles(chain);
                return NULL;
            }
            free_conllu_tokens(tokens, token_count);
            tokens = NULL;
            token_count = 0;
            token_capacity = 0;
            line = strtok_r(NULL, "\n", &line_save);
            continue;
        }
        if (*line == '#') {
            if (strncmp(line, "# sent_id", 9) == 0 && token_count > 0) {
                if (!append_conllu_sentence(chain, tokens, token_count)) {
                    free_conllu_tokens(tokens, token_count);
                    free(copy);
                    free_triangles(chain);
                    return NULL;
                }
                free_conllu_tokens(tokens, token_count);
                tokens = NULL;
                token_count = 0;
                token_capacity = 0;
            }
            line = strtok_r(NULL, "\n", &line_save);
            continue;
        }

        char *fields[10] = {0};
        char *field_save = NULL;
        char *field = strtok_r(line, "\t", &field_save);
        int field_count = 0;
        while (field && field_count < 10) {
            fields[field_count++] = field;
            field = strtok_r(NULL, "\t", &field_save);
        }
        /* Skip multi-word-token and empty-node rows; only syntactic word
         * rows receive triangle vertices. */
        if (field_count == 10 && strchr(fields[0], '-') == NULL &&
            strchr(fields[0], '.') == NULL) {
            if (token_count == token_capacity) {
                size_t next_capacity = token_capacity ? token_capacity * 2 : 64;
                ConlluToken *grown = realloc(tokens,
                                             next_capacity * sizeof(ConlluToken));
                if (!grown) {
                    free_conllu_tokens(tokens, token_count);
                    free(copy);
                    free_triangles(chain);
                    return NULL;
                }
                tokens = grown;
                token_capacity = next_capacity;
            }
            tokens[token_count].form = strdup(fields[1]);
            if (!tokens[token_count].form) {
                free_conllu_tokens(tokens, token_count);
                free(copy);
                free_triangles(chain);
                return NULL;
            }
            strncpy(tokens[token_count].upos, fields[3],
                    sizeof(tokens[token_count].upos) - 1);
            tokens[token_count].upos[sizeof(tokens[token_count].upos) - 1] = '\0';
            strncpy(tokens[token_count].deprel, fields[7],
                    sizeof(tokens[token_count].deprel) - 1);
            tokens[token_count].deprel[sizeof(tokens[token_count].deprel) - 1] = '\0';
            token_count++;
        }
        line = strtok_r(NULL, "\n", &line_save);
    }

    if (!append_conllu_sentence(chain, tokens, token_count)) {
        free_conllu_tokens(tokens, token_count);
        free(copy);
        free_triangles(chain);
        return NULL;
    }
    free_conllu_tokens(tokens, token_count);
    free(copy);
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
    if (chain->owns_vocab) vocab_free(chain->vocab);
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
        if (chain->triangles[i].upos[0][0] != '\0') {
            printf("    Roles: %s/%s, %s/%s, %s/%s\n",
                   chain->triangles[i].upos[0], chain->triangles[i].deprel[0],
                   chain->triangles[i].upos[1], chain->triangles[i].deprel[1],
                   chain->triangles[i].upos[2], chain->triangles[i].deprel[2]);
        }
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
