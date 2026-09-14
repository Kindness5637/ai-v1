#include "punctuation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static int is_punctuation(char c) {
    return c == '.' || c == ',' || c == '?' || c == '!' ||
           c == ';' || c == ':' || c == '"' || c == '\'' ||
           c == '(' || c == ')' || c == '-';
}

PunctuationChain *punctuation_create(const TriangleChain *chain, const char *text) {
    if (!chain || !text) return NULL;

    PunctuationChain *punc = malloc(sizeof(PunctuationChain));
    if (!punc) return NULL;

    punc->count = 0;
    punc->capacity = 256;
    punc->marks = malloc(punc->capacity * sizeof(PunctuationMark));
    if (!punc->marks) {
        free(punc);
        return NULL;
    }

    size_t text_len = strlen(text);
    int in_word = 0;
    int tri_idx = 0;
    int pos_in_tri = 0;

    for (size_t i = 0; i < text_len && tri_idx < (int)chain->count; i++) {
        char c = text[i];

        if (is_punctuation(c)) {
            if (punc->count >= punc->capacity) {
                punc->capacity *= 2;
                PunctuationMark *temp = realloc(punc->marks, punc->capacity * sizeof(PunctuationMark));
                if (!temp) {
                    punctuation_free(punc);
                    return NULL;
                }
                punc->marks = temp;
            }

            int word_id = 0;
            if (pos_in_tri > 0) {
                word_id = chain->triangles[tri_idx].word_ids[pos_in_tri - 1];
            } else if (tri_idx > 0) {
                word_id = chain->triangles[tri_idx - 1].word_ids[2];
            }

            punc->marks[punc->count].id = punc->count + 1;
            punc->marks[punc->count].symbol = c;
            punc->marks[punc->count].position = i;
            punc->marks[punc->count].triangle_id = tri_idx + 1;
            punc->marks[punc->count].word_id = word_id;
            punc->count++;
        }

        if (isspace(c) || is_punctuation(c)) {
            if (in_word) {
                in_word = 0;
                pos_in_tri++;
                if (pos_in_tri >= 3) {
                    tri_idx++;
                    pos_in_tri = 0;
                }
            }
        } else if (!in_word) {
            in_word = 1;
        }
    }

    return punc;
}

void punctuation_free(PunctuationChain *chain) {
    if (!chain) return;
    free(chain->marks);
    free(chain);
}

void punctuation_print(const PunctuationChain *chain) {
    if (!chain) return;

    printf("\n=== Punctuation Marks ===\n");
    printf("Found %zu marks\n\n", chain->count);

    printf("%-6s %-10s %-12s %-10s\n", "ID", "Symbol", "Triangle", "WordID");
    printf("--------------------------------------------\n");

    for (size_t i = 0; i < chain->count; i++) {
        printf("%-6d %-10c %-12d %-10d\n",
               chain->marks[i].id,
               chain->marks[i].symbol,
               chain->marks[i].triangle_id,
               chain->marks[i].word_id);
    }
}

int punctuation_count_symbol(const PunctuationChain *chain, char symbol) {
    if (!chain) return 0;
    int count = 0;
    for (size_t i = 0; i < chain->count; i++) {
        if (chain->marks[i].symbol == symbol) count++;
    }
    return count;
}
