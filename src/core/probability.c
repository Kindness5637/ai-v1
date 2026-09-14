#include "probability.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void add_cooccurrence(CoOccurrenceMatrix *matrix, int word_id, int neighbor_id, int position) {
    for (size_t i = 0; i < matrix->count; i++) {
        if (matrix->entries[i].word_id == word_id && 
            matrix->entries[i].neighbor_id == neighbor_id &&
            matrix->entries[i].position == position) {
            matrix->entries[i].count++;
            return;
        }
    }

    if (matrix->count >= matrix->capacity) {
        matrix->capacity = matrix->capacity == 0 ? 64 : matrix->capacity * 2;
        CoOccurrence *temp = realloc(matrix->entries, matrix->capacity * sizeof(CoOccurrence));
        if (!temp) return;
        matrix->entries = temp;
    }

    matrix->entries[matrix->count].word_id = word_id;
    matrix->entries[matrix->count].neighbor_id = neighbor_id;
    matrix->entries[matrix->count].position = position;
    matrix->entries[matrix->count].count = 1;
    matrix->count++;
}

typedef struct {
    int word_id;
    int before_id;
    int after_id;
    int before_count;
    int after_count;
} SequenceEntry;

static void add_sequence(SequenceEntry **entries, size_t *count, size_t *cap, int word_id, int before_id, int after_id) {
    for (size_t i = 0; i < *count; i++) {
        if ((*entries)[i].word_id == word_id) {
            if (before_id > 0) {
                if ((*entries)[i].before_id == before_id) {
                    (*entries)[i].before_count++;
                }
            }
            if (after_id > 0) {
                if ((*entries)[i].after_id == after_id) {
                    (*entries)[i].after_count++;
                }
            }
            return;
        }
    }

    if (*count >= *cap) {
        *cap = *cap == 0 ? 64 : *cap * 2;
        SequenceEntry *temp = realloc(*entries, *cap * sizeof(SequenceEntry));
        if (!temp) return;
        *entries = temp;
    }

    (*entries)[*count].word_id = word_id;
    (*entries)[*count].before_id = before_id;
    (*entries)[*count].after_id = after_id;
    (*entries)[*count].before_count = before_id > 0 ? 1 : 0;
    (*entries)[*count].after_count = after_id > 0 ? 1 : 0;
    (*count)++;
}

ProbabilityMatrix *prob_create(const TriangleChain *chain) {
    if (!chain) return NULL;

    CoOccurrenceMatrix cooc = {NULL, 0, 0};
    SequenceEntry *seq_entries = NULL;
    size_t seq_count = 0;
    size_t seq_cap = 0;

    for (size_t i = 0; i < chain->count; i++) {
        int word_a = chain->triangles[i].word_ids[0];
        int word_b = chain->triangles[i].word_ids[1];
        int word_c = chain->triangles[i].word_ids[2];

        if (word_a > 0) {
            if (word_b > 0) add_cooccurrence(&cooc, word_a, word_b, 0);
            if (word_c > 0) add_cooccurrence(&cooc, word_a, word_c, 0);
        }
        if (word_b > 0) {
            if (word_a > 0) add_cooccurrence(&cooc, word_b, word_a, 1);
            if (word_c > 0) add_cooccurrence(&cooc, word_b, word_c, 1);
        }
        if (word_c > 0) {
            if (word_a > 0) add_cooccurrence(&cooc, word_c, word_a, 2);
            if (word_b > 0) add_cooccurrence(&cooc, word_c, word_b, 2);
        }

        int prev_word = (i > 0) ? chain->triangles[i - 1].word_ids[2] : 0;
        int next_first = (i < chain->count - 1) ? chain->triangles[i + 1].word_ids[0] : 0;

        if (word_a > 0) add_sequence(&seq_entries, &seq_count, &seq_cap, word_a, prev_word, next_first);
        if (word_b > 0) add_sequence(&seq_entries, &seq_count, &seq_cap, word_b, prev_word, next_first);
        if (word_c > 0) add_sequence(&seq_entries, &seq_count, &seq_cap, word_c, prev_word, next_first);
    }

    size_t unique_words = 0;
    int *seen = calloc(chain->vocab->count + 1, sizeof(int));
    if (!seen) {
        free(cooc.entries);
        free(seq_entries);
        return NULL;
    }

    for (size_t i = 0; i < cooc.count; i++) {
        if (!seen[cooc.entries[i].word_id]) {
            seen[cooc.entries[i].word_id] = 1;
            unique_words++;
        }
    }

    ProbabilityMatrix *prob = malloc(sizeof(ProbabilityMatrix));
    if (!prob) {
        free(seen);
        free(cooc.entries);
        free(seq_entries);
        return NULL;
    }

    prob->count = unique_words;
    prob->words = malloc(unique_words * sizeof(WordProbability));
    if (!prob->words) {
        free(prob);
        free(seen);
        free(cooc.entries);
        free(seq_entries);
        return NULL;
    }

    size_t word_idx = 0;
    for (int w = 1; w <= (int)chain->vocab->count; w++) {
        if (!seen[w]) continue;

        size_t entry_count = 0;
        int total = 0;

        for (size_t i = 0; i < cooc.count; i++) {
            if (cooc.entries[i].word_id == w) {
                entry_count++;
                total += cooc.entries[i].count;
            }
        }

        prob->words[word_idx].word_id = w;
        prob->words[word_idx].count = entry_count;
        prob->words[word_idx].neighbor_ids = malloc(entry_count * sizeof(int));
        prob->words[word_idx].positions = malloc(entry_count * sizeof(int));
        prob->words[word_idx].probabilities = malloc(entry_count * sizeof(double));

        size_t n = 0;
        for (size_t i = 0; i < cooc.count; i++) {
            if (cooc.entries[i].word_id == w) {
                prob->words[word_idx].neighbor_ids[n] = cooc.entries[i].neighbor_id;
                prob->words[word_idx].positions[n] = cooc.entries[i].position;
                prob->words[word_idx].probabilities[n] = (double)cooc.entries[i].count / total;
                n++;
            }
        }

        word_idx++;
    }

    prob->seq_count = seq_count;
    prob->sequences = malloc(seq_count * sizeof(SequenceProbability));
    if (prob->sequences) {
        for (size_t i = 0; i < seq_count; i++) {
            prob->sequences[i].word_id = seq_entries[i].word_id;
            prob->sequences[i].before_count = seq_entries[i].before_count;
            prob->sequences[i].after_count = seq_entries[i].after_count;
            
            prob->sequences[i].before_ids = malloc(sizeof(int));
            prob->sequences[i].after_ids = malloc(sizeof(int));
            prob->sequences[i].before_probs = malloc(sizeof(double));
            prob->sequences[i].after_probs = malloc(sizeof(double));
            
            if (prob->sequences[i].before_ids) {
                prob->sequences[i].before_ids[0] = seq_entries[i].before_id;
                prob->sequences[i].before_probs[0] = 1.0;
            }
            if (prob->sequences[i].after_ids) {
                prob->sequences[i].after_ids[0] = seq_entries[i].after_id;
                prob->sequences[i].after_probs[0] = 1.0;
            }
        }
    }

    free(seen);
    free(cooc.entries);
    free(seq_entries);

    return prob;
}

void prob_free(ProbabilityMatrix *prob) {
    if (!prob) return;
    for (size_t i = 0; i < prob->count; i++) {
        free(prob->words[i].neighbor_ids);
        free(prob->words[i].positions);
        free(prob->words[i].probabilities);
    }
    free(prob->words);
    
    for (size_t i = 0; i < prob->seq_count; i++) {
        free(prob->sequences[i].before_ids);
        free(prob->sequences[i].after_ids);
        free(prob->sequences[i].before_probs);
        free(prob->sequences[i].after_probs);
    }
    free(prob->sequences);
    free(prob);
}

void prob_print(const ProbabilityMatrix *prob) {
    if (!prob) return;

    printf("\n=== Word Probability Matrix (with Positions) ===\n");

    for (size_t i = 0; i < prob->count; i++) {
        printf("\nWordID %d neighbors:\n", prob->words[i].word_id);
        for (size_t j = 0; j < prob->words[i].count; j++) {
            const char *pos;
            switch (prob->words[i].positions[j]) {
                case 0: pos = "A"; break;
                case 1: pos = "B"; break;
                case 2: pos = "C"; break;
                default: pos = "?"; break;
            }
            printf("  -> WordID %d at pos %s: %.2f%%\n",
                   prob->words[i].neighbor_ids[j],
                   pos,
                   prob->words[i].probabilities[j] * 100);
        }
    }
}

void prob_print_word(const ProbabilityMatrix *prob, int word_id) {
    if (!prob) return;

    for (size_t i = 0; i < prob->count; i++) {
        if (prob->words[i].word_id == word_id) {
            printf("\nWordID %d probability distribution:\n", word_id);
            for (size_t j = 0; j < prob->words[i].count; j++) {
                const char *pos;
                switch (prob->words[i].positions[j]) {
                    case 0: pos = "A"; break;
                    case 1: pos = "B"; break;
                    case 2: pos = "C"; break;
                    default: pos = "?"; break;
                }
                printf("  -> WordID %d at pos %s: %.2f%%\n",
                       prob->words[i].neighbor_ids[j],
                       pos,
                       prob->words[i].probabilities[j] * 100);
            }
            return;
        }
    }
    printf("WordID %d not found\n", word_id);
}

void prob_print_sequences(const ProbabilityMatrix *prob) {
    if (!prob || !prob->sequences) return;

    printf("\n=== Sequence Probabilities (Before/After) ===\n");
    for (size_t i = 0; i < prob->seq_count; i++) {
        SequenceProbability *seq = &prob->sequences[i];
        printf("\nWordID %d:\n", seq->word_id);
        
        if (seq->before_count > 0 && seq->before_ids[0] > 0) {
            printf("  Before: WordID %d (100%%)\n", seq->before_ids[0]);
        } else {
            printf("  Before: None (start of chain)\n");
        }
        
        if (seq->after_count > 0 && seq->after_ids[0] > 0) {
            printf("  After:  WordID %d (100%%)\n", seq->after_ids[0]);
        } else {
            printf("  After:  None (end of chain)\n");
        }
    }
}
