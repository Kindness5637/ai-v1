#include "circle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static int is_conjunction(const char *word) {
    const char *conjunctions[] = {
        "and", "or", "but", "nor", "for", "yet", "so",
        "because", "although", "while", "if", "when",
        "after", "before", "since", "until", "unless",
        NULL
    };
    for (int i = 0; conjunctions[i]; i++) {
        if (strcmp(word, conjunctions[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

CircleChain *circle_create(const TriangleChain *chain, double multiplier) {
    if (!chain) return NULL;

    CircleChain *circles = malloc(sizeof(CircleChain));
    if (!circles) return NULL;

    circles->multiplier = multiplier;
    circles->count = 0;
    circles->circles = NULL;

    size_t capacity = 256;
    circles->circles = malloc(capacity * sizeof(Circle));
    if (!circles->circles) {
        free(circles);
        return NULL;
    }

    for (size_t i = 0; i < chain->count; i++) {
        for (int p = 0; p < 3; p++) {
            int word_id = chain->triangles[i].word_ids[p];
            if (word_id > 0 && word_id <= (int)chain->vocab->count) {
                const char *word = chain->vocab->words[word_id - 1].text;
                if (is_conjunction(word)) {
                    if (circles->count >= capacity) {
                        capacity *= 2;
                        Circle *temp = realloc(circles->circles, capacity * sizeof(Circle));
                        if (!temp) {
                            circle_free(circles);
                            return NULL;
                        }
                        circles->circles = temp;
                    }

                    circles->circles[circles->count].id = circles->count + 1;
                    circles->circles[circles->count].word_id = word_id;
                    circles->circles[circles->count].word = strdup(word);
                    circles->circles[circles->count].before_triangle_id = (i > 0) ? i : 0;
                    circles->circles[circles->count].after_triangle_id = (i < chain->count - 1) ? i + 2 : 0;
                    circles->circles[circles->count].value = (circles->count + 1) * multiplier;
                    circles->count++;
                }
            }
        }
    }

    return circles;
}

void circle_free(CircleChain *circles) {
    if (!circles) return;
    for (size_t i = 0; i < circles->count; i++) {
        free(circles->circles[i].word);
    }
    free(circles->circles);
    free(circles);
}

void circle_print(const CircleChain *circles) {
    if (!circles) return;

    printf("\n=== Circle Chain (Conjunctions) ===\n");
    printf("Multiplier: %.2f\n", circles->multiplier);
    printf("Found %zu conjunctions\n\n", circles->count);

    printf("%-8s %-12s %-10s %-12s %-10s\n", "ID", "Word", "WordID", "Before Tri", "After Tri");
    printf("----------------------------------------------------------\n");

    for (size_t i = 0; i < circles->count; i++) {
        printf("%-8d %-12s %-10d %-12d %-10d\n",
               circles->circles[i].id,
               circles->circles[i].word,
               circles->circles[i].word_id,
               circles->circles[i].before_triangle_id,
               circles->circles[i].after_triangle_id);
    }
}

double circle_calculate_impact(const CircleChain *circles, const TriangleChain *chain) {
    if (!circles || !chain) return 0;

    double total_impact = 0;

    for (size_t i = 0; i < circles->count; i++) {
        double before_value = 0;
        double after_value = 0;

        if (circles->circles[i].before_triangle_id > 0) {
            int tri_idx = circles->circles[i].before_triangle_id - 1;
            for (int p = 0; p < 3; p++) {
                before_value += chain->triangles[tri_idx].word_ids[p];
            }
        }

        if (circles->circles[i].after_triangle_id > 0) {
            int tri_idx = circles->circles[i].after_triangle_id - 1;
            for (int p = 0; p < 3; p++) {
                after_value += chain->triangles[tri_idx].word_ids[p];
            }
        }

        double impact = circles->circles[i].value * before_value * after_value;
        total_impact += impact;
    }

    return total_impact;
}
