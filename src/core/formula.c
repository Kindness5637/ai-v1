#include "formula.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

FormulaResult *formula_calculate(const TriangleChain *chain, int power) {
    if (!chain) return NULL;

    FormulaResult *result = malloc(sizeof(FormulaResult));
    if (!result) return NULL;

    result->power = power;
    result->sum_a = 0;
    result->sum_b = 0;
    result->sum_c = 0;

    for (size_t i = 0; i < chain->count; i++) {
        double tri_pos = (double)(i + 1);
        double multiplier = pow(tri_pos, power);

        if (chain->triangles[i].word_ids[0] > 0) {
            result->sum_a += chain->triangles[i].word_ids[0] * multiplier;
        }
        if (chain->triangles[i].word_ids[1] > 0) {
            result->sum_b += chain->triangles[i].word_ids[1] * multiplier;
        }
        if (chain->triangles[i].word_ids[2] > 0) {
            result->sum_c += chain->triangles[i].word_ids[2] * multiplier;
        }
    }

    result->total = result->sum_a + result->sum_b + result->sum_c;

    return result;
}

void formula_free(FormulaResult *result) {
    free(result);
}

void formula_print(const FormulaResult *result) {
    if (!result) return;

    printf("\n=== Formula Calculation ===\n");
    printf("Power: %d\n", result->power);
    printf("Sum A: %.0f\n", result->sum_a);
    printf("Sum B: %.0f\n", result->sum_b);
    printf("Sum C: %.0f\n", result->sum_c);
    printf("Total: %.0f\n", result->total);
}
