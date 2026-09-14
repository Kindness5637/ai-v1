#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

MatrixTable *matrix_create(const TriangleChain *chain) {
    if (!chain) return NULL;

    MatrixTable *matrix = malloc(sizeof(MatrixTable));
    if (!matrix) return NULL;

    matrix->count = chain->count;
    matrix->rows = malloc(matrix->count * sizeof(TriangleRow));
    if (!matrix->rows) {
        free(matrix);
        return NULL;
    }

    matrix->grand_total = 0;

    for (size_t i = 0; i < chain->count; i++) {
        matrix->rows[i].triangle_id = chain->triangles[i].id;
        matrix->rows[i].word_a = chain->triangles[i].word_ids[0];
        matrix->rows[i].word_b = chain->triangles[i].word_ids[1];
        matrix->rows[i].word_c = chain->triangles[i].word_ids[2];
        matrix->rows[i].sum = matrix->rows[i].word_a + matrix->rows[i].word_b + matrix->rows[i].word_c;
        matrix->grand_total += matrix->rows[i].sum;
    }

    return matrix;
}

void matrix_free(MatrixTable *matrix) {
    if (!matrix) return;
    free(matrix->rows);
    free(matrix);
}

void matrix_print(const MatrixTable *matrix) {
    if (!matrix) return;

    printf("\n=== Matrix Table ===\n");
    printf("%-12s %-10s %-10s %-10s %-10s\n", "Triangle", "A", "B", "C", "Sum");
    printf("----------------------------------------------------\n");

    for (size_t i = 0; i < matrix->count; i++) {
        printf("%-12d %-10d %-10d %-10d %-10.0f\n",
               matrix->rows[i].triangle_id,
               matrix->rows[i].word_a,
               matrix->rows[i].word_b,
               matrix->rows[i].word_c,
               matrix->rows[i].sum);
    }

    printf("----------------------------------------------------\n");
    printf("%-12s %-10s %-10s %-10s %-10.0f\n", "TOTAL", "", "", "", matrix->grand_total);
}

void matrix_save_csv(const MatrixTable *matrix, const char *filename) {
    if (!matrix || !filename) return;

    FILE *f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "Triangle,A,B,C,Sum\n");
    for (size_t i = 0; i < matrix->count; i++) {
        fprintf(f, "%d,%d,%d,%d,%.0f\n",
                matrix->rows[i].triangle_id,
                matrix->rows[i].word_a,
                matrix->rows[i].word_b,
                matrix->rows[i].word_c,
                matrix->rows[i].sum);
    }
    fprintf(f, "TOTAL,,,,%.0f\n", matrix->grand_total);

    fclose(f);
    printf("Saved to %s\n", filename);
}
