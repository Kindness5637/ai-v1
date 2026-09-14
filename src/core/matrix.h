#ifndef MATRIX_TABLE_H
#define MATRIX_TABLE_H

#include "triangle.h"

typedef struct {
    int triangle_id;
    int word_a;
    int word_b;
    int word_c;
    double sum;
} TriangleRow;

typedef struct {
    TriangleRow *rows;
    size_t count;
    double grand_total;
} MatrixTable;

MatrixTable *matrix_create(const TriangleChain *chain);
void matrix_free(MatrixTable *matrix);
void matrix_print(const MatrixTable *matrix);
void matrix_save_csv(const MatrixTable *matrix, const char *filename);

#endif
