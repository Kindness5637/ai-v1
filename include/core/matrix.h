#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
    size_t rows;
    size_t cols;
    double *data;
} Matrix;

Matrix *matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix *m);
Matrix *matrix_copy(const Matrix *m);
void matrix_print(const Matrix *m);

Matrix *matrix_add(const Matrix *a, const Matrix *b);
Matrix *matrix_sub(const Matrix *a, const Matrix *b);
Matrix *matrix_mul(const Matrix *a, const Matrix *b);
Matrix *matrix_scale(const Matrix *m, double scalar);
Matrix *matrix_transpose(const Matrix *m);

Matrix *matrix_dot(const Matrix *a, const Matrix *b);
double matrix_sum(const Matrix *m);
double matrix_mean(const Matrix *m);

int matrix_save(const Matrix *m, const char *filename);
Matrix *matrix_load(const char *filename);

#endif
