#ifndef FORMULA_H
#define FORMULA_H

#include "triangle.h"

typedef struct {
    double sum_a;
    double sum_b;
    double sum_c;
    double total;
    int power;
} FormulaResult;

FormulaResult *formula_calculate(const TriangleChain *chain, int power);
void formula_free(FormulaResult *result);
void formula_print(const FormulaResult *result);

#endif
