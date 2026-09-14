#ifndef CIRCLE_H
#define CIRCLE_H

#include "triangle.h"

typedef struct {
    int id;
    int word_id;
    char *word;
    int before_triangle_id;
    int after_triangle_id;
    double value;
} Circle;

typedef struct {
    Circle *circles;
    size_t count;
    double multiplier;
} CircleChain;

CircleChain *circle_create(const TriangleChain *chain, double multiplier);
void circle_free(CircleChain *circles);
void circle_print(const CircleChain *circles);
double circle_calculate_impact(const CircleChain *circles, const TriangleChain *chain);

#endif
