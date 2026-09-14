#ifndef ROTATION_H
#define ROTATION_H

#include "triangle.h"
#include <math.h>

typedef struct {
    int from_triangle_id;
    int to_triangle_id;
    int shared_word_id;
    double angle;
} TriangleEdge;

typedef struct {
    TriangleEdge *edges;
    size_t count;
    size_t capacity;
} TriangleChain_Links;

TriangleChain_Links *chain_links_create(void);
void chain_links_free(TriangleChain_Links *links);
void chain_links_build(TriangleChain_Links *links, const TriangleChain *chain);
void chain_links_print(const TriangleChain_Links *links);

#endif
