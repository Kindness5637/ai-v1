#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "triangle.h"
#include <math.h>

typedef struct {
    double x;
    double y;
} Point2D;

typedef struct {
    int triangle_id;
    Point2D vertices[3];
    Point2D center;
} TriangleGeometry;

typedef struct {
    TriangleGeometry *geometries;
    size_t count;
} TriangleGeometries;

typedef struct {
    int from_id;
    int to_id;
    double angle_degrees;
    Point2D connection_point;
} TriangleRotation;

typedef struct {
    TriangleRotation *rotations;
    size_t count;
} TriangleRotations;

TriangleGeometries *geom_create(const TriangleChain *chain);
void geom_free(TriangleGeometries *geom);
void geom_print(const TriangleGeometries *geom);

TriangleRotations *rotations_create(const TriangleGeometries *geom);
void rotations_free(TriangleRotations *rot);
void rotations_print(const TriangleRotations *rot);

#endif
