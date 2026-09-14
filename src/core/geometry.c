#include "geometry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEG_TO_RAD (M_PI / 180.0)
#define RAD_TO_DEG (180.0 / M_PI)
#define TRIANGLE_SIZE 2.0
#define TRIANGLE_SPACING 4.0

TriangleGeometries *geom_create(const TriangleChain *chain) {
    if (!chain) return NULL;

    TriangleGeometries *geom = malloc(sizeof(TriangleGeometries));
    if (!geom) return NULL;

    geom->count = chain->count;
    geom->geometries = malloc(geom->count * sizeof(TriangleGeometry));
    if (!geom->geometries) {
        free(geom);
        return NULL;
    }

    for (size_t i = 0; i < chain->count; i++) {
        geom->geometries[i].triangle_id = chain->triangles[i].id;

        double base_x = i * TRIANGLE_SPACING;
        double base_y = 0;

        geom->geometries[i].vertices[0].x = base_x;
        geom->geometries[i].vertices[0].y = base_y + TRIANGLE_SIZE;

        geom->geometries[i].vertices[1].x = base_x - TRIANGLE_SIZE;
        geom->geometries[i].vertices[1].y = base_y;

        geom->geometries[i].vertices[2].x = base_x + TRIANGLE_SIZE;
        geom->geometries[i].vertices[2].y = base_y;

        geom->geometries[i].center.x = base_x;
        geom->geometries[i].center.y = base_y + TRIANGLE_SIZE / 3.0;
    }

    return geom;
}

void geom_free(TriangleGeometries *geom) {
    if (!geom) return;
    free(geom->geometries);
    free(geom);
}

void geom_print(const TriangleGeometries *geom) {
    if (!geom) return;

    printf("\n=== Triangle Geometry ===\n");
    for (size_t i = 0; i < geom->count; i++) {
        printf("\nTriangle %d:\n", geom->geometries[i].triangle_id);
        printf("  A: (%.1f, %.1f)\n", geom->geometries[i].vertices[0].x, geom->geometries[i].vertices[0].y);
        printf("  B: (%.1f, %.1f)\n", geom->geometries[i].vertices[1].x, geom->geometries[i].vertices[1].y);
        printf("  C: (%.1f, %.1f)\n", geom->geometries[i].vertices[2].x, geom->geometries[i].vertices[2].y);
        printf("  Center: (%.1f, %.1f)\n", geom->geometries[i].center.x, geom->geometries[i].center.y);
    }
}

TriangleRotations *rotations_create(const TriangleGeometries *geom) {
    if (!geom || geom->count < 2) return NULL;

    TriangleRotations *rot = malloc(sizeof(TriangleRotations));
    if (!rot) return NULL;

    rot->count = geom->count - 1;
    rot->rotations = malloc(rot->count * sizeof(TriangleRotation));
    if (!rot->rotations) {
        free(rot);
        return NULL;
    }

    for (size_t i = 0; i < rot->count; i++) {
        rot->rotations[i].from_id = geom->geometries[i].triangle_id;
        rot->rotations[i].to_id = geom->geometries[i + 1].triangle_id;

        double dx = geom->geometries[i + 1].center.x - geom->geometries[i].center.x;
        double dy = geom->geometries[i + 1].center.y - geom->geometries[i].center.y;

        rot->rotations[i].angle_degrees = atan2(dy, dx) * RAD_TO_DEG;

        rot->rotations[i].connection_point.x = (geom->geometries[i].vertices[2].x + geom->geometries[i + 1].vertices[1].x) / 2.0;
        rot->rotations[i].connection_point.y = (geom->geometries[i].vertices[2].y + geom->geometries[i + 1].vertices[1].y) / 2.0;
    }

    return rot;
}

void rotations_free(TriangleRotations *rot) {
    if (!rot) return;
    free(rot->rotations);
    free(rot);
}

void rotations_print(const TriangleRotations *rot) {
    if (!rot) return;

    printf("\n=== Triangle Rotations ===\n");
    printf("%-12s %-12s %-15s %-20s\n", "From", "To", "Angle", "Connection");
    printf("--------------------------------------------------------------\n");

    for (size_t i = 0; i < rot->count; i++) {
        printf("T%-11d T%-11d %7.1f°        (%.1f, %.1f)\n",
               rot->rotations[i].from_id,
               rot->rotations[i].to_id,
               rot->rotations[i].angle_degrees,
               rot->rotations[i].connection_point.x,
               rot->rotations[i].connection_point.y);
    }
}
