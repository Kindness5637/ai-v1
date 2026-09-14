#include "rotation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

TriangleChain_Links *chain_links_create(void) {
    TriangleChain_Links *links = malloc(sizeof(TriangleChain_Links));
    if (!links) return NULL;
    links->edges = NULL;
    links->count = 0;
    links->capacity = 0;
    return links;
}

void chain_links_free(TriangleChain_Links *links) {
    if (!links) return;
    free(links->edges);
    free(links);
}

void chain_links_build(TriangleChain_Links *links, const TriangleChain *chain) {
    if (!links || !chain || chain->count < 2) return;

    for (size_t i = 0; i < chain->count - 1; i++) {
        int c_word_id = chain->triangles[i].word_ids[2];
        int next_a_word_id = chain->triangles[i + 1].word_ids[0];

        if (c_word_id == next_a_word_id && c_word_id != 0) {
            if (links->count >= links->capacity) {
                links->capacity = links->capacity == 0 ? 16 : links->capacity * 2;
                TriangleEdge *temp = realloc(links->edges, links->capacity * sizeof(TriangleEdge));
                if (!temp) return;
                links->edges = temp;
            }

            links->edges[links->count].from_triangle_id = chain->triangles[i].id;
            links->edges[links->count].to_triangle_id = chain->triangles[i + 1].id;
            links->edges[links->count].shared_word_id = c_word_id;
            links->edges[links->count].angle = 60.0;
            links->count++;
        }
    }
}

void chain_links_print(const TriangleChain_Links *links) {
    if (!links) return;

    printf("\n=== Triangle Chain Links ===\n");
    printf("%-12s %-12s %-12s %-10s\n", "From", "To", "Shared Word", "Angle");
    printf("--------------------------------------------------\n");

    for (size_t i = 0; i < links->count; i++) {
        printf("T%-11d T%-11d WordID:%-6d %.1f°\n",
               links->edges[i].from_triangle_id,
               links->edges[i].to_triangle_id,
               links->edges[i].shared_word_id,
               links->edges[i].angle);
    }

    if (links->count == 0) {
        printf("No direct links found (C of triangle != A of next triangle)\n");
    }
}
