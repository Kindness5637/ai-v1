/* Adversarial tests for the UPOS role plumbing.
 *
 * Why these exist: evaluation now feeds real UPOS roles into the network
 * (they used to be hardcoded 0). If the tag mapping or the chain fill ever
 * silently degrades to 0, the experiment would read as "roles do not help"
 * when in fact no role ever reached the network. These tests try to break
 * that assumption rather than confirm it: NULL/empty/unknown/lowercase tags,
 * duplicate detection, range checking, the shared-vocab remap that training
 * uses, and short sentences that must not invent roles.
 *
 * Build & run (from the repo root):
 *     gcc -Wall -Wextra -Isrc -Iinclude -fopenmp -o /tmp/test_role_ids \
 *         tests/test_role_ids.c src/core/triangle.c src/core/word.c -lm
 *     /tmp/test_role_ids
 * or simply: make test
 */
#include "core/triangle.h"
#include "core/word.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do {                                                  \
    checks++;                                                                  \
    if (!(cond)) {                                                             \
        failures++;                                                            \
        printf("FAIL: %s\n", (msg));                                           \
    } else {                                                                   \
        printf("ok  : %s\n", (msg));                                           \
    }                                                                          \
} while (0)

/* The 17 Universal Dependencies UPOS tags, in the order used by
   triangle_role_id(). */
static const char *TAGS[] = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
};

/* Minimal but well-formed CoNLL-U: 3 tokens -> exactly 1 triangle.
   Columns: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC */
static const char *SAMPLE =
    "# sent_id = 1\n"
    "# text = The library closes\n"
    "1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\t_\n"
    "2\tlibrary\tlibrary\tNOUN\t_\t_\t3\tnsubj\t_\t_\n"
    "3\tcloses\tclose\tVERB\t_\t_\t0\troot\t_\t_\n"
    "\n";

int main(void) {
    /* -- 1. Adversarial inputs to the mapping itself ---------------------- */
    CHECK(triangle_role_id(NULL) == 0, "NULL tag -> 0 (holding), no crash");
    CHECK(triangle_role_id("") == 0, "empty tag -> 0 (holding)");
    CHECK(triangle_role_id("_") == 0, "CoNLL-U placeholder '_' -> 0 (holding)");
    CHECK(triangle_role_id("NOTATAG") == 0, "unknown tag -> 0 (holding)");
    /* Documented boundary, not a bug we want silent: UD writes tags in upper
       case, so a lower-cased file would make EVERY token holding. The
       evaluator prints "Eval roles: ... N holding" so this is visible
       in-run rather than silent. */
    CHECK(triangle_role_id("det") == 0,
          "lowercase 'det' -> 0: mapping is case-sensitive by design");

    /* -- 2. All tags distinct, in range, non-zero ------------------------- */
    int seen[TRIANGLE_ROLE_FEATURE_DIM];
    for (int i = 0; i < TRIANGLE_ROLE_FEATURE_DIM; i++) seen[i] = 0;
    int all_ok = 1;
    const char *bad = "";
    size_t ntags = sizeof(TAGS) / sizeof(TAGS[0]);
    for (size_t i = 0; i < ntags; i++) {
        int id = triangle_role_id(TAGS[i]);
        if (id <= 0 || id >= TRIANGLE_ROLE_FEATURE_DIM) {
            all_ok = 0; bad = TAGS[i]; break;
        }
        if (seen[id]) { all_ok = 0; bad = TAGS[i]; break; }  /* collision */
        seen[id] = 1;
    }
    if (!all_ok) printf("      (offending tag: %s)\n", bad);
    CHECK(all_ok,
          "all 17 UPOS tags map to distinct ids in 1..31 (in range, no collision)");
    CHECK(triangle_role_id("DET") == triangle_role_id("DET"),
          "mapping is deterministic");

    /* -- 3. The chain must actually carry the tags ------------------------ */
    TriangleChain *chain = create_triangles_from_conllu(SAMPLE);
    CHECK(chain != NULL, "3-token conllu sentence parses");
    if (chain) {
        CHECK(chain->count == 1, "3-token sentence yields exactly 1 triangle");
        if (chain->count == 1) {
            CHECK(chain->triangles[0].role_ids[0] == triangle_role_id("DET") &&
                  chain->triangles[0].role_ids[0] > 0,
                  "slot0 role id comes from the UPOS column (DET)");
            CHECK(chain->triangles[0].role_ids[1] == triangle_role_id("NOUN") &&
                  chain->triangles[0].role_ids[1] > 0,
                  "slot1 role id comes from the UPOS column (NOUN)");
            CHECK(chain->triangles[0].role_ids[2] == triangle_role_id("VERB") &&
                  chain->triangles[0].role_ids[2] > 0,
                  "slot2 role id comes from the UPOS column (VERB)");
            /* Distinct tags must give distinct ids: this catches a mutant
               that maps everything to one constant (where the equality
               checks above would still pass vacuously). */
            CHECK(chain->triangles[0].role_ids[0] != chain->triangles[0].role_ids[1] &&
                  chain->triangles[0].role_ids[1] != chain->triangles[0].role_ids[2],
                  "DET/NOUN/VERB produce pairwise-distinct role ids");
            CHECK(strcmp(chain->triangles[0].upos[0], "DET") == 0,
                  "raw upos string preserved next to the numeric id");
            CHECK(strcmp(chain->triangles[0].deprel[1], "nsubj") == 0,
                  "deprel parsed from column 8");
            /* Break-test: if every role were 0 the whole change is inert. */
            CHECK(chain->triangles[0].role_ids[0] != 0 &&
                  chain->triangles[0].role_ids[1] != 0 &&
                  chain->triangles[0].role_ids[2] != 0,
                  "no slot silently fell back to holding (role 0)");
        }
        free_triangles(chain);
    }

    /* -- 4. Training path: shared-vocab remap must not drop roles --------- */
    Vocabulary *vocab = vocab_create();
    CHECK(vocab != NULL, "shared vocabulary allocates");
    TriangleChain *shared = NULL;
    if (vocab) {
        shared = create_triangles_from_conllu_with_vocab(SAMPLE, vocab);
        CHECK(shared != NULL, "shared-vocab conllu chain builds");
        if (shared) {
            /* create_triangles_from_conllu_with_vocab rewrites word_ids only;
               roles must survive that rewrite, because this is what the
               training chain is built with. */
            CHECK(shared->triangles[0].role_ids[0] == triangle_role_id("DET") &&
                  shared->triangles[0].role_ids[0] > 0,
                  "role_ids survive the shared-vocab remap used for training");
            CHECK(shared->triangles[0].word_ids[0] > 0,
                  "word_ids were remapped into the shared vocabulary");
        }
    }

    /* -- 5. Boundary: too few tokens must not invent triangles/roles ------ */
    TriangleChain *tiny = create_triangles_from_conllu(
        "# sent_id = 2\n"
        "1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\t_\n"
        "2\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_\n"
        "\n");
    CHECK(tiny == NULL || tiny->count == 0,
          "2-token sentence yields no triangles (nothing to read roles from)");
    if (tiny) free_triangles(tiny);

    /* -- 6. Empty / malformed input must not crash ------------------------ */
    TriangleChain *empty = create_triangles_from_conllu("");
    CHECK(empty == NULL || empty->count == 0, "empty input yields no triangles");
    if (empty) free_triangles(empty);

    if (shared) free_triangles(shared);
    if (vocab) vocab_free(vocab);

    printf("\n%d checks, %d failure(s)\n", checks, failures);
    return failures ? 1 : 0;
}
