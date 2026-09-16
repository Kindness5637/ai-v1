/* Known-answer tests for the role/word entropy diagnostic.
 *
 * Why: H(role|word) is the number that decides whether the role channel
 * carries information beyond the word identity. If it is computed wrongly,
 * that decision is wrong -- so this test uses a hand-computable fixture
 * rather than a smoke check.
 *
 * Fixture: two 3-token sentences -> exactly one triangle each, so every
 * token is counted exactly once (overlapping windows would inflate counts).
 *
 *   sent1: w(NOUN) x(NOUN) y(NOUN)
 *   sent2: w(VERB) x(NOUN) z(NOUN)
 *
 * Tagged tokens: 6.  Distinct words: 4.
 *   w: 1 NOUN + 1 VERB  -> H(role|w) = 1.0 bit          (maximally ambiguous)
 *   x: 2 NOUN           -> 0                            (deterministic)
 *   y: 1 NOUN           -> 0
 *   z: 1 NOUN           -> 0
 *   marginal: 5 NOUN + 1 VERB
 *     H(role)      = -(5/6)log2(5/6) - (1/6)log2(1/6) = 0.6500224
 *   token-weighted H(role|word) = (2/6)(1.0) + 0      = 0.3333333
 *   word-weighted  H(role|word) = (1.0 + 0 + 0 + 0)/4 = 0.25
 *   mutual information = 0.6500224 - 0.3333333         = 0.3166891
 *
 * Build & run: make test
 */
#include "core/triangle.h"
#include "core/word.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

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

#define CHECK_NEAR(actual, expected, tol, msg) do {                            \
    checks++;                                                                  \
    double _a = (actual), _e = (expected);                                     \
    if (fabs(_a - _e) > (tol)) {                                               \
        failures++;                                                            \
        printf("FAIL: %s (got %.7f, want %.7f)\n", (msg), _a, _e);             \
    } else {                                                                   \
        printf("ok  : %s (%.7f)\n", (msg), _a);                                \
    }                                                                          \
} while (0)

/* Shared fixture. Note the trailing blank line closes each sentence. */
static const char *FIXTURE =
    "# sent_id = 1\n"
    "1\tw\tw\tNOUN\t_\t_\t0\troot\t_\t_\n"
    "2\tx\tx\tNOUN\t_\t_\t0\troot\t_\t_\n"
    "3\ty\ty\tNOUN\t_\t_\t0\troot\t_\t_\n"
    "\n"
    "# sent_id = 2\n"
    "1\tw\tw\tVERB\t_\t_\t0\troot\t_\t_\n"
    "2\tx\tx\tNOUN\t_\t_\t0\troot\t_\t_\n"
    "3\tz\tz\tNOUN\t_\t_\t0\troot\t_\t_\n"
    "\n";

int main(void) {
    /* Adversarial: NULL out / NULL chain must not crash or write garbage. */
    CHECK(triangle_role_entropy(NULL, NULL) == -1, "NULL out-param -> -1");
    RoleEntropyStats bad;
    memset(&bad, 0xAB, sizeof(bad));
    CHECK(triangle_role_entropy(NULL, &bad) == -1, "NULL chain -> -1");

    TriangleChain *chain = create_triangles_from_conllu(FIXTURE);
    CHECK(chain != NULL, "fixture parses");
    if (!chain) {
        printf("\n%d checks, %d failure(s)\n", checks, failures);
        return 1;
    }

    /* The fixture must produce exactly 2 triangles, or the counts below are
       not the counts we hand-derived. Guard that first. */
    CHECK(chain->count == 2, "fixture yields exactly 2 triangles");

    RoleEntropyStats st;
    int rc = triangle_role_entropy(chain, &st);
    CHECK(rc == 0, "entropy computation succeeds");

    CHECK(st.tokens == 6, "counts 6 tagged tokens");
    CHECK(st.distinct_words == 4, "counts 4 distinct words");
    CHECK(st.deterministic_words == 3, "w is the only ambiguous word (3 deterministic)");
    CHECK(st.ambiguous_tokens == 2, "2 tokens belong to ambiguous words");
    CHECK(strcmp(st.top_ambiguous_word, "w") == 0,
          "most ambiguous word is 'w'");
    CHECK_NEAR(st.top_ambiguous_entropy_bits, 1.0, 1e-9,
               "H(role|w) for a 50/50 word is exactly 1 bit");

    CHECK_NEAR(st.marginal_entropy_bits, 0.6500224, 1e-6,
               "marginal H(role) matches hand computation");
    CHECK_NEAR(st.cond_entropy_bits, 0.3333333, 1e-6,
               "token-weighted H(role|word) matches hand computation");
    CHECK_NEAR(st.cond_entropy_per_word_bits, 0.25, 1e-9,
               "word-weighted H(role|word) matches hand computation");
    CHECK_NEAR(st.mutual_info_bits, 0.3166891, 1e-6,
               "mutual information H(role) - H(role|word) matches");

    /* Sanity invariants that must hold for ANY chain. */
    CHECK(st.cond_entropy_bits <= st.marginal_entropy_bits + 1e-12,
          "H(role|word) never exceeds H(role)");
    CHECK(st.mutual_info_bits >= -1e-12, "mutual information is non-negative");

    free_triangles(chain);

    /* Holding tokens (role 0, i.e. plain text with no UPOS) must be excluded
       entirely rather than counted as a role. */
    TriangleChain *plain = create_triangles("the library closes at five");
    CHECK(plain != NULL, "plain-text chain builds");
    if (plain) {
        RoleEntropyStats pst;
        CHECK(triangle_role_entropy(plain, &pst) == 0,
              "plain-text chain computes without error");
        CHECK(pst.tokens == 0,
              "plain text has no tagged tokens (roles are 0 = holding)");
        CHECK_NEAR(pst.marginal_entropy_bits, 0.0, 1e-12,
                   "empty input yields zero entropy, not NaN");
        CHECK_NEAR(pst.cond_entropy_bits, 0.0, 1e-12,
                   "empty input yields zero conditional entropy");
        free_triangles(plain);
    }

    printf("\n%d checks, %d failure(s)\n", checks, failures);
    return failures ? 1 : 0;
}
