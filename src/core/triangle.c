#include "triangle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

static char **tokenize(const char *sentence, size_t *word_count) {
    char *copy = strdup(sentence);
    if (!copy) return NULL;

    size_t capacity = 16;
    char **words = malloc(capacity * sizeof(char *));
    if (!words) {
        free(copy);
        return NULL;
    }

    *word_count = 0;
    char *token = strtok(copy, " ,.!?;:\n");
    while (token) {
        if (*word_count >= capacity) {
            capacity *= 2;
            char **temp = realloc(words, capacity * sizeof(char *));
            if (!temp) {
                for (size_t i = 0; i < *word_count; i++) free(words[i]);
                free(words);
                free(copy);
                return NULL;
            }
            words = temp;
        }
        words[*word_count] = strdup(token);
        (*word_count)++;
        token = strtok(NULL, " ,.!?;:\n");
    }

    free(copy);
    return words;
}

static char *str_to_lower(const char *str) {
    char *lower = malloc(strlen(str) + 1);
    if (!lower) return NULL;
    for (size_t i = 0; str[i]; i++) {
        lower[i] = tolower((unsigned char)str[i]);
    }
    lower[strlen(str)] = '\0';
    return lower;
}

int triangle_role_id(const char *upos) {
    static const char *roles[] = {
        "", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
        "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ",
        "SYM", "VERB", "X"
    };
    if (!upos) return 0;
    for (int i = 1; i < (int)(sizeof(roles) / sizeof(roles[0])); i++) {
        if (strcmp(upos, roles[i]) == 0) return i;
    }
    return 0;
}

/* Shannon entropy in bits of a count distribution over `n` bins. */
static double entropy_bits(const size_t *counts, size_t n, size_t total) {
    if (total == 0) return 0.0;
    double h = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (counts[i] == 0) continue;
        double p = (double)counts[i] / (double)total;
        h -= p * (log(p) / log(2.0));
    }
    return h;
}

int triangle_role_entropy(const TriangleChain *chain, RoleEntropyStats *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!chain || !chain->vocab) return -1;

    size_t vocab_size = chain->vocab->count;
    if (vocab_size == 0) return -1;

    size_t cells = (vocab_size + 1) * TRIANGLE_ROLE_FEATURE_DIM;
    size_t *joint = (size_t *)calloc(cells, sizeof(size_t));
    size_t *per_word = (size_t *)calloc(vocab_size + 1, sizeof(size_t));
    size_t *per_role = (size_t *)calloc(TRIANGLE_ROLE_FEATURE_DIM, sizeof(size_t));
    if (!joint || !per_word || !per_role) {
        fprintf(stderr, "triangle_role_entropy: allocation failed\n");
        free(joint);
        free(per_word);
        free(per_role);
        return -1;
    }

    size_t tokens = 0;
    for (size_t t = 0; t < chain->count; t++) {
        for (int p = 0; p < 3; p++) {
            int word = chain->triangles[t].word_ids[p];
            int role = chain->triangles[t].role_ids[p];
            if (word <= 0 || word > (int)vocab_size) continue;
            if (role <= 0 || role >= TRIANGLE_ROLE_FEATURE_DIM) continue; /* holding */
            joint[(size_t)word * TRIANGLE_ROLE_FEATURE_DIM + (size_t)role]++;
            per_word[word]++;
            per_role[role]++;
            tokens++;
        }
    }
    out->tokens = tokens;
    if (tokens == 0) {
        free(joint);
        free(per_word);
        free(per_role);
        return 0;
    }

    out->marginal_entropy_bits =
        entropy_bits(per_role, TRIANGLE_ROLE_FEATURE_DIM, tokens);

    for (size_t w = 1; w <= vocab_size; w++) {
        if (per_word[w] == 0) continue;
        const size_t *row = &joint[w * TRIANGLE_ROLE_FEATURE_DIM];
        double h_w = entropy_bits(row, TRIANGLE_ROLE_FEATURE_DIM, per_word[w]);
        size_t roles_taken = 0;
        for (int r = 0; r < TRIANGLE_ROLE_FEATURE_DIM; r++) {
            if (row[r] > 0) roles_taken++;
        }
        out->distinct_words++;
        if (roles_taken == 1) out->deterministic_words++;
        else out->ambiguous_tokens += per_word[w];
        out->cond_entropy_bits += ((double)per_word[w] / (double)tokens) * h_w;
        out->cond_entropy_per_word_bits += h_w;
        if (h_w > out->top_ambiguous_entropy_bits) {
            out->top_ambiguous_entropy_bits = h_w;
            out->top_ambiguous_word_id = (int)w;
            const char *text = vocab_get_word(chain->vocab, (int)w);
            if (text) {
                strncpy(out->top_ambiguous_word, text,
                        sizeof(out->top_ambiguous_word) - 1);
                out->top_ambiguous_word[sizeof(out->top_ambiguous_word) - 1] = '\0';
            }
        }
    }

    if (out->distinct_words > 0) {
        out->cond_entropy_per_word_bits /= (double)out->distinct_words;
    }
    out->mutual_info_bits =
        out->marginal_entropy_bits - out->cond_entropy_bits;

    free(joint);
    free(per_word);
    free(per_role);
    return 0;
}

TriangleChain *create_triangles(const char *sentence) {
    size_t word_count = 0;
    char **words = tokenize(sentence, &word_count);
    if (!words || word_count == 0) return NULL;

    TriangleChain *chain = malloc(sizeof(TriangleChain));
    if (!chain) {
        for (size_t i = 0; i < word_count; i++) free(words[i]);
        free(words);
        return NULL;
    }

    chain->vocab = vocab_create();
    chain->registry = registry_create();
    chain->owns_vocab = 1;
    if (!chain->vocab || !chain->registry) {
        vocab_free(chain->vocab);
        registry_free(chain->registry);
        free(chain);
        for (size_t i = 0; i < word_count; i++) free(words[i]);
        free(words);
        return NULL;
    }

    /* Use overlapping sliding windows.  A sequence A B C D produces
     * (A,B,C) and (B,C,D), rather than two unrelated chunks. */
    size_t tri_count = word_count >= 3 ? word_count - 2 : 0;
    chain->triangles = malloc(tri_count * sizeof(Triangle));
    if (!chain->triangles) {
        vocab_free(chain->vocab);
        registry_free(chain->registry);
        free(chain);
        for (size_t i = 0; i < word_count; i++) free(words[i]);
        free(words);
        return NULL;
    }

    chain->count = tri_count;
    size_t word_idx = 0;

    for (size_t i = 0; i < tri_count; i++) {
        chain->triangles[i].id = i + 1;
        for (size_t j = 0; j < 3; j++) {
            word_idx = i + j;
            if (word_idx < word_count) {
                char *lower = str_to_lower(words[word_idx]);
                chain->triangles[i].words[j] = lower;
                int word_id = vocab_get_or_add(chain->vocab, lower);
                chain->triangles[i].word_ids[j] = word_id;
                chain->triangles[i].role_ids[j] = 0;
                registry_add(chain->registry, word_id, i + 1, j, lower);
            } else {
                chain->triangles[i].words[j] = strdup("-");
                chain->triangles[i].word_ids[j] = 0;
                chain->triangles[i].role_ids[j] = 0;
            }
        }
    }

    for (size_t i = 0; i < word_count; i++) free(words[i]);
    free(words);

    return chain;
}

TriangleChain *create_triangles_with_vocab(const char *sentence, Vocabulary *vocab) {
    if (!vocab) return NULL;
    TriangleChain *chain = create_triangles(sentence);
    if (!chain) return NULL;

    Vocabulary *local_vocab = chain->vocab;
    for (size_t i = 0; i < chain->count; i++) {
        for (int p = 0; p < 3; p++) {
            if (chain->triangles[i].word_ids[p] <= 0) continue;
            chain->triangles[i].word_ids[p] =
                vocab_get_or_add(vocab, chain->triangles[i].words[p]);
        }
    }
    for (size_t i = 0; i < chain->registry->count; i++) {
        chain->registry->entries[i].word_id =
            vocab_get_or_add(vocab, chain->registry->entries[i].word);
    }
    vocab_free(local_vocab);
    chain->vocab = vocab;
    chain->owns_vocab = 0;
    return chain;
}

typedef struct {
    char *form;
    char upos[16];
    char deprel[32];
} ConlluToken;

static void free_conllu_tokens(ConlluToken *tokens, size_t count) {
    if (!tokens) return;
    for (size_t i = 0; i < count; i++) free(tokens[i].form);
    free(tokens);
}

static int append_conllu_sentence(TriangleChain *chain,
                                  ConlluToken *tokens, size_t token_count) {
    if (!chain || token_count < 3) return 1;

    /* Preserve sentence boundaries while using overlapping windows. */
    size_t triangle_count = token_count - 2;
    Triangle *grown = realloc(chain->triangles,
                              (chain->count + triangle_count) * sizeof(Triangle));
    if (!grown) return 0;
    chain->triangles = grown;

    for (size_t i = 0; i < triangle_count; i++) {
        Triangle *triangle = &chain->triangles[chain->count + i];
        triangle->id = (int)(chain->count + i + 1);
        for (int p = 0; p < 3; p++) {
            size_t token_index = i + (size_t)p;
            if (token_index < token_count) {
                char *lower = str_to_lower(tokens[token_index].form);
                if (!lower) return 0;
                triangle->words[p] = lower;
                triangle->word_ids[p] = vocab_get_or_add(chain->vocab, lower);
                triangle->role_ids[p] = triangle_role_id(tokens[token_index].upos);
                registry_add(chain->registry, triangle->word_ids[p],
                             (size_t)triangle->id, p, lower);
                strncpy(triangle->upos[p], tokens[token_index].upos,
                        sizeof(triangle->upos[p]) - 1);
                triangle->upos[p][sizeof(triangle->upos[p]) - 1] = '\0';
                strncpy(triangle->deprel[p], tokens[token_index].deprel,
                        sizeof(triangle->deprel[p]) - 1);
                triangle->deprel[p][sizeof(triangle->deprel[p]) - 1] = '\0';
            } else {
                triangle->words[p] = strdup("-");
                if (!triangle->words[p]) return 0;
                triangle->word_ids[p] = 0;
                triangle->role_ids[p] = 0;
                triangle->upos[p][0] = '\0';
                triangle->deprel[p][0] = '\0';
            }
        }
    }
    chain->count += triangle_count;
    return 1;
}

TriangleChain *create_triangles_from_conllu(const char *content) {
    if (!content) return NULL;

    TriangleChain *chain = calloc(1, sizeof(TriangleChain));
    if (!chain) return NULL;
    chain->vocab = vocab_create();
    chain->registry = registry_create();
    chain->owns_vocab = 1;
    if (!chain->vocab || !chain->registry) {
        vocab_free(chain->vocab);
        registry_free(chain->registry);
        free(chain);
        return NULL;
    }

    char *copy = strdup(content);
    if (!copy) {
        free_triangles(chain);
        return NULL;
    }

    ConlluToken *tokens = NULL;
    size_t token_count = 0, token_capacity = 0;
    char *line_save = NULL;
    char *line = strtok_r(copy, "\n", &line_save);
    while (line) {
        while (*line == '\r') line++;
        if (*line == '\0') {
            if (!append_conllu_sentence(chain, tokens, token_count)) {
                free_conllu_tokens(tokens, token_count);
                free(copy);
                free_triangles(chain);
                return NULL;
            }
            free_conllu_tokens(tokens, token_count);
            tokens = NULL;
            token_count = 0;
            token_capacity = 0;
            line = strtok_r(NULL, "\n", &line_save);
            continue;
        }
        if (*line == '#') {
            if (strncmp(line, "# sent_id", 9) == 0 && token_count > 0) {
                if (!append_conllu_sentence(chain, tokens, token_count)) {
                    free_conllu_tokens(tokens, token_count);
                    free(copy);
                    free_triangles(chain);
                    return NULL;
                }
                free_conllu_tokens(tokens, token_count);
                tokens = NULL;
                token_count = 0;
                token_capacity = 0;
            }
            line = strtok_r(NULL, "\n", &line_save);
            continue;
        }

        char *fields[10] = {0};
        char *field_save = NULL;
        char *field = strtok_r(line, "\t", &field_save);
        int field_count = 0;
        while (field && field_count < 10) {
            fields[field_count++] = field;
            field = strtok_r(NULL, "\t", &field_save);
        }
        /* Skip multi-word-token and empty-node rows; only syntactic word
         * rows receive triangle vertices. */
        if (field_count == 10 && strchr(fields[0], '-') == NULL &&
            strchr(fields[0], '.') == NULL) {
            if (token_count == token_capacity) {
                size_t next_capacity = token_capacity ? token_capacity * 2 : 64;
                ConlluToken *grown = realloc(tokens,
                                             next_capacity * sizeof(ConlluToken));
                if (!grown) {
                    free_conllu_tokens(tokens, token_count);
                    free(copy);
                    free_triangles(chain);
                    return NULL;
                }
                tokens = grown;
                token_capacity = next_capacity;
            }
            tokens[token_count].form = strdup(fields[1]);
            if (!tokens[token_count].form) {
                free_conllu_tokens(tokens, token_count);
                free(copy);
                free_triangles(chain);
                return NULL;
            }
            strncpy(tokens[token_count].upos, fields[3],
                    sizeof(tokens[token_count].upos) - 1);
            tokens[token_count].upos[sizeof(tokens[token_count].upos) - 1] = '\0';
            strncpy(tokens[token_count].deprel, fields[7],
                    sizeof(tokens[token_count].deprel) - 1);
            tokens[token_count].deprel[sizeof(tokens[token_count].deprel) - 1] = '\0';
            token_count++;
        }
        line = strtok_r(NULL, "\n", &line_save);
    }

    if (!append_conllu_sentence(chain, tokens, token_count)) {
        free_conllu_tokens(tokens, token_count);
        free(copy);
        free_triangles(chain);
        return NULL;
    }
    free_conllu_tokens(tokens, token_count);
    free(copy);
    return chain;
}

TriangleChain *create_triangles_from_conllu_with_vocab(const char *content,
                                                       Vocabulary *vocab) {
    if (!vocab) return NULL;
    TriangleChain *chain = create_triangles_from_conllu(content);
    if (!chain) return NULL;

    Vocabulary *local_vocab = chain->vocab;
    for (size_t i = 0; i < chain->count; i++) {
        for (int p = 0; p < 3; p++) {
            if (chain->triangles[i].word_ids[p] <= 0) continue;
            chain->triangles[i].word_ids[p] =
                vocab_get_or_add(vocab, chain->triangles[i].words[p]);
        }
    }
    for (size_t i = 0; i < chain->registry->count; i++) {
        chain->registry->entries[i].word_id =
            vocab_get_or_add(vocab, chain->registry->entries[i].word);
    }
    vocab_free(local_vocab);
    chain->vocab = vocab;
    chain->owns_vocab = 0;
    return chain;
}

void free_triangles(TriangleChain *chain) {
    if (!chain) return;
    for (size_t i = 0; i < chain->count; i++) {
        for (size_t j = 0; j < 3; j++) {
            free(chain->triangles[i].words[j]);
        }
    }
    free(chain->triangles);
    if (chain->owns_vocab) vocab_free(chain->vocab);
    registry_free(chain->registry);
    free(chain);
}

void print_triangles(const TriangleChain *chain) {
    if (!chain) return;

    for (size_t i = 0; i < chain->count; i++) {
        printf("\nTriangle [ID:%d]:\n", chain->triangles[i].id);
        printf("    %s (WordID:%d) (A)\n",
               chain->triangles[i].words[0],
               chain->triangles[i].word_ids[0]);
        printf("       /\\\n");
        printf("      /  \\\n");
        printf("     /    \\\n");
        printf("    /      \\\n");
        printf(" %s (WordID:%d) (B)---%s (WordID:%d) (C)\n",
               chain->triangles[i].words[1],
               chain->triangles[i].word_ids[1],
               chain->triangles[i].words[2],
               chain->triangles[i].word_ids[2]);
        if (chain->triangles[i].upos[0][0] != '\0') {
            printf("    Roles: %s/%s, %s/%s, %s/%s\n",
                   chain->triangles[i].upos[0], chain->triangles[i].deprel[0],
                   chain->triangles[i].upos[1], chain->triangles[i].deprel[1],
                   chain->triangles[i].upos[2], chain->triangles[i].deprel[2]);
        }
    }
}

void print_vocabulary(const TriangleChain *chain) {
    if (!chain || !chain->vocab) return;

    printf("\n=== Vocabulary ===\n");
    printf("%-8s %-15s\n", "WordID", "Word");
    printf("------------------------\n");
    for (size_t i = 0; i < chain->vocab->count; i++) {
        printf("%-8d %-15s\n",
               chain->vocab->words[i].id,
               chain->vocab->words[i].text);
    }
}

void print_registry(const TriangleChain *chain) {
    if (!chain || !chain->registry) return;
    registry_print(chain->registry);
}

RelationalRegistry *relational_registry_create(size_t initial_vocab_size) {
    RelationalRegistry *reg = calloc(1, sizeof(RelationalRegistry));
    if (!reg) return NULL;
    if (initial_vocab_size > 0) {
        relational_registry_ensure_vocab(reg, initial_vocab_size);
    }
    reg->transition_capacity = 256;
    reg->transitions = malloc(reg->transition_capacity * sizeof(RelationalTransition));
    if (!reg->transitions) {
        free(reg->word_stats);
        free(reg);
        return NULL;
    }
    return reg;
}

void relational_registry_free(RelationalRegistry *reg) {
    if (!reg) return;
    free(reg->word_stats);
    free(reg->transitions);
    free(reg);
}

void relational_registry_ensure_vocab(RelationalRegistry *reg, size_t vocab_size) {
    if (!reg || vocab_size <= reg->vocab_size) return;
    RelationalWordStats *new_stats = realloc(reg->word_stats, (vocab_size + 1) * sizeof(RelationalWordStats));
    if (!new_stats) return;
    memset(new_stats + reg->vocab_size + 1, 0, (vocab_size - reg->vocab_size) * sizeof(RelationalWordStats));
    reg->word_stats = new_stats;
    reg->vocab_size = vocab_size;
}

static void record_transition(RelationalRegistry *reg, int from_id, int to_id, int transition_type) {
    if (!reg || from_id <= 0 || to_id <= 0) return;
    for (size_t i = 0; i < reg->transition_count; i++) {
        if (reg->transitions[i].from_word_id == from_id &&
            reg->transitions[i].to_word_id == to_id &&
            reg->transitions[i].transition_type == transition_type) {
            reg->transitions[i].count++;
            return;
        }
    }
    if (reg->transition_count >= reg->transition_capacity) {
        size_t new_cap = reg->transition_capacity * 2;
        RelationalTransition *new_tr = realloc(reg->transitions, new_cap * sizeof(RelationalTransition));
        if (!new_tr) return;
        reg->transitions = new_tr;
        reg->transition_capacity = new_cap;
    }
    reg->transitions[reg->transition_count++] = (RelationalTransition){
        .from_word_id = from_id,
        .to_word_id = to_id,
        .transition_type = transition_type,
        .count = 1
    };
}

void relational_registry_ingest_chain(RelationalRegistry *reg, const TriangleChain *chain) {
    if (!reg || !chain || !chain->vocab) return;
    relational_registry_ensure_vocab(reg, chain->vocab->count);

    for (size_t i = 0; i < chain->count; i++) {
        const Triangle *t = &chain->triangles[i];
        int w0 = t->word_ids[0];
        int w1 = t->word_ids[1];
        int w2 = t->word_ids[2];

        /* Structural position — record ONCE per spatial triangle */
        if (w0 > 0 && (size_t)w0 <= reg->vocab_size) {
            reg->word_stats[w0].left_count++;
            reg->word_stats[w0].total_count++;
        }
        if (w1 > 0 && (size_t)w1 <= reg->vocab_size) {
            reg->word_stats[w1].center_count++;
            reg->word_stats[w1].total_count++;
        }
        if (w2 > 0 && (size_t)w2 <= reg->vocab_size) {
            reg->word_stats[w2].right_count++;
            reg->word_stats[w2].total_count++;
        }

        /* Forward relational transitions */
        record_transition(reg, w0, w1, 0); /* L -> C */
        record_transition(reg, w1, w2, 1); /* C -> R */

        /* Backward relational transitions */
        record_transition(reg, w2, w1, 2); /* R -> C */
        record_transition(reg, w1, w0, 3); /* C -> L */
    }


    /* Update asymmetry scores */
    for (size_t w = 1; w <= reg->vocab_size; w++) {
        uint64_t L = reg->word_stats[w].left_count;
        uint64_t R = reg->word_stats[w].right_count;
        if (L + R == 0) {
            reg->word_stats[w].asymmetry = 0.0;
        } else {
            reg->word_stats[w].asymmetry = (double)((int64_t)L - (int64_t)R) / (double)(L + R);
        }
    }
}

uint64_t relational_registry_get_transition_count(const RelationalRegistry *reg,
                                                  int from_id, int to_id,
                                                  int transition_type) {
    if (!reg) return 0;
    for (size_t i = 0; i < reg->transition_count; i++) {
        if (reg->transitions[i].from_word_id == from_id &&
            reg->transitions[i].to_word_id == to_id &&
            reg->transitions[i].transition_type == transition_type) {
            return reg->transitions[i].count;
        }
    }
    return 0;
}

void relational_registry_report(const RelationalRegistry *reg, const Vocabulary *vocab, size_t top_n) {
    if (!reg || !vocab) return;

    printf("\n=== DISCOVERED STRUCTURAL PATTERN HYPOTHESES (Unsupervised Mode B) ===\n");
    printf("%-5s %-15s %-10s %-8s %-8s %-8s %-10s\n",
           "ID", "Word", "Support", "Left", "Center", "Right", "Asymmetry");
    printf("-------------------------------------------------------------------------\n");

    size_t printed = 0;
    for (size_t w = 1; w <= reg->vocab_size && printed < top_n; w++) {
        uint64_t L = reg->word_stats[w].left_count;
        uint64_t R = reg->word_stats[w].right_count;
        uint64_t C = reg->word_stats[w].center_count;
        uint64_t support = L + R + C;

        if (support < 5) continue; /* Minimum evidence threshold for hypothesis reporting */

        const char *word_text = vocab_get_word(vocab, (int)w);
        printf("%-5d %-15s %-10llu %-8llu %-8llu %-8llu %-+10.3f\n",

               (int)w, word_text ? word_text : "?",
               (unsigned long long)support,
               (unsigned long long)L,
               (unsigned long long)C,
               (unsigned long long)R,
               reg->word_stats[w].asymmetry);
        printed++;
    }
    printf("================================================================---------\n");
}
