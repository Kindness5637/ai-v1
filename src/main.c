#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core/triangle.h"
#include "core/probability.h"
#include "core/neural.h"
#include "core/formula.h"
#include "core/matrix.h"
#include "core/reconstruct.h"
#include "core/circle.h"
#include "core/graph.h"
#include "core/context_graph.h"
#include "core/learn.h"
#include "core/punctuation.h"
#include "core/backprop.h"

char *read_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) return NULL;

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    char *content = malloc(size + 1);
    if (!content) {
        fclose(file);
        return NULL;
    }

    fread(content, 1, size, file);
    content[size] = '\0';
    fclose(file);

    return content;
}

void print_usage(void) {
    printf("Usage:\n");
    printf("  ./triangle.out <file>              - Process file (first pass)\n");
    printf("  ./triangle.out <file> -train       - Train neural network\n");
    printf("  ./triangle.out <file> -eval        - Evaluate predictions\n");
    printf("  ./triangle.out <file> -formula <p> - Calculate formula with power p\n");
    printf("  ./triangle.out <file> -matrix      - Create matrix table\n");
    printf("  ./triangle.out <file> -reconstruct - Reconstruct text from IDs\n");
    printf("  ./triangle.out <file> -circle <m>  - Analyze conjunctions with multiplier m\n");
    printf("  ./triangle.out <file> -graph <m>   - Build graph with circle multiplier m\n");
    printf("  ./triangle.out <file> -context-graph - Build rotated context graph\n");
    printf("  ./triangle.out <file> -context-query <a> <b> - Query ordered context\n");
    printf("  ./triangle.out <old> -learn <new>  - Learn from new text using old as base\n");
    printf("  ./triangle.out <file> -backprop    - Train with backpropagation\n");
}

static int nearest_word_id(const BackpropNetwork *network, const double *vector) {
    int best = 0;
    double best_distance = INFINITY;

    for (int word = 0; word < network->vocab_size; word++) {
        double distance = 0.0;
        for (int d = 0; d < network->embed_dim; d++) {
            double delta = vector[d] - network->embeddings[word * network->embed_dim + d];
            distance += delta * delta;
        }
        if (distance < best_distance) {
            best_distance = distance;
            best = word;
        }
    }

    return best + 1;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const char *filename = argv[1];
    char *file_content = NULL;

    FILE *f = fopen(filename, "r");
    if (f) {
        fclose(f);
        file_content = read_file(filename);
        if (file_content) {
            printf("Loading file: %s\n", filename);
        } else {
            fprintf(stderr, "Error reading file: %s\n", filename);
            return 1;
        }
    } else {
        file_content = strdup(filename);
        if (!file_content) {
            fprintf(stderr, "Error allocating memory\n");
            return 1;
        }
        printf("Input: %s\n", filename);
    }

    TriangleChain *chain = create_triangles(file_content);
    if (!chain) {
        fprintf(stderr, "Failed to create triangles\n");
        free(file_content);
        return 1;
    }

    printf("Created %zu triangle(s)\n", chain->count);
    printf("Vocabulary size: %zu\n", chain->vocab->count);
    if (argc <= 2 || (strcmp(argv[2], "-backprop") != 0 &&
                      strcmp(argv[2], "-backprop-cuda") != 0 &&
                      strcmp(argv[2], "-train") != 0 &&
                      strcmp(argv[2], "-context-graph") != 0 &&
                      strcmp(argv[2], "-context-query") != 0)) {
        print_triangles(chain);
    }

    if (argc > 2 && strcmp(argv[2], "-train") == 0) {
        printf("\nCreating training set...\n");
        TrainingSet *training = training_create(chain);
        if (!training) {
            fprintf(stderr, "Failed to create training set\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        printf("Training samples: %zu\n", training->count);

        printf("\nCreating neural network...\n");
        int max_vocab = chain->vocab->count > 100 ? 100 : chain->vocab->count;
        int hidden = 32;
        NeuralNetwork *nn = nn_create(4, hidden, max_vocab);
        if (!nn) {
            fprintf(stderr, "Failed to create neural network\n");
            training_free(training);
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        printf("Training (5 epochs)...\n");
        nn_train(nn, training, 5, 0.1);

        printf("\nEvaluating...\n");
        EvaluationOutput *eval = evaluate(nn, chain);
        if (eval) {
            evaluation_print(eval);
            evaluation_free(eval);
        }

        nn_save(nn, "model.bin");
        printf("\nModel saved to model.bin\n");

        nn_free(nn);
        training_free(training);
    } else if (argc > 2 && strcmp(argv[2], "-eval") == 0) {
        printf("\nLoading model...\n");
        NeuralNetwork *nn = nn_load("model.bin");
        if (!nn) {
            fprintf(stderr, "Failed to load model\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        printf("\nEvaluating...\n");
        EvaluationOutput *eval = evaluate(nn, chain);
        if (eval) {
            evaluation_print(eval);
            evaluation_free(eval);
        }

        nn_free(nn);
    } else if (argc > 2 && strcmp(argv[2], "-formula") == 0) {
        int power = 1;
        if (argc > 3) {
            power = atoi(argv[3]);
        }

        printf("Calculating formula with power %d...\n", power);

        FormulaResult *formula = formula_calculate(chain, power);
        if (formula) {
            formula_print(formula);
            formula_free(formula);
        }
    } else if (argc > 2 && strcmp(argv[2], "-matrix") == 0) {
        printf("Creating matrix table...\n");

        MatrixTable *matrix = matrix_create(chain);
        if (matrix) {
            matrix_print(matrix);
            matrix_save_csv(matrix, "matrix.csv");
            matrix_free(matrix);
        }
    } else if (argc > 2 && strcmp(argv[2], "-reconstruct") == 0) {
        printf("Reconstructing text...\n");

        ReconstructedText *text = reconstruct(chain);
        if (text) {
            reconstruct_print(text);
            reconstruct_compare(text, file_content);
            reconstruct_free(text);
        }
    } else if (argc > 2 && strcmp(argv[2], "-circle") == 0) {
        double multiplier = 1.0;
        if (argc > 3) {
            multiplier = atof(argv[3]);
        }

        printf("Analyzing conjunctions with multiplier %.2f...\n", multiplier);

        CircleChain *circles = circle_create(chain, multiplier);
        if (circles) {
            circle_print(circles);

            double impact = circle_calculate_impact(circles, chain);
            printf("\nTotal conjunction impact: %.0f\n", impact);

            circle_free(circles);
        }
    } else if (argc > 2 && strcmp(argv[2], "-graph") == 0) {
        double multiplier = 1.0;
        if (argc > 3) {
            multiplier = atof(argv[3]);
        }

        printf("Building graph with multiplier %.2f...\n", multiplier);

        CircleChain *circles = circle_create(chain, multiplier);
        Graph *graph = graph_create(chain, circles);
        if (graph) {
            graph_print(graph);
            graph_save_dot(graph, "graph.dot");
            graph_free(graph);
        }
    } else if (argc > 2 && strcmp(argv[2], "-context-graph") == 0) {
        ContextGraph *context_graph = context_graph_create(chain);
        if (!context_graph) {
            fprintf(stderr, "Failed to create context graph\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }
        context_graph_print(context_graph);
        context_graph_save_dot(context_graph, "context_graph.dot");
        context_graph_free(context_graph);
    } else if (argc > 4 && strcmp(argv[2], "-context-query") == 0) {
        int first_id = 0;
        int second_id = 0;
        for (size_t i = 0; i < chain->vocab->count; i++) {
            if (strcmp(chain->vocab->words[i].text, argv[3]) == 0) first_id = (int)i + 1;
            if (strcmp(chain->vocab->words[i].text, argv[4]) == 0) second_id = (int)i + 1;
        }
        if (first_id == 0 || second_id == 0) {
            fprintf(stderr, "One or both query words are not in the vocabulary\n");
        } else {
            ContextGraph *context_graph = context_graph_create(chain);
            if (!context_graph) {
                fprintf(stderr, "Failed to create context graph\n");
                free_triangles(chain);
                free(file_content);
                return 1;
            }
            context_graph_query(context_graph, chain, first_id, second_id, 20);
            context_graph_free(context_graph);
        }
    } else if (argc > 2 && strcmp(argv[2], "-learn") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Error: -learn requires new text/file argument\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        const char *new_input = argv[3];
        char *new_content = NULL;

        FILE *nf = fopen(new_input, "r");
        if (nf) {
            fclose(nf);
            new_content = read_file(new_input);
            printf("Learning from file: %s\n", new_input);
        } else {
            new_content = strdup(new_input);
            printf("Learning from: %s\n", new_input);
        }

        if (!new_content) {
            fprintf(stderr, "Error reading new input\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        TriangleChain *new_chain = create_triangles(new_content);
        if (!new_chain) {
            fprintf(stderr, "Failed to create triangles for new input\n");
            free(new_content);
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        printf("\nOld: %zu triangles, %zu words\n", chain->count, chain->vocab->count);
        printf("New: %zu triangles, %zu words\n", new_chain->count, new_chain->vocab->count);

        LearningOutput *learning = learn_compare(chain, new_chain);
        if (learning) {
            learning_print(learning);

            TriangleChain *merged = learn_merge(chain, new_chain);
            if (merged) {
                printf("\nMerged: %zu triangles, %zu words\n", merged->count, merged->vocab->count);

                FormulaResult *formula_old = formula_calculate(chain, 1);
                FormulaResult *formula_new = formula_calculate(new_chain, 1);
                FormulaResult *formula_merged = formula_calculate(merged, 1);

                if (formula_old && formula_new && formula_merged) {
                    printf("\nFormula comparison:\n");
                    printf("  Old total:   %.0f\n", formula_old->total);
                    printf("  New total:   %.0f\n", formula_new->total);
                    printf("  Merged total: %.0f\n", formula_merged->total);
                    printf("  Old + New:   %.0f\n", formula_old->total + formula_new->total);
                    formula_free(formula_old);
                    formula_free(formula_new);
                    formula_free(formula_merged);
                }

                for (size_t i = 0; i < merged->count; i++) {
                    for (int p = 0; p < 3; p++) {
                        free(merged->triangles[i].words[p]);
                    }
                }
                free(merged->triangles);
                free(merged->vocab);
                free(merged);
            }

            learning_free(learning);
        }

        free(new_content);
        free_triangles(new_chain);
    } else if (argc > 2 &&
               (strcmp(argv[2], "-backprop") == 0 ||
                strcmp(argv[2], "-backprop-cuda") == 0)) {
        int use_cuda = strcmp(argv[2], "-backprop-cuda") == 0;
        printf("Starting %s backpropagation training on full vocabulary...\n\n",
               use_cuda ? "CUDA" : "CPU");

        int vocab_size = (int)chain->vocab->count;
        int embed_dim = 32;
        BackpropTrainer *trainer = backprop_create(vocab_size, embed_dim, 128,
                                                    3 * embed_dim, 50, 0.1);
        if (!trainer) {
            fprintf(stderr, "Failed to create trainer\n");
            free_triangles(chain);
            free(file_content);
            return 1;
        }

        if (use_cuda) {
            if (backprop_train_cuda(trainer, chain, 2) != 0) {
                backprop_free(trainer);
                free_triangles(chain);
                free(file_content);
                return 1;
            }
        } else {
            backprop_train(trainer, chain);
        }
        backprop_save_model(trainer, "backprop_model.bin");

        printf("\n=== Final Results ===\n");
        printf("Vocabulary: %zu words\n", chain->vocab->count);
        printf("Triangles: %zu\n", chain->count);

        if (trainer->log_count > 0) {
            printf("Final loss: %.6f\n", trainer->logs[trainer->log_count - 1].loss);
        }

        printf("\nTesting predictions on first 10 triangles:\n");
        int evaluation_correct = 0;
        int evaluation_total = 0;
        for (size_t t = 0; t < 10 && t < chain->count; t++) {
            int pred_ids[3];
            for (int target_position = 0; target_position < 3; target_position++) {
                double input[2 * 32];
                for (int q = 0; q < 2; q++) {
                    int source_position = (target_position + q + 1) % 3;
                    int wid = chain->triangles[t].word_ids[source_position] - 1;
                    if (wid < 0) wid = 0;
                    if (wid >= vocab_size) wid = vocab_size - 1;
                    for (int d = 0; d < embed_dim; d++) {
                        input[q * embed_dim + d] =
                            trainer->network->embeddings[wid * embed_dim + d];
                    }
                }

                double output[3 * 32];
                backprop_predict(trainer, input, output);
                pred_ids[target_position] = nearest_word_id(
                    trainer->network, output + target_position * embed_dim);
            }

            int pred_id0 = pred_ids[0];
            int pred_id1 = pred_ids[1];
            int pred_id2 = pred_ids[2];
            int predictions[3] = {pred_id0, pred_id1, pred_id2};
            for (int p = 0; p < 3; p++) {
                evaluation_correct += predictions[p] == chain->triangles[t].word_ids[p];
                evaluation_total++;
            }

            printf("  Triangle %zu: actual=[%d,%d,%d] predicted_ids=[%d,%d,%d]\n",
                   t + 1,
                   chain->triangles[t].word_ids[0],
                   chain->triangles[t].word_ids[1],
                   chain->triangles[t].word_ids[2],
                   pred_id0, pred_id1, pred_id2);
        }
        if (evaluation_total > 0) {
            printf("Evaluation exact word accuracy: %.1f%% (%d/%d)\n",
                   100.0 * evaluation_correct / evaluation_total,
                   evaluation_correct, evaluation_total);
        }

        backprop_free(trainer);
    } else {
        ProbabilityMatrix *prob = prob_create(chain);
        if (prob) {
            prob_print_sequences(prob);
            prob_free(prob);
        }
    }

    free_triangles(chain);
    free(file_content);
    return 0;
}
