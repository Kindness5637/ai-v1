CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -fopenmp
LDFLAGS = -lm -fopenmp
SRC = src/main.c src/core/triangle.c src/core/word.c src/core/geometry.c src/core/probability.c src/core/matcher.c src/core/neural.c src/core/formula.c src/core/matrix.c src/core/reconstruct.c src/core/circle.c src/core/graph.c src/core/context_graph.c src/core/learn.c src/core/punctuation.c src/core/backprop.c src/core/backprop_cuda_stub.c src/core/eval_cuda_stub.c
OUT = triangle.out

CUDA = nvcc
CUDA_FLAGS = -O3 -arch=sm_75 -Iinclude -Xcompiler -fopenmp
CUDA_SRC = src/main.c src/core/triangle.c src/core/word.c src/core/geometry.c src/core/probability.c src/core/matcher.c src/core/neural.c src/core/formula.c src/core/matrix.c src/core/reconstruct.c src/core/circle.c src/core/graph.c src/core/context_graph.c src/core/learn.c src/core/punctuation.c src/core/backprop.c
CUDA_C = $(CUDA_SRC:.c=.cuda.o)
CUDA_O = src/core/backprop_cuda.cuda.o src/core/eval_cuda.cuda.o
CUDA_OUT = triangle_cuda.out

TEST_OUT = tests/test_role_ids.out
TEST_SRC = tests/test_role_ids.c src/core/triangle.c src/core/word.c
TEST2_OUT = tests/test_role_entropy.out
TEST2_SRC = tests/test_role_entropy.c src/core/triangle.c src/core/word.c

all: $(OUT)

$(OUT): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Adversarial tests for the UPOS role plumbing. -Isrc is needed because the
# test lives in tests/ and resolves "core/..." relative to the source tree.
test: $(TEST_OUT) $(TEST2_OUT)
	./$(TEST_OUT)
	./$(TEST2_OUT)

$(TEST_OUT): $(TEST_SRC)
	$(CC) $(CFLAGS) -Isrc -o $@ $^ $(LDFLAGS)

$(TEST2_OUT): $(TEST2_SRC)
	$(CC) $(CFLAGS) -Isrc -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OUT) $(CUDA_OUT) $(CUDA_C) $(CUDA_O) $(TEST_OUT) $(TEST2_OUT)

cuda: $(CUDA_OUT)

$(CUDA_OUT): $(CUDA_C) $(CUDA_O)
	$(CUDA) -o $@ $^ -lm -Xcompiler -fopenmp

%.cuda.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

src/core/backprop_cuda.cuda.o: src/core/backprop_cuda.cu
	$(CUDA) $(CUDA_FLAGS) -c -o $@ $<

src/core/eval_cuda.cuda.o: src/core/eval_cuda.cu
	$(CUDA) $(CUDA_FLAGS) -c -o $@ $<

.PHONY: all clean cuda test
