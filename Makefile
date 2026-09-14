CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -fopenmp
LDFLAGS = -lm -fopenmp
SRC = src/main.c src/core/triangle.c src/core/word.c src/core/geometry.c src/core/probability.c src/core/matcher.c src/core/neural.c src/core/formula.c src/core/matrix.c src/core/reconstruct.c src/core/circle.c src/core/graph.c src/core/learn.c src/core/punctuation.c src/core/backprop.c
OUT = triangle.out

all: $(OUT)

$(OUT): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OUT)

.PHONY: all clean
