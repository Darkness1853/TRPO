CC = gcc
CFLAGS = -Wall -O -pthread
LDFLAGS = -lopenblas -lm -lpthread

all: myblas

myblas: myblas.c
	$(CC) $(CFLAGS) -o myblas myblas.c $(LDFLAGS)

clean:
	rm -f myblas

.PHONY: all clean