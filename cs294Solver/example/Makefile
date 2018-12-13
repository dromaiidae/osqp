SHELL := /bin/bash
CC = gcc
FLAGS = -std=gnu99 -O3 -DNDEBUG -g0 -msse4.2 -mavx2

default: cs294Solver

cs294sparse: cs294sparse.c
	$(CC) $(FLAGS) cs294sparse.c -o cs294sparse

cs294Solver: scratchpadExec.c
	$(CC) $(FLAGS) hashmap.c scratchpadExec.c  -o cs294Solver

cs294sparse.c:
	$(CC) $(FLAGS) cs294sparse.c -c

scratchpadExec.c:
	$(CC) $(FLAGS) scratchpadExec.c -c

hashmap.c:
	$(CC) $(FLAGS) hashmap.c -c

utils.c:
	$(CC) $(FLAGS) utils.c -c
