#include <stdio.h>
#include <stdlib.h>

void *allocArray(int rows, int cols) {
    return malloc(sizeof(double[rows][cols])); // allocate 1 2D-array
}

int readArray(int rows, int cols, double array[rows][cols], const char *filepath) {
    FILE *data;
    if ((data = fopen(filepath, "rb")) == NULL) {
        fprintf(stderr, "IOError: Cannot open file '%s'\n", filepath);
        return 1;
    }
    fread(array, sizeof(double[rows][cols]), 1, data);
    fclose(data);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 2) {
        fprintf(stderr, "Too many arguments. Arguments are: filepath (optional)");
        return 1;
    }
    char *filepath;
    if (argc == 1) {
        filepath = "data/matrix.mat";
    } else {
        filepath = argv[1];
    }

    int cols = 450;
    int rows = 450;
    double (*matrix)[cols] = allocArray(rows, cols);

    if (readArray(rows, cols, matrix, filepath) != 0) {
        return 1;
    }

    int n = 20;
    int m = 20;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }

    free(matrix);
    return 0;
}

