#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Idea: we can allocate a 1-D array and index properly to see if it improves locality
void *allocArray(int rows, int cols) {
    return malloc(sizeof(double[rows][cols])); // allocate 1 2D-array
}

void cholesky_unoptimized(int rows, int cols, double array[rows][cols], double decomp[rows][cols]) {
    int i, j;
    int sum, sumIndex;
    for (i = 0; i < rows; i++) {
        for (j = 0; j <= i; j++) {
            sum = 0;
            sumIndex = 0;
            if (i != j) {
                for (sumIndex = 0; sumIndex < j; sumIndex++) {
                    sum += decomp[i][sumIndex]*decomp[j][sumIndex];
                }
                double divisor = (1 / decomp[j][j]);
                decomp[i][j] = divisor * (array[i][j] - sum);
            } else {
                for (sumIndex = 0; sumIndex < j; sumIndex++) {
                    sum += (decomp[j][sumIndex]*decomp[j][sumIndex]);
                }
                decomp[i][j] = sqrt(array[j][j] - sum);
            }
        }
    }
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
        fprintf(stderr, "Too many arguments. Usage is: %s <filepath (optional)>\n", argv[0]);
        return 1;
    }
    char *filepath;
    if (argc == 1) {
        filepath = "data/matrix2.mat";
    } else {
        filepath = argv[1];
    }

    int cols = 450;
    int rows = 450;
    double (*matrix)[cols] = allocArray(rows, cols);
    double (*decomp_cholesky)[cols] = allocArray(rows, cols);

    if (readArray(rows, cols, matrix, filepath) != 0) {
        return 1;
    }

    cholesky_unoptimized(rows, cols, matrix, decomp_cholesky);

    int m = 450;
    int n = 450;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", decomp_cholesky[i][j]);
        }
        printf("\n");
    }

    free(matrix);
    free(decomp_cholesky);
    return 0;
}

