#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

// Idea: we can allocate a 1-D array and index properly to see if it improves locality
void *allocArray(int rows, int cols) {
    return malloc(sizeof(double[rows][cols])); // allocate 1 2D-array
}

void print_matrix(int m, int n, double matrix[m][n]) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", matrix[i][j]);
            }
        printf("\n");
    }
}
void cholesky_row_unoptimized(int dimension, double array[dimension][dimension], double decomp[dimension][dimension]) {
    int i, j;
    int sum, sumIndex;
    for (i = 0; i < dimension; i++) {
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

void cholesky_column_unoptimized_2(int dimension, double array[dimension][dimension]) {
    int i;
    int j;
    int k;
    for (j = 0; j < dimension; j++) {
        for (k = 0; k < j; k++) {
            for (i = j; i < dimension; i++) {
                array[i][j] = array[i][j] - array[i][k]*array[j][k];
            }
        }
        array[j][j] = sqrt(array[j][j]);
        for (i = j+1; i < dimension; i++) {
            array[i][j] = array[i][j] / array[j][j];
        }
    }
    print_matrix(dimension, dimension, array);
} 

// Adapted from https://www.lume.ufrgs.br/bitstream/handle/10183/151001/001009773.pdf
void cholesky_column_unoptimized(int dimension, double array[dimension][dimension], double decomp[dimension][dimension]) {
    int j;
    double sum;
    for (j = 0; j < dimension; j++) { // iterating over columns
        sum = 0;
        int row_sum_index;
        for (row_sum_index = 0; row_sum_index < j; row_sum_index++) {
            sum += array[j][row_sum_index]*array[j][row_sum_index];
        }
        array[j][j] = sqrt(array[j][j] - sum);
    }
    int i;
    for (i = j+1; i < dimension; i++) {
        sum = 0;
        int rest_of_column;
        for (rest_of_column = 0; rest_of_column < j; rest_of_column++) {
            sum += array[i][rest_of_column] * array[j][rest_of_column];
        }
        double divisor = (1.0 / array[j][j]);
        array[i][j] = divisor * (array[i][j] - sum);
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

float time_cholesky_row_unoptimized(int dimension, double array[dimension][dimension], double decomp[dimension][dimension]) {
    int avg = 0;
    int num_runs = 10;
    int i;
    int msec;
    for (i = 0; i < 10; i++) {
        clock_t start = clock();
        clock_t end;
        cholesky_row_unoptimized(dimension, array, decomp);
        end = clock();
        msec = (end - start) * 1000 / CLOCKS_PER_SEC; // https://stackoverflow.com/questions/459691/best-timing-method-in-c
        avg += msec;
    }
   printf("Cholesky Row Unoptimized took %d milliseconds to complete on average of %d runs\n", avg / num_runs, num_runs);
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
    
    cholesky_column_unoptimized_2(rows, matrix);

    free(matrix);
    free(decomp_cholesky);
    return 0;
}
