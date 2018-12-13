#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define BILLION 1000000000L /* https://www.cs.rutgers.edu/~pxk/416/notes/c-tutorials/gettime.html */
#define DIMENSION 450


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

// SIMD on Transpose
void update_divide_col_sqrt_simd_T(int col, int dimension, double array[dimension][dimension]) {
    __m256d column_values, divisor, result;

    double load_array[] = {0,0,0,0};
    // 1 / sqrt and just multiply
    double square_rooted = sqrt(array[col][col]);
    double print[] = {0,0,0,0};

    array[col][col] = square_rooted; // unsure if there is a sqrt vector
    divisor = _mm256_set_pd(1/square_rooted, 1/square_rooted, 1/square_rooted, 1/square_rooted);

    int i;
    for (i=col+1; i+3 < dimension; i += 4) {
        // load 4 values from same column
        column_values = _mm256_set_pd(array[col][i+3], array[col][i+2], array[col][i+1], array[col][i]);
        // divide by the square root
        result = _mm256_mul_pd(column_values, divisor);
        // store the answer
        _mm256_store_pd(&array[col][i], result);
    }
    // Update remainder of values
    for (; i < dimension; i++) {
        array[col][i] = array[col][i] / square_rooted;
    }
}

// SIMD update on Transpose
void update_mod_col_simd_T(int col_to_update_j, int prev_col_k, int dimension, double array[dimension][dimension]) {
    __m256d a_ij, column_values, multiplicand, multiplication_result, new_a_ij;
    double load_array[] = {0,0,0,0};
    double value_jk = array[prev_col_k][col_to_update_j];
    multiplicand = _mm256_set_pd(value_jk, value_jk, value_jk, value_jk); // a_jk


    int i;
    for (i=col_to_update_j; i+3 < dimension; i += 4) {
        // load 4 consecutive values from k column
        column_values = _mm256_set_pd(array[prev_col_k][i+3], array[prev_col_k][i+2], array[prev_col_k][i+1], array[prev_col_k][i]);
        // multiply a_ik * a_jk
        multiplication_result = _mm256_mul_pd(column_values, multiplicand); // array[prev_col_k][i], array[prev_col_k][i+1] array[prev_col_k][i+2] ..
        // load a_ij to perform a_ij -= a_ik * a_jk
        a_ij = _mm256_set_pd(array[col_to_update_j][i+3], array[col_to_update_j][i+2], array[col_to_update_j][i+1], array[col_to_update_j][i]);
        // perform the subtraction
        new_a_ij = _mm256_sub_pd(a_ij, multiplication_result); // [prev_col_k][i], i+1, i+2, i+3 a_ij
        // load the proper values
        _mm256_store_pd(&array[col_to_update_j][i], new_a_ij);
    }
    // Update remainder of values
    for (; i < dimension; i++) {
        array[col_to_update_j][i] -= array[prev_col_k][i] * array[prev_col_k][col_to_update_j];
    }
}

// SIMD update -- not memory efficient
void update_divide_col_sqrt_simd(int col, int dimension, double array[dimension][dimension]) {
    __m256d column_values, divisor, result;
    double load_array[] = {0,0,0,0};
    double print[] = {0,0,0,0};
    // 1 / sqrt and just multiply
    double square_rooted = sqrt(array[col][col]);

    divisor = _mm256_set_pd(square_rooted, square_rooted, square_rooted, square_rooted);

    array[col][col] = square_rooted; // unsure if there is a sqrt vector

    int i;
    for (i=col+1; i+3 < dimension; i += 4) {
        // load 4 values from same column
        column_values = _mm256_set_pd(array[i+3][col], array[i+2][col], array[i+1][col], array[i][col]);
        // divide by the square root
        result = _mm256_div_pd(column_values, divisor);

        // load the answer
        _mm256_store_pd(&load_array, result);
        // reassign
        array[i][col] = load_array[0];
        array[i+1][col] = load_array[1];
        array[i+2][col] = load_array[2];
        array[i+3][col] = load_array[3];
    }
    // Update remainder of values
    for (; i < dimension; i++) {
        array[i][col] = array[i][col] / square_rooted;
    }
}

// SIMD update -- not memory efficient
void update_mod_col_simd(int col_to_update_j, int prev_col_k, int dimension, double array[dimension][dimension]) {
    __m256d a_ij, column_values, multiplicand, multiplication_result, new_a_ij;
    double load_array[] = {0,0,0,0};
    double value_jk = array[col_to_update_j][prev_col_k];

    multiplicand = _mm256_set_pd(value_jk, value_jk, value_jk, value_jk); // a_jk

    int i;
    for (i=col_to_update_j; i+3 < dimension; i += 4) {
        // load 4 consecutive values from k column
        column_values = _mm256_set_pd(array[i+3][prev_col_k], array[i+2][prev_col_k], array[i+1][prev_col_k], array[i+0][prev_col_k]);
        // multiply a_ik * a_jk
        multiplication_result = _mm256_mul_pd(column_values, multiplicand);
        // load a_ij to perform a_ij -= a_ik * a_jk
        a_ij = _mm256_set_pd(array[i+3][col_to_update_j], array[i+2][col_to_update_j], array[i+1][col_to_update_j], array[i][col_to_update_j]);
        // perform the subtraction
        new_a_ij = _mm256_sub_pd(a_ij, multiplication_result);
        // load the answer
        _mm256_store_pd(((double *)load_array), new_a_ij);
        // reassign
        array[i][col_to_update_j] = load_array[0];
        array[i+1][col_to_update_j] = load_array[1];
        array[i+2][col_to_update_j] = load_array[2];
        array[i+3][col_to_update_j] = load_array[3];
    }
    // Update remainder of values
    for (; i < dimension; i++) {
        array[i][col_to_update_j] -= array[i][prev_col_k] * array[col_to_update_j][prev_col_k];
    }
}

void cholesky_simd_smartloading(int dimension, double array[dimension][dimension]) {
    int j;  
    int k;
    for (j = 0; j < dimension; j++) {
        for (k = 0; k < j; k++) {
            update_mod_col_simd_T(j, k, dimension, array);
        }
        update_divide_col_sqrt_simd_T(j, dimension, array);
        // printf("coldiv %d %lf\n", j, array[2][2]);

    }
    //print_matrix(dimension, dimension, array);
}

void cholesky_simd(int dimension, double array[dimension][dimension]) {
    int j;  
    int k;
    for (j = 0; j < dimension; j++) {
        for (k = 0; k < j; k++) {
            update_mod_col_simd(j, k, dimension, array);
        }
        update_divide_col_sqrt_simd(j, dimension, array);
    }
    print_matrix(dimension, dimension, array);
}

// Adapted from https://courses.engr.illinois.edu/cs554/fa2015/notes/07_cholesky.pdf
void cholesky_column_unoptimized(int dimension, double array[dimension][dimension]) {
    int i, j, k;
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

void transpose(int dimension, double array[dimension][dimension], double transposed[dimension][dimension]) {
    double* data = malloc(sizeof(double)*dimension);
    int row;
    int col;
    int transpose_row_insert;
    for (row = 0; row < dimension; row++) {
        for (col = 0; col < dimension; col++) {
            data[col] = array[row][col];
        }
        for (transpose_row_insert = 0; transpose_row_insert < dimension; transpose_row_insert++) {
            transposed[transpose_row_insert][row] = data[transpose_row_insert];
        }
        memset(data, 0, dimension);
    }
    free(data);
}

float time_cholesky_row_unoptimized(int dimension, double array[dimension][dimension], double decomp[dimension][dimension]) {


    struct timespec start, end;
    uint64_t diff;

    clock_gettime(CLOCK_REALTIME, &start);
    cholesky_row_unoptimized(dimension, array, decomp);
    clock_gettime(CLOCK_REALTIME, &end);
    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    printf("Cholesky Row Unoptimized took %d seconds (%llu nanoseconds) to complete\n", end.tv_sec - start.tv_sec, (long long unsigned int) diff);
}

float time_cholesky_column_unoptimized(int dimension, double array[dimension][dimension]) {

    struct timespec start, end;
    uint64_t diff;

    clock_gettime(CLOCK_REALTIME, &start);
    cholesky_column_unoptimized(dimension, array);
    clock_gettime(CLOCK_REALTIME, &end);
    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    printf("Cholesky Column Unoptimized took %d seconds (%llu nanoseconds) to complete\n", end.tv_sec - start.tv_sec, (long long unsigned int) diff);
}


void time_cholesky_SIMD(int dimension, double array[dimension][dimension]) {
    clock_t start, end;
    start = clock();
    cholesky_simd(dimension, array);
    end = clock();
    printf("Cholesky optimized with SIMD took %d milliseconds to complete\n", (end - start) * 1000 / CLOCKS_PER_SEC);
}

void time_cholesky_simd_smartloading(int dimension, double array[dimension][dimension]) {
    struct timespec start, end;
    uint64_t diff;

    clock_gettime(CLOCK_REALTIME, &start);
    cholesky_simd_smartloading(dimension, array);
    clock_gettime(CLOCK_REALTIME, &end);
    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    printf("Cholesky Column with SIMD took %d seconds (%llu nanoseconds) to complete\n", end.tv_sec - start.tv_sec, (long long unsigned int) diff);
}


int main(int argc, char **argv) {
    if (argc > 2) {
        fprintf(stderr, "Too many arguments. Usage is: %s <filepath (optional)>\n", argv[0]);
        return 1;
    }
    char *filepath;
    if (argc == 1) {
        filepath = "../data/matrix2.mat";
    } else {
        filepath = argv[1];
    }

    int cols = DIMENSION;
    int rows = DIMENSION;
    double (*matrix)[cols] = allocArray(rows, cols);
    double (*decomp_cholesky)[cols] = allocArray(rows, cols);
    double (*transposed)[cols] = allocArray(rows, cols);

    if (readArray(rows, cols, matrix, filepath) != 0) {
        return 1;
    }

    transpose(rows, matrix, transposed);

    // print_matrix(rows, rows, matrix);

    // time_cholesky_column_unoptimized(rows, matrix);

    // time_cholesky_row_unoptimized(rows, matrix, decomp_cholesky);
    // cholesky_simd(rows, matrix);
    // time_cholesky_SIMD(rows, matrix);

    time_cholesky_simd_smartloading(rows, transposed);

    free(matrix);
    free(decomp_cholesky);
    free(transposed);
    return 0;
}