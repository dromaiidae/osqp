#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define BILLION 1000000000L /* https://www.cs.rutgers.edu/~pxk/416/notes/c-tutorials/gettime.html */
#define DIMENSION 160
#define FILEPATH "../data/matrix_160.mat"

int* SPARSE_ROW[DIMENSION];

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

void print_matrix(int m, int n, double matrix[m][n]) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", matrix[i][j]);
            }
        printf("\n");
    }
}

void print_sparse() {
    int row = 0;
    int k;
    for (row = 0; row < DIMENSION; row++) {
        printf("Row: %d Nonzero_indices ", row);
        k = 0;
        while (SPARSE_ROW[row][k] != -1) {
            printf("%d ", SPARSE_ROW[row][k]);
            k += 1;
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

int count_nonzeros_in_row(int row, int dimension, double array[dimension][dimension]) {
    int i;
    int nonzeros = 0;
    for (i = 0; i < dimension; i++) {
        if (array[row][i] != 0) {
            nonzeros += 1;
        }
    }
    return nonzeros;
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



void column_sparse_cholesky(double array[DIMENSION][DIMENSION]) {
    int j;
    int k_struct_index;
    for (j = 0; j < DIMENSION; j++) {
        k_struct_index = 0;
        while (SPARSE_ROW[j][k_struct_index] != -1) {
            // printf("Row %d Col %d has value %f\n", j, SPARSE_ROW[j][k_struct_index], array[j][SPARSE_ROW[j][k_struct_index]]);
            update_mod_col_simd_T(j, SPARSE_ROW[j][k_struct_index], DIMENSION, array);
            k_struct_index += 1;
        }
        update_divide_col_sqrt_simd_T(j, DIMENSION, array);
    }
}

void create_sparse_representation(int rows, int cols, double array[rows][cols]) {
    int row;
    int col;
    int col_index;
    int i; 
    int nonzeros;
    
    for (row = 0; row < DIMENSION; row++) {
        nonzeros = count_nonzeros_in_row(row, DIMENSION, array);

        SPARSE_ROW[row] = malloc(sizeof(double)*(nonzeros+1));
        col_index = 0;
        for (col = 0; col < row; col++) {
            if (array[row][col] != 0) {
                SPARSE_ROW[row][col_index] = col;
                col_index += 1;
            }
        }
        SPARSE_ROW[row][col_index] = -1;
    }
}

void time_cholesky_sparse(double array[DIMENSION][DIMENSION]) {
    struct timespec start, end;
    uint64_t diff;

    clock_gettime(CLOCK_REALTIME, &start);
    column_sparse_cholesky(array);
    clock_gettime(CLOCK_REALTIME, &end);

    /*
    double (*transposedprint)[DIMENSION] = allocArray(DIMENSION, DIMENSION);
    transpose(DIMENSION, array, transposedprint);
    print_matrix(DIMENSION, DIMENSION, transposedprint);
    free(transposedprint);
    */
    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    printf("Cholesky Sparse took %d seconds (%llu nanoseconds) to complete\n", end.tv_sec - start.tv_sec, (long long unsigned int) diff);
    
}
int main(int argc, char **argv) {

    if (argc > 2) {
        fprintf(stderr, "Too many arguments. Usage is: %s <filepath (optional)>\n", argv[0]);
        return 1;
    }
    char *filepath;
    if (argc == 1) {
        filepath = "../data/matrix.mat";
    } else {
        filepath = argv[1];
    }

    int cols = DIMENSION;
    int rows = DIMENSION;

    double (*matrix)[cols] = allocArray(rows, cols);

    if (readArray(rows, rows, matrix, FILEPATH) != 0) {
        return 1;
    }

    create_sparse_representation(DIMENSION, DIMENSION, matrix);

    double (*transposed)[DIMENSION] = allocArray(DIMENSION, DIMENSION);
    transpose(DIMENSION, matrix, transposed);
    time_cholesky_sparse(transposed);

    free(matrix);
    free(transposed);

    return 0;
}
