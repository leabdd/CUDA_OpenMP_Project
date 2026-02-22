#include <cstdlib>
#include <cuda.h>
#include <openssl/md5.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matmul.h"

#define CUDA_ERR_CHECK(x)                                                           \
    do {                                                                            \
        cudaError_t err = x;                                                        \
        if ((err) != cudaSuccess) {                                                 \
            printf("Error \"%s\" at %s :%d \n", cudaGetErrorString(err), __FILE__,  \
                __LINE__);                                                          \
            exit(-1);                                                               \
        }                                                                           \
    } while (0)

#define TILE_WIDTH 32

typedef float *Matrix;

char *getMD5DigestStr(Matrix m, int N) {
    MD5_CTX ctx;
    unsigned char sum[MD5_DIGEST_LENGTH];
    char *retval, *ptr;

    MD5_Init(&ctx);
    MD5_Update(&ctx, m, N * N);
    MD5_Final(sum, &ctx);

    retval = (char *)calloc(MD5_DIGEST_LENGTH * 2 + 1, sizeof(*retval));
    ptr = retval;

    for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
        snprintf(ptr, 3, "%02X", sum[i]);
        ptr += 2;
    }
    return retval;
}

__global__ void matrix_mult_kernel(const Matrix A, const Matrix B, 
                                       const Matrix C, const int N) {
    float Cval = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N &&  col < N) {
        for (int e = 0; e < N; ++e)
            Cval += A[row * N + e] * B[e * N + col];
        C[row * N + col] = Cval;
    }
    return;
}

__global__ void matrix_mult_tile_kernel(const Matrix A, const Matrix B, 
                                       const Matrix C, const int N) {
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Cval = 0.0;

    // Loop over tiles
    for (int m = 0; m < N / TILE_WIDTH; m++) {
        Ads[ty][tx] = A[Row*N + (m * TILE_WIDTH + tx)];
        Bds[ty][tx] = B[(m * TILE_WIDTH + ty)*N + Col];
        __syncthreads();

        // Loop within tile
        for (int k = 0; k < TILE_WIDTH; ++k)
            Cval += Ads[ty][k] * Bds[k][tx];
        __syncthreads();  
    }
    C[Row * N + Col] = Cval;
    return;
}

double matrix_mult_gpu(const Matrix A, const Matrix B, const Matrix C, const int N, bool use_tiling) {
    struct timespec start;
    struct timespec end;
    size_t size = N * N * sizeof(float);
    Matrix Ad, Bd, Cd;
    /* No device copy needed for N; pass N by value to kernel */

    CUDA_ERR_CHECK(cudaMalloc(&Ad, size));
    CUDA_ERR_CHECK(cudaMalloc(&Bd, size));
    CUDA_ERR_CHECK(cudaMalloc(&Cd, size));

    TIME_GET(start);

    CUDA_ERR_CHECK(cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice));

    dim3 dBlock(32, 32);
    dim3 dGrid(N / 32, N / 32);
    /* pass N directly (host int) instead of dereferencing device pointer) */
    if (use_tiling) {
        matrix_mult_tile_kernel<<<dGrid, dBlock>>>(Ad, Bd, Cd, N);
    } else {
        matrix_mult_kernel<<<dGrid, dBlock>>>(Ad, Bd, Cd, N);
    }
    CUDA_ERR_CHECK(cudaGetLastError());

    CUDA_ERR_CHECK(cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost));
    
    TIME_GET(end);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    return TIME_DIFF(start, end);
}

int calc_mat(const int N, bool use_tiling) {
    double result;
    Matrix A, B, C;
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));
    C = (float *)malloc(N * N * sizeof(float));

    srand(0);

    for (int i = 0; i < N * N; i++) {
        A[i] = rand();
        B[i] = rand();
    }
    char *A_md5 = getMD5DigestStr(A, N);
    char *B_md5 = getMD5DigestStr(B, N);
    printf("Input A MD5: %s\n", A_md5);
    printf("Input B MD5: %s\n", B_md5);
    free(A_md5);
    free(B_md5);

    result = matrix_mult_gpu(A, B, C, N, use_tiling);
    char *md5 = getMD5DigestStr(C, N);
    printf("Result GPU Size %d with%s Tiling:  MD5: %s Time: %lf\n", N, use_tiling ? "" : "out", md5, result);
    
    // calculate GFLOPS
    double flops = 2.0 * N * N * N; // 2*N^3 operations
    double gflops = flops / (result * 1.0e9);
    printf("GFLOPS: %lf\n", gflops);

    free(md5);

    free(A);
    free(B);
    free(C);
    return 0;
}

int main(char *argv[], int argc) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <use_tiling (true or false)>\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]);
    if (N <= 32 || N % 32 != 0) {
        printf("Matrix size must be a positive multiple of 32.\n");
        return -1;
    }

    bool use_tiling = (strcmp(argv[2], "true") == 0);

    calc_mat(N, use_tiling);
    return 0;
}