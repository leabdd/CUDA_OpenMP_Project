#include <cuda.h>
#include <cublas_v2.h>
#include <cstdlib>
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

#define CUBLAS_ERR_CHECK(x)                                                         \
    do {                                                                            \
        cublasStatus_t s = (x);                                                     \
        if (s != CUBLAS_STATUS_SUCCESS) {                                           \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__,                   \
                __LINE__, (int)s);                                                  \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while(0)                                                                      \

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

double matrix_mult_gpu(const Matrix A, const Matrix B, const Matrix C, const int N) {
    struct timespec start;
    struct timespec end;
    size_t size = N * N * sizeof(float);
    Matrix Ad, Bd, Cd;
    cublasHandle_t handle;



    CUDA_ERR_CHECK(cudaMalloc(&Ad, size));
    CUDA_ERR_CHECK(cudaMalloc(&Bd, size));
    CUDA_ERR_CHECK(cudaMalloc(&Cd, size));

    CUBLAS_ERR_CHECK(cublasCreate(&handle));

    TIME_GET(start);

    CUDA_ERR_CHECK(cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice));

    // cuBLAS calculates C = alpha*op(A)*op(B) + beta*C
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_ERR_CHECK(cublasSgemm(handle,
        CUBLAS_OP_T, // Operation on A (CUBLAS_OP_N for no transpose)
        CUBLAS_OP_T, // Operation on B (CUBLAS_OP_N for no transpose)
        N,           // number of rows of matrix A and C
        N,           // number of cols of matrix B and C
        N,           // number of cols of matrix A and rows of matrix B
        &alpha,      // scalar multiplier for A*B
        Ad,          // device pointer to matrix A
        N,           // leading dimension of matrix A
        Bd,          // device pointer to matrix B
        N,           // leading dimension of matrix B
        &beta,       // scalar multiplier for C
        Cd,          // device pointer to matrix C
        N            // leading dimension of matrix C
    ));

    CUDA_ERR_CHECK(cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost));

    TIME_GET(end);

    CUBLAS_ERR_CHECK(cublasDestroy(handle));

    CUDA_ERR_CHECK(cudaFree(Ad));
    CUDA_ERR_CHECK(cudaFree(Bd));
    CUDA_ERR_CHECK(cudaFree(Cd));

    return TIME_DIFF(start, end);
}

int calc_mat(const int N) {
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

    result = matrix_mult_gpu(A, B, C, N);

    Matrix C_trans = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_trans[i * N + j] = C[j * N + i];
        }
    }

    char *md5 = getMD5DigestStr(C_trans, N);
    printf("Result GPU Size %d: MD5: %s Time: %lf\n", N, md5, result);

    // calculate GFLOPS
    double flops = 2.0 * N * N * N; // 2*N^3 operations
    double gflops = flops / (result * 1.0e9);
    printf("GFLOPS: %lf\n", gflops);

    free(md5);
    free(A);
    free(B);
    free(C);
    free(C_trans);
    return 0;
}

int main(char *argv[], int argc) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]);
    if (N <= 32 || N % 32 != 0) {
        printf("Matrix size must be a positive multiple of 32.\n");
        return -1;
    }

    calc_mat(N);
    return 0;
}