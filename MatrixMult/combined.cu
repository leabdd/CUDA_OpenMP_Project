#include <cstdlib>
#include <cuda.h>
#include <cublas_v2.h>
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
    } while(0)   

typedef float *MatrixFloat;
typedef double *MatrixDouble;

char *getMD5DigestStrFloat(MatrixFloat m, int N) {
  MD5_CTX ctx;
  unsigned char sum[MD5_DIGEST_LENGTH];
  char *retval, *ptr;

  MD5_Init(&ctx);
  MD5_Update(&ctx, m, N * N * sizeof(float));
  MD5_Final(sum, &ctx);

  retval = (char *)calloc(MD5_DIGEST_LENGTH * 2 + 1, sizeof(*retval));
  ptr = retval;

  for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
    snprintf(ptr, 3, "%02X", sum[i]);
    ptr += 2;
  }
  return retval;
}

char *getMD5DigestStrDouble(MatrixDouble m, int N) {
  MD5_CTX ctx;
  unsigned char sum[MD5_DIGEST_LENGTH];
  char *retval, *ptr;

  MD5_Init(&ctx);
  MD5_Update(&ctx, m, N * N * sizeof(double));
  MD5_Final(sum, &ctx);

  retval = (char *)calloc(MD5_DIGEST_LENGTH * 2 + 1, sizeof(*retval));
  ptr = retval;

  for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
    snprintf(ptr, 3, "%02X", sum[i]);
    ptr += 2;
  }
  return retval;
}

__global__ void matrix_mult_kernel_float(const MatrixFloat A, const MatrixFloat B, 
                                       const MatrixFloat C, const int N) {
    float Cval = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int e = 0; e < N; ++e)
        Cval += A[row * N + e] * B[e * N + col];
    C[row * N + col] = Cval;
    return;
}

__global__ void matrix_mult_kernel_double(const MatrixDouble A, const MatrixDouble B, 
                                       const MatrixDouble C, const int N) {
    double Cval = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int e = 0; e < N; ++e)
        Cval += A[row * N + e] * B[e * N + col];
    C[row * N + col] = Cval;
    return;
}

MatrixFloat cuda_matrix_mult_gpu_float(const MatrixFloat A, const MatrixFloat B, const MatrixFloat C, const int N, bool use_tiling) {
    struct timespec start;
    struct timespec end;
    size_t size = N * N * sizeof(float);
    MatrixFloat Ad, Bd, Cd;
    /* No device copy needed for N; pass N by value to kernel */

    TIME_GET(start);
    CUDA_ERR_CHECK(cudaMalloc(&Ad, size));
    CUDA_ERR_CHECK(cudaMalloc(&Bd, size));
    CUDA_ERR_CHECK(cudaMalloc(&Cd, size));
    CUDA_ERR_CHECK(cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice));

    dim3 dBlock(32, 32);
    dim3 dGrid(N / 32, N / 32);
    /* pass N directly (host int) instead of dereferencing device pointer) */
    if (use_tiling) {
        // matrix_mult_tile_kernel<<<dGrid, dBlock>>>(Ad, Bd, Cd, N);
    } else {
        matrix_mult_kernel_float<<<dGrid, dBlock>>>(Ad, Bd, Cd, N);
    }
    CUDA_ERR_CHECK(cudaGetLastError());

    CUDA_ERR_CHECK(cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost));
    TIME_GET(end);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    return C;
}

MatrixDouble cuda_matrix_mult_gpu_double(const MatrixDouble A, const MatrixDouble B, const MatrixDouble C, const int N, bool use_tiling) {
    struct timespec start;
    struct timespec end;
    size_t size = N * N * sizeof(double);
    MatrixDouble Ad, Bd, Cd;
    /* No device copy needed for N; pass N by value to kernel */

    TIME_GET(start);
    CUDA_ERR_CHECK(cudaMalloc(&Ad, size));
    CUDA_ERR_CHECK(cudaMalloc(&Bd, size));
    CUDA_ERR_CHECK(cudaMalloc(&Cd, size));
    CUDA_ERR_CHECK(cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice));

    dim3 dBlock(32, 32);
    dim3 dGrid(N / 32, N / 32);
    /* pass N directly (host int) instead of dereferencing device pointer) */
    if (use_tiling) {
        // matrix_mult_tile_kernel<<<dGrid, dBlock>>>(Ad, Bd, Cd, N);
    } else {
        matrix_mult_kernel_double<<<dGrid, dBlock>>>(Ad, Bd, Cd, N);
    }
    CUDA_ERR_CHECK(cudaGetLastError());

    CUDA_ERR_CHECK(cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost));
    TIME_GET(end);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    return C;
}

MatrixFloat cublas_matrix_mult_gpu_float(const MatrixFloat A, const MatrixFloat B, const MatrixFloat C, const int N) {
    struct timespec start;
    struct timespec end;
    size_t size = N * N * sizeof(float);
    MatrixFloat Ad, Bd, Cd;
    cublasHandle_t handle;

    TIME_GET(start);

    CUDA_ERR_CHECK(cudaMalloc(&Ad, size));
    CUDA_ERR_CHECK(cudaMalloc(&Bd, size));
    CUDA_ERR_CHECK(cudaMalloc(&Cd, size));
    CUDA_ERR_CHECK(cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice));

    CUBLAS_ERR_CHECK(cublasCreate(&handle));

    // cuBLAS calculates C = alpha*op(A)*op(B) + beta*C
    const float alpha = 1.0;
    const float beta = 0.0;
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

    CUBLAS_ERR_CHECK(cublasDestroy(handle));

    CUDA_ERR_CHECK(cudaFree(Ad));
    CUDA_ERR_CHECK(cudaFree(Bd));
    CUDA_ERR_CHECK(cudaFree(Cd));

    TIME_GET(end);

    return C;
}

MatrixDouble cublas_matrix_mult_gpu_double(const MatrixDouble A, const MatrixDouble B, const MatrixDouble C, const int N) {
    struct timespec start;
    struct timespec end;
    size_t size = N * N * sizeof(double);
    MatrixDouble Ad, Bd, Cd;
    cublasHandle_t handle;

    TIME_GET(start);

    CUDA_ERR_CHECK(cudaMalloc(&Ad, size));
    CUDA_ERR_CHECK(cudaMalloc(&Bd, size));
    CUDA_ERR_CHECK(cudaMalloc(&Cd, size));
    CUDA_ERR_CHECK(cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice));

    CUBLAS_ERR_CHECK(cublasCreate(&handle));

    // cuBLAS calculates C = alpha*op(A)*op(B) + beta*C
    const double alpha = 1.0;
    const double beta = 0.0;
    CUBLAS_ERR_CHECK(cublasDgemm(handle,
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

    CUBLAS_ERR_CHECK(cublasDestroy(handle));

    CUDA_ERR_CHECK(cudaFree(Ad));
    CUDA_ERR_CHECK(cudaFree(Bd));
    CUDA_ERR_CHECK(cudaFree(Cd));

    TIME_GET(end);

    return C;
}

int calc_mat(const int N) {
    double result;
    MatrixDouble A_double, B_double, C_cuda_double, C_cublas_double;
    A_double = (double *)malloc(N * N * sizeof(double));
    B_double = (double *)malloc(N * N * sizeof(double));
    C_cuda_double = (double *)malloc(N * N * sizeof(double));
    C_cublas_double = (double *)malloc(N * N * sizeof(double));

    
    MatrixFloat A_Float, B_Float, C_cuda_float, C_cublas_float;
    A_Float = (float *)malloc(N * N * sizeof(float));
    B_Float = (float *)malloc(N * N * sizeof(float));
    C_cuda_float = (float *)malloc(N * N * sizeof(float));
    C_cublas_float = (float *)malloc(N * N * sizeof(float));

    srand(0);

    for (int i = 0; i < N * N; i++) {
        A_double[i] = rand();
        B_double[i] = rand();
        
        A_Float[i] = rand();
        B_Float[i] = rand();
    }
    // printf("First element A[0,0]=%f B[0,0]=%f\n", A[0], B[0]);

    C_cuda_double = cuda_matrix_mult_gpu_double(A_double, B_double, C_cuda_double, N, false);
    C_cublas_double = cublas_matrix_mult_gpu_double(A_double, B_double, C_cublas_double, N);

    C_cuda_float = cuda_matrix_mult_gpu_float(A_Float, B_Float, C_cuda_float, N, false);
    C_cublas_float = cublas_matrix_mult_gpu_float(A_Float, B_Float, C_cublas_float, N);

    // result = matrix_mult_gpu(A, B, C, N);
    // char *md5 = getMD5DigestStr(C, N);
    // printf("Result GPU Size %d:  MD5: %s Time: %lf\n", N, md5, result);
    // free(md5);

    // adjust cublas and cuda results comparison because cublas is column major

    MatrixDouble C_cublas_double_trans = (double *)malloc(N * N * sizeof(double));
    MatrixFloat C_cublas_float_trans = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_cublas_double_trans[i * N + j] = C_cublas_double[j * N + i];
            C_cublas_float_trans[i * N + j] = C_cublas_float[j * N + i];
        }
    }

    // difference between cuda and cublas results for double as mean absolute error
    double error_double = 0.0;
    for (int i = 0; i < N * N; i++) {
        error_double += fabs(C_cuda_double[i] - C_cublas_double_trans[i]);
    }
    error_double /= (N * N);
    printf("Mean absolute error between CUDA and cuBLAS double results: %f\n", error_double);

    // printing C_cuda_double and C_cublas_double for debugging
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("C_cuda_double[%d,%d]=%f ", i, j, C_cuda_double[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("C_cublas_double[%d,%d]=%f ", i, j, C_cublas_double_trans[i * N + j]);
        }
        printf("\n");
    }

    // difference between cuda and cublas results for float as mean absolute error
    double error_float = 0.0;
    for (int i = 0; i < N * N; i++) {
        error_float += fabs(C_cuda_float[i] - C_cublas_float_trans[i]);
    }
    error_float /= (N * N);
    printf("Mean absolute error between CUDA and cuBLAS float results: %f\n", error_float);

    // printing C_cuda_float and C_cublas_float for debugging
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("C_cuda_float[%d,%d]=%f ", i, j, C_cuda_float[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("C_cublas_float[%d,%d]=%f ", i, j, C_cublas_float_trans[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    free(A_double);
    free(B_double);
    free(C_cuda_double);
    free(C_cublas_double);
    free(C_cublas_double_trans);

    free(A_Float);
    free(B_Float);
    free(C_cuda_float);
    free(C_cublas_float);
    free(C_cublas_float_trans);
    return 0;
}

int main() {
    const int N = 32; // Example size, can be modified
    calc_mat(N);
    return 0;
}