#include <cstdlib>
#include <cuda.h>
#include <openssl/md5.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matadd.h"

#define CUDA_ERR_CHECK(x)                                                      \
  do {                                                                         \
    cudaError_t err = x;                                                       \
    if ((err) != cudaSuccess) {                                                \
      printf("Error \"%s\" at %s :%d \n", cudaGetErrorString(err), __FILE__,   \
             __LINE__);                                                        \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

__global__ void matrix_add_kernel(const Matrix A, const Matrix B,
                                  const Matrix C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  C[row * NELEMENTS + col] =
      A[row * NELEMENTS + col] + B[row * NELEMENTS + col];
  return;
}

double matrix_add_gpu(const Matrix A, const Matrix B, const Matrix C) {
  struct timespec start;
  struct timespec end;
  size_t size = NELEMENTS * NELEMENTS * sizeof(int);
  Matrix Ad, Bd, Cd;

  TIME_GET(start);
  CUDA_ERR_CHECK(cudaMalloc(&Ad, size));
  CUDA_ERR_CHECK(cudaMalloc(&Bd, size));
  CUDA_ERR_CHECK(cudaMalloc(&Cd, size));

  CUDA_ERR_CHECK(cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice));

  dim3 dBlock(32, 32);
  dim3 dGrid(NELEMENTS / 32, NELEMENTS / 32);
  matrix_add_kernel<<<dGrid, dBlock>>>(Ad, Bd, Cd);
  CUDA_ERR_CHECK(cudaGetLastError());

  CUDA_ERR_CHECK(cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost));
  TIME_GET(end);

  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cd);
  return TIME_DIFF(start, end);
}

int main() {
  double result;
  Matrix A, B, C;
  A = (int *)malloc(NELEMENTS * NELEMENTS * sizeof(int));
  B = (int *)malloc(NELEMENTS * NELEMENTS * sizeof(int));
  C = (int *)malloc(NELEMENTS * NELEMENTS * sizeof(int));

  srand(0);

  for (int i = 0; i < NELEMENTS * NELEMENTS; i++) {
    A[i] = rand();
    B[i] = rand();
  }

  result = matrix_add_gpu(A, B, C);
  printf("Result GPU: %s Time: %lf\n", getMD5DigestStr(C), result);

  free(A);
  free(B);
  free(C);
}
