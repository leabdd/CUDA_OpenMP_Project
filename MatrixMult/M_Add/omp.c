#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matadd.h"
#include <omp.h>

void matrix_add_gpu(const Matrix A, const Matrix B, const Matrix C) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < NELEMENTS * NELEMENTS; i++) {
    C[i] = A[i] + B[i];
  }
  return;
}

int main() {
  struct timespec start;
  struct timespec end;
  Matrix A, B, C;
  A = (int *)malloc(NELEMENTS * NELEMENTS * sizeof(int));
  B = (int *)malloc(NELEMENTS * NELEMENTS * sizeof(int));
  C = (int *)malloc(NELEMENTS * NELEMENTS * sizeof(int));

  srand(0);

  for (int i = 0; i < NELEMENTS * NELEMENTS; i++) {
    A[i] = rand();
    B[i] = rand();
  }

  TIME_GET(start);
#pragma omp target enter data map(to : A[0 : NELEMENTS * NELEMENTS],           \
                                      B[0 : NELEMENTS * NELEMENTS])
#pragma omp target enter data map(alloc : C[0 : NELEMENTS * NELEMENTS])
  matrix_add_gpu(A, B, C);
#pragma omp target exit data map(from : C[0 : NELEMENTS * NELEMENTS])
  TIME_GET(end);
  printf("Result GPU: %s Time: %lf\n", getMD5DigestStr(C),
         TIME_DIFF(start, end));

  free(A);
  free(B);
  free(C);
}