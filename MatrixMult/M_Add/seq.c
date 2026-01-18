#include "matadd.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_add_seq(const Matrix A, const Matrix B, const Matrix C) {
  for (int i = 0; i < NELEMENTS; i++) {
    for (int j = 0; j < NELEMENTS; j++) {
      C[i * NELEMENTS + j] = A[i * NELEMENTS + j] + B[i * NELEMENTS + j];
    }
  }
  return;
}

int main() {
  struct timespec start;
  struct timespec end;
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

  TIME_GET(start);
  matrix_add_seq(A, B, C);
  TIME_GET(end);
  printf("Result Seq: %s Time: %lf\n", getMD5DigestStr(C),
         TIME_DIFF(start, end));

  free(A);
  free(B);
  free(C);
}