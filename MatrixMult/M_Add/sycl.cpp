#include <sycl/sycl.hpp>

#include "matadd.h"

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

  sycl::queue q;

  TIME_GET(start);
  Matrix d_A = sycl::malloc_device<int>(NELEMENTS * NELEMENTS, q);
  Matrix d_B = sycl::malloc_device<int>(NELEMENTS * NELEMENTS, q);
  Matrix d_C = sycl::malloc_device<int>(NELEMENTS * NELEMENTS, q);

  q.memcpy(d_A, A, NELEMENTS * NELEMENTS * sizeof(int));
  q.memcpy(d_B, B, NELEMENTS * NELEMENTS * sizeof(int)).wait();

  q.parallel_for(sycl::range<1>(NELEMENTS * NELEMENTS), [=](sycl::id<1> idx) {
     d_C[idx] = d_A[idx] + d_B[idx];
   }).wait();

  q.memcpy(C, d_C, NELEMENTS * NELEMENTS * sizeof(int)).wait();

  sycl::free(d_A, q);
  sycl::free(d_B, q);
  sycl::free(d_C, q);

  TIME_GET(end);
  printf("Result SYCL: %s Time: %lf\n", getMD5DigestStr(C),
         TIME_DIFF(start, end));

  free(A);
  free(B);
  free(C);
}