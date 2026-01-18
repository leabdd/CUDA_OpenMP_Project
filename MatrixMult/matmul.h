#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

#define TIME_GET(timer) clock_gettime(CLOCK_MONOTONIC, &timer)

#define TIME_DIFF(timer1, timer2)                                              \
  ((timer2.tv_sec * 1.0E+9 + timer2.tv_nsec) -                                 \
   (timer1.tv_sec * 1.0E+9 + timer1.tv_nsec)) /                                \
      1.0E+9

/* typedef float *Matrix;

char *getMD5DigestStr(Matrix m); */

#endif // MATMUL_H