#ifndef MATADD_H
#define MATADD_H

#include <stddef.h>

#define TIME_GET(timer) clock_gettime(CLOCK_MONOTONIC, &timer)

#define TIME_DIFF(timer1, timer2)                                              \
  ((timer2.tv_sec * 1.0E+9 + timer2.tv_nsec) -                                 \
   (timer1.tv_sec * 1.0E+9 + timer1.tv_nsec)) /                                \
      1.0E+9

typedef int *Matrix;

static const size_t NELEMENTS = 1 << 14; // 16384

#ifdef __cplusplus
extern "C" {
#endif
char *getMD5DigestStr(Matrix m);
void matrix_add_seq(const Matrix A, const Matrix B, const Matrix C);
#ifdef __cplusplus
}
#endif

#endif // MATADD_H