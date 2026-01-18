#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FREE_IF_NOT_NULL(ptr)                                                  \
  do {                                                                         \
    if ((ptr) != NULL) {                                                       \
      free(ptr);                                                               \
      (ptr) = NULL;                                                            \
    }                                                                          \
  } while (0)

typedef struct timespec timer;

#define TIME_GET(timer) clock_gettime(CLOCK_MONOTONIC, &timer)

#define TIME_DIFF(timer1, timer2)                                              \
  ((timer2.tv_sec * 1.0E+9 + timer2.tv_nsec) -                                 \
   (timer1.tv_sec * 1.0E+9 + timer1.tv_nsec)) /                                \
      1.0E+9

#define TIME_PRINT(timer1, timer2, msg)                                        \
  printf("%s: %lf sec\n", msg, TIME_DIFF(timer1, timer2));

#endif // UTILS_H