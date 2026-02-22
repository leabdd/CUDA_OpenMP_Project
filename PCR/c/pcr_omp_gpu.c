#include "diagonal.h"
#include "sle.h"
#include "solver.h"
#include "util.h"

#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <string.h>

#define EPSILON 1e-30

// when including this with pragma omp declare target this function makes the code much slower, so we inline it and duplicate the logic in the kernel instead
/* static inline float compute_decoupling_coeffs(float decoupling_value,
                                              float into_value) {
  return -into_value / (decoupling_value == 0 ? EPSILON : decoupling_value);
} */

int pcr(triSLE_t *sle, timer *start, timer *end) {
  if (sle == NULL) {
    return -1;
  }
  size_t n = sle->b->n;
  if (n == 0) {
    return 0; // Nothing to do
  }

  size_t total_levels = (size_t)ceil(log2((float)n));

  float *a_swap = (float *)malloc(n * sizeof(float));
  float *b_swap = (float *)malloc(n * sizeof(float));
  float *c_swap = (float *)malloc(n * sizeof(float));
  float *d_swap = (float *)malloc(n * sizeof(float));

  if (a_swap == NULL || b_swap == NULL || c_swap == NULL || d_swap == NULL) {
    free(a_swap);
    free(b_swap);
    free(c_swap);
    free(d_swap);
    return -1;
  }

  // Extract pointers for OpenMP map clauses - keep these fixed for the map region
  float *a_data = sle->a->data;
  float *b_data = sle->b->data;
  float *c_data = sle->c->data;
  float *d_data = sle->d->data;
  float *x_data = sle->x->data;

  // Use separate pointers for tracking current arrays (avoid swapping mapped pointers)
  float *a_src, *b_src, *c_src, *d_src;
  float *a_dst, *b_dst, *c_dst, *d_dst;

  TIME_GET(*start);
  
  // Map data pointers to GPU for duration of computation
  // Create data region so memory stays on the GPU between iterations
  #pragma omp target data map(to: a_data[0:n], b_data[0:n], c_data[0:n], d_data[0:n]) \
                        map(alloc: a_swap[0:n], b_swap[0:n], c_swap[0:n], d_swap[0:n]) \
                        map(from: x_data[0:n])
  {
    // Sequential loop runs on the host
    for (size_t level = 0; level < total_levels; level++) {
        size_t stride = 1 << level;
        
        // Determine current and next array pointers without modifying the mapped ones
        if (level % 2 == 0) {
          a_src = a_data; b_src = b_data; c_src = c_data; d_src = d_data;
          a_dst = a_swap; b_dst = b_swap; c_dst = c_swap; d_dst = d_swap;
        } else {
          a_src = a_swap; b_src = b_swap; c_src = c_swap; d_src = d_swap;
          a_dst = a_data; b_dst = b_data; c_dst = c_data; d_dst = d_data;
        }
        
        // Update Step Kernel runs on GPU
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < (int)n; i++) {
            int iRight = i + (int)stride;
            int iLeft = i - (int)stride;

            float decoupling_value = iLeft < 0 ? 1.f : b_src[iLeft];
            const float alpha = -a_src[i] / (decoupling_value == 0 ? 1e-9f : decoupling_value);

            decoupling_value = iRight < (int)n ? b_src[iRight] : 1.f;
            const float gamma = -c_src[i] / (decoupling_value == 0 ? 1e-9f : decoupling_value);

            const float sa_iLeft = iLeft < 0 ? 0.0f : a_src[iLeft];
            const float sc_iLeft = iLeft < 0 ? 0.0f : c_src[iLeft];
            const float sd_iLeft = iLeft < 0 ? 0.0f : d_src[iLeft];

            const float sa_iRight = iRight >= (int)n ? 0.0f : a_src[iRight];
            const float sc_iRight = iRight >= (int)n ? 0.0f : c_src[iRight];
            const float sd_iRight = iRight >= (int)n ? 0.0f : d_src[iRight];

            a_dst[i] = alpha * sa_iLeft;
            c_dst[i] = gamma * sc_iRight;
            b_dst[i] = b_src[i] + alpha * sc_iLeft + gamma * sa_iRight;
            d_dst[i] = d_src[i] + alpha * sd_iLeft + gamma * sd_iRight;
        }
    }

    // Added SIMD directive, but seen no performance improvement
    // Back substitution kernel
    if (total_levels % 2 == 1) {
        // Odd Levels: Results are in swap arrays
        #pragma omp target teams distribute parallel for simd
        for (size_t i = 0; i < n; i++) {
            x_data[i] = d_swap[i] / b_swap[i];
        }
    } else {
        // Even Levels: Results are in original arrays
        #pragma omp target teams distribute parallel for simd
        for (size_t i = 0; i < n; i++) {
            x_data[i] = d_data[i] / b_data[i];
        }
    }
  } // Data is automatically copied back to x_data on the host here

  TIME_GET(*end);

  free(a_swap);
  free(b_swap);
  free(c_swap);
  free(d_swap);

  return 0;
}
