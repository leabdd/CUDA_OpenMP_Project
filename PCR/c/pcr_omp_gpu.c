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

static inline float compute_decoupling_coeffs(float decoupling_value,
                                              float into_value) {
  return -into_value / (decoupling_value == 0 ? EPSILON : decoupling_value);
}

int pcr(triSLE_t *sle, timer *start, timer *end) {
  if (sle == NULL) {
    return -1;
  }
  size_t n = sle->b->n;
  if (n == 0) {
    return 0; // Nothing to do
  }

  TIME_GET(*start);

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

  // Extract pointers for OpenMP map clauses (NVCC doesn't handle pointer deref in maps)
  float *a_data = sle->a->data;
  float *b_data = sle->b->data;
  float *c_data = sle->c->data;
  float *d_data = sle->d->data;
  float *x_data = sle->x->data;

  // Map data pointers to GPU for duration of computation
  #pragma omp target enter data map(to: a_data[0:n], b_data[0:n], c_data[0:n], d_data[0:n], x_data[0:n]) \
                                  map(alloc: a_swap[0:n], b_swap[0:n], c_swap[0:n], d_swap[0:n])

  #pragma omp target
  {
    // All iterations run on GPU
    for (size_t level = 0; level < total_levels; level++) {
      size_t stride = 1 << level;
      
      // Call update_kernel directly because NVCC doesnt allow nested pragma inside function call.
      #pragma omp parallel for
      for (int i = 0; i < (int)n; i++) {
        int iRight = i + (int)stride;
        int iLeft = i - (int)stride;

        float decoupling_value = iLeft < 0 ? 1.f : b_data[iLeft];
        const float alpha = -a_data[i] / (decoupling_value == 0 ? EPSILON : decoupling_value);

        decoupling_value = iRight < (int)n ? b_data[iRight] : 1.f;
        const float gamma = -c_data[i] / (decoupling_value == 0 ? EPSILON : decoupling_value);

        /* const float alpha =
            compute_decoupling_coeffs(iLeft < 0 ? 1.f : b_data[iLeft], a_data[i]);
        const float gamma =
            compute_decoupling_coeffs(iRight < (int)n ? b_data[iRight] : 1.f, c_data[i]);
 */
        const float sa_iLeft = iLeft < 0 ? 0.0f : a_data[iLeft];
        const float sc_iLeft = iLeft < 0 ? 0.0f : c_data[iLeft];
        const float sd_iLeft = iLeft < 0 ? 0.0f : d_data[iLeft];

        const float sa_iRight = iRight >= (int)n ? 0.0f : a_data[iRight];
        const float sc_iRight = iRight >= (int)n ? 0.0f : c_data[iRight];
        const float sd_iRight = iRight >= (int)n ? 0.0f : d_data[iRight];

        a_swap[i] = alpha * sa_iLeft;
        c_swap[i] = gamma * sc_iRight;
        b_swap[i] = b_data[i] + alpha * sc_iLeft + gamma * sa_iRight;
        d_swap[i] = d_data[i] + alpha * sd_iLeft + gamma * sd_iRight;
      }

      // Swap pointers within target region (valid for device pointers)
      float *tmp;
      tmp = a_data;
      a_data = a_swap;
      a_swap = tmp;

      tmp = b_data;
      b_data = b_swap;
      b_swap = tmp;

      tmp = c_data;
      c_data = c_swap;
      c_swap = tmp;

      tmp = d_data;
      d_data = d_swap;
      d_swap = tmp;
    }

    // Back substitution: x[i] = d[i] / b[i]
    #pragma omp teams distribute parallel for simd
    for (size_t i = 0; i < n; i++) {
      x_data[i] = d_data[i] / b_data[i];
    }
  }

  // Exit target data region and copy results back to host
  #pragma omp target exit data map(from: x_data[0:n]) \
                               map(release: a_data[0:n], b_data[0:n], c_data[0:n], d_data[0:n]) \
                               map(delete: a_swap[0:n], b_swap[0:n], c_swap[0:n], d_swap[0:n])

  TIME_GET(*end);

  // If total_levels is odd, pointers were swapped an odd number of times
  // Need to copy data from temporary buffers back to original SLE arrays
  /* if ((total_levels % 2) != 0) {
    memcpy(sle->a->data, a_swap, n * sizeof(float));
    memcpy(sle->b->data, b_swap, n * sizeof(float));
    memcpy(sle->c->data, c_swap, n * sizeof(float));
    memcpy(sle->d->data, d_swap, n * sizeof(float));
  } else {
    // Even swaps: data is in a_data, b_data, c_data, d_data
    memcpy(sle->a->data, a_data, n * sizeof(float));
    memcpy(sle->b->data, b_data, n * sizeof(float));
    memcpy(sle->c->data, c_data, n * sizeof(float));
    memcpy(sle->d->data, d_data, n * sizeof(float));
  } */

  free(a_swap);
  free(b_swap);
  free(c_swap);
  free(d_swap);

  return 0;
}
