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

static inline int update_step(triSLE_t *restrict inout_sle,
                              float *restrict tmp_a, float *restrict tmp_b,
                              float *restrict tmp_c, float *restrict tmp_d,
                              size_t level) {
  size_t n = inout_sle->b->n;
  size_t stride = 1 << level;

  // Pointers for faster access
  float *restrict sa = inout_sle->a->data;
  float *restrict sb = inout_sle->b->data;
  float *restrict sc = inout_sle->c->data;
  float *restrict sd = inout_sle->d->data;

#pragma omp parallel for
  for (int i = 0; i < (int)n; i++) {
    int iRight = i + (int)stride;
    int iLeft = i - (int)stride;

    const float alpha =
        compute_decoupling_coeffs(iLeft < 0 ? 1.f : sb[iLeft], sa[i]);
    const float gamma =
        compute_decoupling_coeffs(iRight < (int)n ? sb[iRight] : 1.f, sc[i]);

    const float sa_iLeft = iLeft < 0 ? 0.0f : sa[iLeft];
    const float sc_iLeft = iLeft < 0 ? 0.0f : sc[iLeft];
    const float sd_iLeft = iLeft < 0 ? 0.0f : sd[iLeft];

    const float sa_iRight = iRight >= (int)n ? 0.0f : sa[iRight];
    const float sc_iRight = iRight >= (int)n ? 0.0f : sc[iRight];
    const float sd_iRight = iRight >= (int)n ? 0.0f : sd[iRight];

    tmp_a[i] = alpha * sa_iLeft;
    tmp_c[i] = gamma * sc_iRight;
    tmp_b[i] = sb[i] + alpha * sc_iLeft + gamma * sa_iRight;
    tmp_d[i] = sd[i] + alpha * sd_iLeft + gamma * sd_iRight;
  }

  return 0;
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

  float *a_data_tmp = (float *)malloc(n * sizeof(float));
  float *b_data_tmp = (float *)malloc(n * sizeof(float));
  float *c_data_tmp = (float *)malloc(n * sizeof(float));
  float *d_data_tmp = (float *)malloc(n * sizeof(float));

  if (a_data_tmp == NULL || b_data_tmp == NULL || c_data_tmp == NULL ||
      d_data_tmp == NULL) {
    free(a_data_tmp);
    free(b_data_tmp);
    free(c_data_tmp);
    free(d_data_tmp);
    return -1; // Memory allocation failure
  }

  for (size_t level = 0; level < total_levels; level++) {
    update_step(sle, a_data_tmp, b_data_tmp, c_data_tmp, d_data_tmp, level);
    // Swap pointers
    float *swap;
    swap = sle->a->data;
    sle->a->data = a_data_tmp;
    a_data_tmp = swap;

    swap = sle->b->data;
    sle->b->data = b_data_tmp;
    b_data_tmp = swap;

    swap = sle->c->data;
    sle->c->data = c_data_tmp;
    c_data_tmp = swap;

    swap = sle->d->data;
    sle->d->data = d_data_tmp;
    d_data_tmp = swap;
  }

#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    sle->x->data[i] = sle->d->data[i] / sle->b->data[i];
  }

  // If total_levels is odd, the pointers have been swapped an odd number of
  // times. sle->...->data is pointing to the temporary buffers and ..._data_tmp
  // is pointing to the original buffers. We need to copy the data back to the
  // original buffers and restore the pointers so the caller gets the correct
  // final state and we free the correct memory.
  if ((total_levels % 2) != 0) {
    // Copy data from temporary buffers to original buffers
    memcpy(a_data_tmp, sle->a->data, n * sizeof(float));
    memcpy(b_data_tmp, sle->b->data, n * sizeof(float));
    memcpy(c_data_tmp, sle->c->data, n * sizeof(float));
    memcpy(d_data_tmp, sle->d->data, n * sizeof(float));

    // Swap pointers back. After this, sle->...->data will point to the original
    // buffers (which now have the correct data), and ..._data_tmp will point
    // to the temporary buffers, which can then be safely freed.
    float *swap;
    swap = sle->a->data;
    sle->a->data = a_data_tmp;
    a_data_tmp = swap;

    swap = sle->b->data;
    sle->b->data = b_data_tmp;
    b_data_tmp = swap;

    swap = sle->c->data;
    sle->c->data = c_data_tmp;
    c_data_tmp = swap;

    swap = sle->d->data;
    sle->d->data = d_data_tmp;
    d_data_tmp = swap;
  }

  TIME_GET(*end);

  free(a_data_tmp);
  free(b_data_tmp);
  free(c_data_tmp);
  free(d_data_tmp);

  return 0;
}
