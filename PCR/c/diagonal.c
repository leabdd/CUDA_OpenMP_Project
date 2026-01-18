#include "diagonal.h"
#include "util.h"

#include <stdlib.h>

int diagonal_create(diagonal_t **diag, int n) {
  diagonal_t *d = (diagonal_t *)calloc(1, sizeof(diagonal_t));
  if (d == NULL) {
    goto error;
  }

  d->n = (size_t)n;
  d->data = (float *)calloc(n, sizeof(float));
  if (d->data == NULL) {
    goto error;
  }

  *diag = d;
  return 0; // Success

error:
  FREE_IF_NOT_NULL(d->data);
  FREE_IF_NOT_NULL(d);

  return -1; // Memory allocation failed
}

int diagonal_destroy(diagonal_t *diag) {
  FREE_IF_NOT_NULL(diag->data);
  FREE_IF_NOT_NULL(diag);

  return 0; // Success
}