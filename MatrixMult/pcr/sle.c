#include "sle.h"
#include "diagonal.h"
#include "util.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

int triSLE_create(triSLE_t **soe, int n) {
  triSLE_t *p = (triSLE_t *)calloc(1, sizeof(triSLE_t));
  if (p == NULL) {
    return -1; // Memory allocation failed
  }

  if (diagonal_create(&p->a, n) != 0 || diagonal_create(&p->b, n) != 0 ||
      diagonal_create(&p->c, n) != 0 || diagonal_create(&p->d, n) != 0 ||
      diagonal_create(&p->x, n) != 0) {
    triSLE_destroy(p);
    return -1; // Memory allocation failed
  }

  *soe = p;
  return 0; // Success
}

int triSLE_destroy(triSLE_t *soe) {
  if (soe == NULL) {
    return -1; // Invalid parameter
  }

  diagonal_destroy(soe->a);
  diagonal_destroy(soe->b);
  diagonal_destroy(soe->c);
  diagonal_destroy(soe->d);
  diagonal_destroy(soe->x);

  FREE_IF_NOT_NULL(soe);

  soe = NULL;

  return 0; // Success
}

int triSLE_copy(triSLE_t *dest, triSLE_t *src) {
  if (dest == NULL || src == NULL) {
    return -1; // Invalid parameter
  }

  memcpy(dest->a->data, src->a->data, src->b->n * sizeof(float));
  memcpy(dest->b->data, src->b->data, src->b->n * sizeof(float));
  memcpy(dest->c->data, src->c->data, src->b->n * sizeof(float));
  memcpy(dest->d->data, src->d->data, src->b->n * sizeof(float));

  return 0; // Success
}

float triSLE_validate_maxrel(triSLE_t *result_system,
                             triSLE_t *initial_system) {
  if (result_system == NULL || initial_system == NULL) {
    return 0.0 / 0.0;
  }

  float result;
  float max_relative_error = 0.0;

  for (size_t i = 0; i < initial_system->b->n; i++) {
    result = initial_system->b->data[i] * result_system->x->data[i];

    if (i > 0) {
      result += initial_system->a->data[i] * result_system->x->data[i - 1];
    }

    if (i < initial_system->b->n - 1) {
      result += initial_system->c->data[i] * result_system->x->data[i + 1];
    }

    // check for NaN
    if (result != result) {
      max_relative_error = result;
      break;
    }

    const float expected = initial_system->d->data[i];
    if (expected != 0.0) {
      float relative_error = fabs(result - expected) / (fabs(expected));
      if (relative_error > max_relative_error) {
        max_relative_error = relative_error;
      }
    }
  }

  return max_relative_error; // Success
}

float triSLE_validate_mape(triSLE_t *result_system, triSLE_t *initial_system) {
  if (result_system == NULL || initial_system == NULL) {
    return 0.0 / 0.0;
  }

  float result;
  float total_absolute_percentage_error = 0.0;

  for (size_t i = 0; i < initial_system->b->n; i++) {
    result = initial_system->b->data[i] * result_system->x->data[i];

    if (i > 0) {
      result += initial_system->a->data[i] * result_system->x->data[i - 1];
    }

    if (i < initial_system->b->n - 1) {
      result += initial_system->c->data[i] * result_system->x->data[i + 1];
    }

    const float expected = initial_system->d->data[i];
    if (expected != 0.0) {
      float absolute_percentage_error =
          fabs((result - expected) / expected) * 100.0;
      total_absolute_percentage_error += absolute_percentage_error;
    } else if (expected == 0.0 && result != 0.0) {
      total_absolute_percentage_error += 1.0;
    }
  }

  return total_absolute_percentage_error / initial_system->b->n; // Success
}