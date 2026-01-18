#include "sle.h"
#include "solver.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>

#if defined(PCR_MAIN)
#define func(system, start, end) pcr(system, start, end)
#define SOLVER_NAME "PCR"
#else
#define func(system, start, end) -1
#define SOLVER_NAME
#error "No solver defined for main.c. "
#endif

int main(int argc, char **argv) {

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <# of equations>\n", argv[0]);
    return -1;
  }

  const int n = atoi(argv[1]);

  triSLE_t *system = NULL;

  if (triSLE_create(&system, n) != 0) {
    return -1; // Failed to create PCR system
  }

  float *lower = system->a->data;
  float *main = system->b->data;
  float *upper = system->c->data;
  float *rhs = system->d->data;

  srand(1234);

  for (int i = 0; i < n; i++) {
    lower[i] =
        ((i > 0) ? ((float)(drand48() * 1e-5)) : 0.0) * (rand() % 2 ? 1 : -1);
    main[i] = (float)(drand48() * 1e2) * (rand() % 2 ? 1 : -1);
    upper[i] = ((i < n - 1) ? ((float)(drand48() * 1e-5)) : 0.0) *
               (rand() % 2 ? 1 : -1);
    rhs[i] = (float)(drand48()) * (rand() % 2 ? 1 : -1);
  }

  // copy A and rhs for verification later

  triSLE_t *system_copy = NULL;
  if (triSLE_create(&system_copy, n) != 0) {
    triSLE_destroy(system);
    return -1; // Failed to create PCR system copy
  }

  if (triSLE_copy(system_copy, system) != 0) {
    triSLE_destroy(system);
    triSLE_destroy(system_copy);
    return -1; // Failed to copy PCR system
  }

  timer start_time, end_time;

  if (func(system, &start_time, &end_time) != 0) {
    triSLE_destroy(system);
    triSLE_destroy(system_copy);
    return -1; // Failed to solve the system
  }

  TIME_PRINT(start_time, end_time, SOLVER_NAME " solve time");

  printf("Max relative error: %e\n",
         triSLE_validate_maxrel(system, system_copy));
  printf("MAPE value: %e%%\n", triSLE_validate_mape(system, system_copy));

  triSLE_destroy(system);
  triSLE_destroy(system_copy);
  return 0;
}