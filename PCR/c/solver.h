/**
 * @file solver.h
 * @brief Parallel Cyclic Reduction (PCR) solver implementations for tridiagonal
 * systems.
 */

#ifndef SOLVER_H
#define SOLVER_H

#include "sle.h"
#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Solve a tridiagonal system using Parallel Cyclic Reduction (CPU
 * implementation).
 *
 * Solves the tridiagonal system Ax = d using the Parallel Cyclic Reduction
 * algorithm on the CPU with OpenMP parallelization. The algorithm proceeds
 * through log(n) stages, each reducing the system size by recursively
 * eliminating elements.
 *
 * The input system must have all matrices initialized:
 * - sle->a: lower diagonal
 * - sle->b: main diagonal
 * - sle->c: upper diagonal
 * - sle->d: right-hand side vector
 *
 * The solution is stored in sle->x.
 *
 * @param[in,out] sle    Pointer to the tridiagonal system to solve.
 *                       On input, contains coefficients and RHS.
 *                       On output, contains solution in sle->x.
 * @param[out]    start  Timer to record the start time.
 * @param[out]    end    Timer to record the end time.
 *
 * @return 0 on success, non-zero error code on failure.
 *
 * @note This implementation uses OpenMP for parallelization on CPU.
 *
 * @see pcr_gpu() for GPU-accelerated implementation.
 */
int pcr(triSLE_t *sle, timer *start, timer *end);

/**
 * @brief Solve a tridiagonal system using Parallel Cyclic Reduction (GPU
 * implementation).
 *
 * TODO: This is your job :-)
 *
 * The input system must have all matrices initialized:
 * - sle->a: lower diagonal
 * - sle->b: main diagonal
 * - sle->c: upper diagonal
 * - sle->d: right-hand side vector
 *
 * The solution is stored in sle->x.
 *
 * @param[in,out] sle    Pointer to the tridiagonal system to solve.
 *                       On input, contains coefficients and RHS.
 *                       On output, contains solution in sle->x.
 * @param[out]    start  Timer to record the start time (Before memory copies
 * in)
 * @param[out]    end    Timer to record the end time (After memory copies out)
 *
 * @return 0 on success, non-zero error code on failure (e.g., GPU not
 * available).
 *
 * @see pcr() for CPU implementation.
 */
int pcr_gpu(triSLE_t *sle, timer *start, timer *end);

#ifdef __cplusplus
}
#endif
#endif // SOLVER_H