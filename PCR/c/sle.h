/**
 * @file sle.h
 * @brief Tridiagonal system of linear equations (SLE) structures and
 * operations.
 *
 * This header defines a tridiagonal system of linear equations data structure
 * and provides functions for creating, managing, and validating solutions.
 * Tridiagonal systems are common in numerical methods such as finite difference
 * schemes and are efficiently solved using specialized algorithms like the
 * Thomas algorithm or Parallel Cyclic Reduction (PCR).
 */

#ifndef SLE_H
#define SLE_H

#include "diagonal.h"

/**
 * @struct triSLE_s
 * @brief Represents a tridiagonal system of linear equations.
 *
 * A tridiagonal system has the form:
 * \f[ A \mathbf{x} = \mathbf{d} \f]
 * where the coefficient matrix A is tridiagonal with:
 * - Lower diagonal: a
 * - Main diagonal: b
 * - Upper diagonal: c
 * - Right-hand side: d
 *
 * @var triSLE_s::a
 *   Lower diagonal elements (size n, first element unused).
 *
 * @var triSLE_s::b
 *   Main diagonal elements (size n).
 *
 * @var triSLE_s::c
 *   Upper diagonal elements (size n, last element unused).
 *
 * @var triSLE_s::d
 *   Right-hand side vector (size n).
 *
 * @var triSLE_s::x
 *   Solution vector (size n), stores the computed solution.
 */
struct triSLE_s {
  diagonal_t *a;
  diagonal_t *b;
  diagonal_t *c;
  diagonal_t *d;

  diagonal_t *x;
};

/**
 * @typedef triSLE_t
 * @brief Convenience typedef for struct triSLE_s.
 */
typedef struct triSLE_s triSLE_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a new tridiagonal system of linear equations.
 *
 * Allocates memory for a new triSLE structure and initializes all
 * diagonal matrices (a, b, c, d, x) with size n.
 *
 * @param[out] params  Pointer to triSLE_t pointer where the new system
 *                     will be stored. Must not be NULL.
 * @param[in]  n       Size of the system (number of equations).
 *
 * @return 0 on success, non-zero error code on failure.
 *
 * @note The caller is responsible for freeing the allocated memory
 *       using triSLE_destroy().
 */
int triSLE_create(triSLE_t **params, int n);

/**
 * @brief Destroy a tridiagonal system and free its resources.
 *
 * Deallocates all memory associated with the system, including all
 * diagonal matrices and the structure itself.
 *
 * @param[in] params  Pointer to the system to destroy.
 *                    If NULL, the function has no effect.
 *
 * @return 0 on success, non-zero error code on failure.
 *
 * @note After calling this function, the pointer becomes invalid
 *       and should not be used.
 */
int triSLE_destroy(triSLE_t *params);

/**
 * @brief Copy one tridiagonal system to another.
 *
 * Copies all matrices (a, b, c, d, x) from the source system to the
 * destination system. Both systems must be already allocated and have
 * the same size.
 *
 * @param[out] dest  Destination system to copy to.
 * @param[in]  src   Source system to copy from.
 *
 * @return 0 on success, non-zero error code on failure.
 *
 * @note Destination array data is overwritten. Source is not modified.
 */
int triSLE_copy(triSLE_t *dest, triSLE_t *src);

/**
 * @brief Validate solution using maximum relative error.
 *
 * Computes the maximum relative error between a computed solution and
 * a reference solution: \f[ \max\frac{|x_{computed} -
 * x_{reference}|}{|x_{reference}|} \f]
 *
 * @param[in] result  The computed solution to validate.
 * @param[in] before  The reference (true/initial) solution.
 *
 * @return Maximum relative error as a float value.
 *         Returns 0.0 if before vector contains zero values.
 *
 * @note Useful for assessing solution accuracy when exact solutions are known.
 */
float triSLE_validate_maxrel(triSLE_t *result, triSLE_t *before);

/**
 * @brief Validate solution using Mean Absolute Percentage Error (MAPE).
 *
 * Computes the mean absolute percentage error between a computed solution
 * and a reference solution: \f[ MAPE = \frac{1}{n} \sum \frac{|x_{computed} -
 * x_{reference}|}{|x_{reference}|} \times 100\% \f]
 *
 * @param[in] result  The computed solution to validate.
 * @param[in] before  The reference (true/initial) solution.
 *
 * @return Mean absolute percentage error as a float value.
 *         Returns 0.0 if before vector contains zero values.
 *
 * @note Provides a statistical measure of average solution error,
 *       useful for comparing multiple solutions or algorithms.
 */
float triSLE_validate_mape(triSLE_t *result, triSLE_t *before);

#ifdef __cplusplus
}
#endif

#endif // SLE_H