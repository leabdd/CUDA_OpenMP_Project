/**
 * @file diagonal.h
 * @brief Diagonal matrix structure and operations.
 *
 * This header defines a diagonal matrix data structure and provides
 * functions for creating and destroying diagonal matrix instances.
 * Diagonal matrices are useful for efficient storage and computation
 * when most matrix elements are zero (only diagonal elements are non-zero).
 */

#ifndef DIAGONAL_H
#define DIAGONAL_H

#include <stddef.h>

/**
 * @struct diagonal_s
 * @brief Represents a diagonal matrix.
 *
 * A diagonal matrix stores only the diagonal elements to optimize
 * memory usage and computation. The data array contains n elements
 * representing the diagonal values.
 *
 * @var diagonal_s::n
 *   Size of the matrix (n x n). Number of diagonal elements.
 *
 * @var diagonal_s::data
 *   Array of n float values representing the diagonal elements.
 */
struct diagonal_s {
  size_t n;
  float *data;
};

/**
 * @typedef diagonal_t
 * @brief Convenience typedef for struct diagonal_s.
 */
typedef struct diagonal_s diagonal_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a new diagonal matrix.
 *
 * Allocates memory for a new diagonal matrix structure and initializes
 * the data array with n float elements.
 *
 * @param[out] diag  Pointer to diagonal_t pointer where the new matrix
 *                   will be stored. Must not be NULL.
 * @param[in]  n     Size of the diagonal matrix (n x n).
 *
 * @return 0 on success, non-zero error code on failure.
 *
 * @note The caller is responsible for freeing the allocated memory
 *       using diagonal_destroy().
 */
int diagonal_create(diagonal_t **diag, int n);

/**
 * @brief Destroy a diagonal matrix and free its resources.
 *
 * Deallocates all memory associated with the diagonal matrix,
 * including the data array and the structure itself.
 *
 * @param[in] diag  Pointer to the diagonal matrix to destroy.
 *                  If NULL, the function has no effect.
 *
 * @return 0 on success, non-zero error code on failure.
 *
 * @note After calling this function, the pointer becomes invalid
 *       and should not be used.
 */
int diagonal_destroy(diagonal_t *diag);

#ifdef __cplusplus
}
#endif

#endif // DIAGONAL_H