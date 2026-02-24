#include "diagonal.h"
#include "sle.h"
#include "solver.h"
#include "util.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define EPSILON 1e-30f
#define BLOCK_SIZE 256

#define CUDA_ERR_CHECK(x)                                                           \
    do {                                                                            \
        cudaError_t err = x;                                                        \
        if ((err) != cudaSuccess) {                                                 \
            printf("Error \"%s\" at %s :%d \n", cudaGetErrorString(err), __FILE__,  \
                __LINE__);                                                          \
            exit(-1);                                                               \
        }                                                                           \
    } while (0)

// Computes decoupling coefficients for PCR update step
__device__ inline float compute_decoupling_coeffs(float decoupling_value, float into_value) {
    // Using fast division for better performance, leads to some accuracy loss 
    return -__fdividef(into_value, (decoupling_value == 0 ? EPSILON : decoupling_value));
}

// Perform one level of PCR elimination to decouple equations
// Each thread updates one equation based on its neighbors at distance 'stride'
__global__ void update_step_kernel(float *__restrict__ sa, float *__restrict__ sb,
                                    float *__restrict__ sc, float *__restrict__ sd,
                                    float *__restrict__ tmp_a, float *__restrict__ tmp_b,
                                    float *__restrict__ tmp_c, float *__restrict__ tmp_d,
                                    int n, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) {
    int iRight = i + stride;
    int iLeft = i - stride;

    // Compute decoupling coefficients for left and right neighbors
    // Eliminate the off-diagonal elements
    const float alpha =
        compute_decoupling_coeffs(iLeft < 0 ? 1.f : sb[iLeft], sa[i]);
    const float gamma =
        compute_decoupling_coeffs(iRight < n ? sb[iRight] : 1.f, sc[i]);

    // Access coefficients from left neighbor
    const float sa_iLeft = iLeft < 0 ? 0.0f : sa[iLeft];
    const float sc_iLeft = iLeft < 0 ? 0.0f : sc[iLeft];
    const float sd_iLeft = iLeft < 0 ? 0.0f : sd[iLeft];

    // Access coefficients from right neighbor
    const float sa_iRight = iRight >= n ? 0.0f : sa[iRight];
    const float sc_iRight = iRight >= n ? 0.0f : sc[iRight];
    const float sd_iRight = iRight >= n ? 0.0f : sd[iRight];

    // Update coefficients 
    tmp_a[i] = alpha * sa_iLeft;
    tmp_c[i] = gamma * sc_iRight;
    tmp_b[i] = sb[i] + alpha * sc_iLeft + gamma * sa_iRight;
    tmp_d[i] = sd[i] + alpha * sd_iLeft + gamma * sd_iRight;
  }
}

// When matrix is diagonal, solution is x[i] = d[i] / b[i]
__global__ void back_substitution_kernel(float *__restrict__ d, float *__restrict__ b,
                                         float *__restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i < n) {
        // Using fast division for better performance, leads to some accuracy loss
        x[i] = __fdividef(d[i], b[i]); 
    }
}

// Solves Ax = d for x
// This is done by decoupling the tridiagonal system 
// Once finished the matrix is diagonal and we can divide to get the solution
int pcr(triSLE_t *sle, timer *start, timer *end) {
    
    // Validate input
    if (sle == NULL) {
        return -1;
    }
  
    size_t n = sle->b->n;
    
    // Handle trivial case
    if (n == 0) {
        return 0;
    }

    // Allocate GPU memory for coefficient matrices and temporary buffers
    // Maintain two sets of arrays to allow in-place updates via swapping
    float *d_a, *d_b, *d_c, *d_d;                   // Current coefficient arrays
    float *d_tmp_a, *d_tmp_b, *d_tmp_c, *d_tmp_d;   // Temporary arrays for updates
    float *d_x;                                     // Solution vector
    
    CUDA_ERR_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_ERR_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_ERR_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
    CUDA_ERR_CHECK(cudaMalloc(&d_d, n * sizeof(float)));
    CUDA_ERR_CHECK(cudaMalloc(&d_tmp_a, n * sizeof(float)));
    CUDA_ERR_CHECK(cudaMalloc(&d_tmp_b, n * sizeof(float)));
    CUDA_ERR_CHECK(cudaMalloc(&d_tmp_c, n * sizeof(float)));
    CUDA_ERR_CHECK(cudaMalloc(&d_tmp_d, n * sizeof(float)));
    CUDA_ERR_CHECK(cudaMalloc(&d_x, n * sizeof(float)));

    TIME_GET(*start);

    // Copy initial data to GPU
    CUDA_ERR_CHECK(cudaMemcpyAsync(d_a, sle->a->data, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpyAsync(d_b, sle->b->data, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpyAsync(d_c, sle->c->data, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpyAsync(d_d, sle->d->data, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaDeviceSynchronize());

    // Requires log2(n) iterations to fully decouple the matrix
    // size_t total_levels = (size_t)ceil(log2((float)n));
    size_t total_levels = (size_t)ceil(log2((float)n + 1.));

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (size_t level = 0; level < total_levels; level++) {
        int stride = 1 << level;

        // Perform one level of elimination
        update_step_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, d_d,
                                                    d_tmp_a, d_tmp_b, d_tmp_c, d_tmp_d,
                                                    (int)n, stride);
        CUDA_ERR_CHECK(cudaPeekAtLastError());

        // Swap current and temporary arrays for next iteration (in-place update)
        float *swap;
        swap = d_a;
        d_a = d_tmp_a;
        d_tmp_a = swap;

        swap = d_b;
        d_b = d_tmp_b;
        d_tmp_b = swap;

        swap = d_c;
        d_c = d_tmp_c;
        d_tmp_c = swap;

        swap = d_d;
        d_d = d_tmp_d;
        d_tmp_d = swap;
    }

    // Matrix is diagonal, direct division gives solution
    back_substitution_kernel<<<grid_size, BLOCK_SIZE>>>(d_d, d_b, d_x, (int)n);
    CUDA_ERR_CHECK(cudaPeekAtLastError());

    // Copy results back to host
    CUDA_ERR_CHECK(cudaMemcpyAsync(sle->x->data, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_ERR_CHECK(cudaDeviceSynchronize());

    TIME_GET(*end);

    // Free GPU memory
    CUDA_ERR_CHECK(cudaFree(d_a));
    CUDA_ERR_CHECK(cudaFree(d_b));
    CUDA_ERR_CHECK(cudaFree(d_c));
    CUDA_ERR_CHECK(cudaFree(d_d));
    CUDA_ERR_CHECK(cudaFree(d_tmp_a));
    CUDA_ERR_CHECK(cudaFree(d_tmp_b));
    CUDA_ERR_CHECK(cudaFree(d_tmp_c));
    CUDA_ERR_CHECK(cudaFree(d_tmp_d));
    CUDA_ERR_CHECK(cudaFree(d_x));

    return 0;
}
