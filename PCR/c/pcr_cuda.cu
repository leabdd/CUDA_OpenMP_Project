#include "diagonal.h"
#include "sle.h"
#include "solver.h"
#include "util.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h> 

#define EPSILON 1e-30

#define CUDA_ERR_CHECK(x)                                                           \
    do {                                                                            \
        cudaError_t err = x;                                                        \
        if ((err) != cudaSuccess) {                                                 \
            printf("Error \"%s\" at %s :%d \n", cudaGetErrorString(err), __FILE__,  \
                __LINE__);                                                          \
            exit(-1);                                                               \
        }                                                                           \
    } while (0)

