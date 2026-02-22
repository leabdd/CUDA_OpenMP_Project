CC = gcc
CC_FLAGS = -O3

CUDA_CC = nvcc
CUDA_FLAGS = -arch=sm_86

ACPP=acpp 
ACPP_EXPLICIT=--acpp-targets="omp;cuda:sm_86"

ICPX=icpx
ICPX_FLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 

OMP_CC = nvc 
OMP_FLAGS = -fopenmp -mp=gpu -gpu=cc86 -Minfo=mp

LINK_FLAGS = -lcrypto

# DEP_SRCS = md5sum.c

all: mm_cuda_double.run mm_cuda_float.run mm_cuBlas_double.run mm_cuBlas_float.run

#ma_seq: seq.c ${DEP_SRCS}
#	$(CC) $(CC_FLAGS) ${LINK_FLAGS} -o $@ $^

mm_cuda_double.run: mm_cuda_double.cu ${DEP_SRCS}
	$(CUDA_CC) $(CC_FLAGS) $(CUDA_FLAGS) ${LINK_FLAGS} -o $@ $^

mm_cuda_float.run: mm_cuda_float.cu ${DEP_SRCS}
	$(CUDA_CC) $(CC_FLAGS) $(CUDA_FLAGS) ${LINK_FLAGS} -o $@ $^

mm_cuBlas_double.run: mm_cuBlas_double.cu ${DEP_SRCS}
	$(CUDA_CC) $(CC_FLAGS) $(CUDA_FLAGS) -lcublas ${LINK_FLAGS} -o $@ $^

mm_cuBlas_float.run: mm_cuBlas_float.cu ${DEP_SRCS}
	$(CUDA_CC) $(CC_FLAGS) $(CUDA_FLAGS) -lcublas ${LINK_FLAGS} -o $@ $^

#ma_omp: omp.c ${DEP_SRCS}
#	$(OMP_CC) $(CC_FLAGS) $(OMP_FLAGS) ${LINK_FLAGS} -o $@ $^

#ma_acpp: sycl.cpp ${DEP_SRCS}
##	$(ACPP) $(CC_FLAGS) ${LINK_FLAGS} -o $@ $^

#ma_acpp_explicit: sycl.cpp ${DEP_SRCS}
#	$(ACPP) $(CC_FLAGS) $(ACPP_EXPLICIT) ${LINK_FLAGS} -o $@ $^

#ma_icpx: sycl.cpp ${DEP_SRCS}
#	$(ICPX) $(CC_FLAGS) $(ICPX_FLAGS) ${LINK_FLAGS} -o $@ $^

clean:
	rm -f mm_cuda_double.run mm_cuda_float.run mm_cuBlas_double.run mm_cuBlas_float.run
