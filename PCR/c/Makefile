NVCC := nvcc
NVCC_FLAGS := -O3 -arch=sm_86
NVCC_FLAGS_NSYS := -O3 -arch=sm_86 -lineinfo

CC := gcc
CFLAGS := -O3 -Wall -Wextra
OPENMP_FLAGS := -fopenmp

NVC := nvc
NVC_FLAGS := -O3 -mp=gpu -gpu=cc86 -Minfo=mp

LIBS := -lm
CUDA_LIBS := -lm -lcuda
LINK_FLAGS := $(LIBS)

# Library source files
C_FILES := sle.c diagonal.c
C_OBJS := $(C_FILES:.c=.o)

# OMP version
OMP_PCR := pcr.c
OMP_PCR_OBJ := pcr.o

# CUDA version
CU_PCR := pcr.cu
CU_PCR_OBJ := pcr_cuda.o
CU_PCR_OBJ_NSYS := pcr_cuda_nsys.o

# OMP GPU version
OMP_GPU_PCR := pcr_omp_gpu.c
OMP_GPU_PCR_OBJ := pcr_omp_gpu.o

# Executables
TARGETS := pcrsolve_omp.run pcrsolve_cuda.run pcrsolve_omp_gpu.run

# Default target
.PHONY: all
all: $(TARGETS)

# Build OMP GPU PCR object file
$(OMP_GPU_PCR_OBJ): $(OMP_GPU_PCR)
	$(NVC) $(NVC_FLAGS) -c $< -o $@

# Build C object files from library sources
%.o: %.c
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c $< -o $@

# Build CUDA PCR object file
$(CU_PCR_OBJ): $(CU_PCR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Build CUDA PCR object file with NSYS profiling flags
$(CU_PCR_OBJ_NSYS): $(CU_PCR)
	$(NVCC) $(NVCC_FLAGS_NSYS) -c $< -o $@

# Build OMP version executable
pcrsolve_omp.run: main.c $(C_OBJS) $(OMP_PCR_OBJ)
	$(CC) $(CFLAGS) -DPCR_MAIN $(OPENMP_FLAGS) $^ $(LIBS) -o $@

# Build CUDA version executable
pcrsolve_cuda.run: main.c $(C_OBJS) $(CU_PCR_OBJ)
	$(NVCC) $(NVCC_FLAGS) -DPCR_MAIN $^ $(CUDA_LIBS) -o $@

# Build CUDA version executable with NSYS profiling flags
pcrsolve_cuda_nsys.run: main.c $(C_OBJS) $(CU_PCR_OBJ_NSYS)
	$(NVCC) $(NVCC_FLAGS_NSYS) -DPCR_MAIN $^ $(CUDA_LIBS) -o $@

# Build OMP GPU version executable
pcrsolve_omp_gpu.run: main.c $(C_OBJS) $(OMP_GPU_PCR_OBJ)
	$(NVC) $(NVC_FLAGS) -DPCR_MAIN $^ $(LIBS) -o $@

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(C_OBJS) $(OMP_PCR_OBJ) $(CU_PCR_OBJ) $(OMP_GPU_PCR_OBJ) $(TARGETS) pcrsolve_cuda_nsys.run

# Clean everything including executables
.PHONY: distclean
distclean: clean
	rm -f *.o $(TARGETS)

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all            		- Build both pcrsolve_omp.run, pcrsolve_cuda.run and pcrsolve_omp_gpu.run (default)"
	@echo "  pcrsolve_omp.run   	- Build OMP version"
	@echo "  pcrsolve_cuda.run  	- Build CUDA GPU version"
	@echo "  pcrsolve_cuda_nsys.run - Build CUDA GPU version with NSYS profiling flags"
	@echo "  pcrsolve_omp_gpu.run  	- Build OMP GPU version"
	@echo "  clean          		- Remove object files and executables"
	@echo "  distclean      		- Remove all generated files"
	@echo "  help           		- Show this help message"
