NVCC := nvcc
NVCC_FLAGS := -O2 -arch=sm_86

CC := gcc
CFLAGS := -O2 -Wall -Wextra
OPENMP_FLAGS := -fopenmp

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

# OMP GPU version
OMP_GPU_PCR := pcr_omp_gpu.c
OMP_GPU_PCR_OBJ := pcr_omp_gpu.o

# Executables
TARGETS := pcrsolve_omp pcrsolve_cuda pcrsolve_omp_gpu

# Default target
.PHONY: all
all: $(TARGETS)

# Build OMP GPU PCR object file
$(OMP_GPU_PCR_OBJ): $(OMP_GPU_PCR)
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fopenmp -c $< -o $@

# Build C object files from library sources
%.o: %.c
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c $< -o $@

# Build CUDA PCR object file
$(CU_PCR_OBJ): $(CU_PCR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Build OMP version executable
pcrsolve_omp: main.c $(C_OBJS) $(OMP_PCR_OBJ)
	$(CC) $(CFLAGS) -DPCR_MAIN $(OPENMP_FLAGS) $^ $(LIBS) -o $@

# Build CUDA version executable
pcrsolve_cuda: main.c $(C_OBJS) $(CU_PCR_OBJ)
	$(NVCC) $(NVCC_FLAGS) -DPCR_MAIN $^ $(CUDA_LIBS) -o $@

# Build OMP GPU version executable
pcrsolve_omp_gpu: main.c $(C_OBJS) $(OMP_GPU_PCR_OBJ)
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fopenmp -DPCR_MAIN $^ $(CUDA_LIBS) -o $@

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(C_OBJS) $(OMP_PCR_OBJ) $(CU_PCR_OBJ) $(OMP_GPU_PCR_OBJ) $(TARGETS)

# Clean everything including executables
.PHONY: distclean
distclean: clean
	rm -f *.o $(TARGETS)

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all            	- Build both pcrsolve_omp, pcrsolve_cuda and pcrsolve_omp_gpu (default)"
	@echo "  pcrsolve_omp   	- Build OMP version"
	@echo "  pcrsolve_cuda  	- Build CUDA GPU version"
	@echo "  pcrsolve_omp_gpu  	- Build OMP GPU version"
	@echo "  clean          	- Remove object files and executables"
	@echo "  distclean      	- Remove all generated files"
	@echo "  help           	- Show this help message"
