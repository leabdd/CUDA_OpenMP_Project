#include <stdio.h>

// Beginning of GPU Architecture definitions
// see https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
      {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
      {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
      {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf("MapSMtoCores for SM %d.%d is undefined."
         "  Default to use %d Cores/SM\n",
         major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

void deviceQuery() {
  cudaDeviceProp prop;
  int nDevices = 0, i;
  cudaError_t ierr;

  ierr = cudaGetDeviceCount(&nDevices);
  if (ierr != cudaSuccess) {
    printf("Sync error: %s\n", cudaGetErrorString(ierr));
  }

  for (i = 0; i < nDevices; ++i) {
    ierr = cudaGetDeviceProperties(&prop, i);
    printf("Device number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n\n", prop.major, prop.minor);

    printf("  Total SMs: %d \n", prop.multiProcessorCount);
    //  if (prop.major == 6 && prop.minor == 1)
    //  	printf("  Cores per SM: %d \n", 128);
    printf("  Cores per SM: %d \n",
           _ConvertSMVer2Cores(prop.major, prop.minor));
    printf("  Total cores: %d\n", _ConvertSMVer2Cores(prop.major, prop.minor) *
                                      prop.multiProcessorCount);
    printf("  Warp size: %d\n\n", prop.warpSize);

    printf("  Shared Memory Per SM: %lu bytes\n",
           prop.sharedMemPerMultiprocessor);
    printf("  Shared Memory Per Block: %lu bytes\n", prop.sharedMemPerBlock);
    printf("  L2 Cache Size: %d bytes\n", prop.l2CacheSize);
    printf("  Total Global Memory: %lu bytes\n\n", prop.totalGlobalMem);

    printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max threads per block: %d\n\n", prop.maxThreadsPerBlock);

    printf("  Max threads in X-dimension of block: %d\n",
           prop.maxThreadsDim[0]);
    printf("  Max threads in Y-dimension of block: %d\n",
           prop.maxThreadsDim[1]);
    printf("  Max threads in Z-dimension of block: %d\n\n",
           prop.maxThreadsDim[2]);

    printf("  Max blocks in X-dimension of grid: %d\n", prop.maxGridSize[0]);
    printf("  Max blocks in Y-dimension of grid: %d\n", prop.maxGridSize[1]);
    printf("  Max blocks in Z-dimension of grid: %d\n\n", prop.maxGridSize[2]);
  }
}

int main() { deviceQuery(); }
