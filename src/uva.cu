#include <cuda_runtime.h>
#include <stdio.h>

// Function to check if device supports UVA
bool deviceSupportsUVA() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    if (prop.unifiedAddressing) {
      return true;
    }
  }

  return false;
}

int main()
{
    if (deviceSupportsUVA()) {
        printf("Device supports UVA\n");
    } else {
        printf("Device does not support UVA\n");
    }
}