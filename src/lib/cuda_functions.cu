#include "cuda_functions.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__ void drawPixel(int numPixels, Fractal* fractal, unsigned char* buffer, int numThreads) {
  for (int pixel = threadIdx.x; pixel < numPixels; pixel += numThreads) {
    buffer[pixel * 4 + 0] = 0xFF; // set to solid red
    buffer[pixel * 4 + 1] = 0x00;
    buffer[pixel * 4 + 2] = 0x00;
    buffer[pixel * 4 + 3] = 0xFF;
  }
}

void renderScreen(int numPixels, Fractal* fractal, unsigned char* buffer) {
  printf("Drawing screen!\n");
  int numThreads = 1024;
  drawPixel<<<1, 1024>>>(
    numPixels,
    fractal,
    buffer,
    numThreads);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}
