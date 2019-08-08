#include "cuda_functions.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__
void drawPixel(int numPixels, Fractal* fractal, unsigned char* buffer, int numThreads, Camera* cam, unsigned int millis) {
  for (int pixel = threadIdx.x; pixel < numPixels; pixel += numThreads) {
    double dst = fractal->DE(0, 0, 1.5);
    buffer[pixel * 4 + 0] = 0xFF; // r
    buffer[pixel * 4 + 1] = (unsigned char) millis; // g
    buffer[pixel * 4 + 2] = 0x00; // b
    buffer[pixel * 4 + 3] = 0x00; // a
  }
}

void renderScreen(int numPixels, Fractal* fractal, Camera* cam, unsigned char* buffer) {
  int numThreads = 1024;
  drawPixel<<<1, numThreads>>>(
    numPixels,
    fractal,
    buffer,
    numThreads,
    cam,
    SDL_GetTicks());
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}
