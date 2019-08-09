#include "cuda_functions.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU: %s: %d: %s\n", file, line, cudaGetErrorString(code));
    if (abort) exit(code);
  }
}

__global__
void drawPixels(int numPixels,
  Fractal* fractal,
  unsigned char* buffer,
  int numThreads,
  Camera* cam,
  unsigned int millis,
  int screen_width,
  int screen_height) {
  for (int pixel = threadIdx.x; pixel < numPixels; pixel += numThreads) {
    double dst = fractal->DE(0, 0, 1.5);
    double delta[3];
    cam->getDeltaFrom2D((double) (pixel % screen_width) / screen_width, (double) (pixel / screen_width) / screen_width, delta);
    buffer[pixel * 4 + 0] = (unsigned char) (delta[0] * 256); // r
    buffer[pixel * 4 + 1] = (unsigned char) (delta[1] * 256); // g
    buffer[pixel * 4 + 2] = (unsigned char) (delta[2] * 256); // b
    buffer[pixel * 4 + 3] = 0x00; // a
  }
}

void renderScreen(int numPixels, Fractal* fractal, Camera* cam, unsigned char* buffer, int screen_width, int screen_height) {
  int numThreads = 1024;
  drawPixels<<<1, numThreads>>>(
    numPixels,
    fractal,
    buffer,
    numThreads,
    cam,
    SDL_GetTicks(),
    screen_width,
    screen_height);
  cudaDeviceSynchronize();
  gpuErrchk(cudaPeekAtLastError()); // to see stdout before handling
}
