#include "cuda_functions.h"

__global__ void drawPixel(int numPixels, Fractal* fractal, unsigned char* buffer) {
  for (int i = threadIdx.x; i < numPixels; i += threadIdx.x) {
    buffer[i * 3] = 0xFF;
  }
}

void renderScreen(int numPixels, Fractal* fractal, unsigned char* buffer) {
  printf("Drawing screen!");
  drawPixel<<<1, 128>>>(
    numPixels,
    fractal,
    buffer);
  cudaDeviceSynchronize();
}
