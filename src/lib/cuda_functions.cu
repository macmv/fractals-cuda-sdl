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
    Camera cam,
    unsigned int millis,
    int screen_width,
    int screen_height) {
  for (int pixel = threadIdx.x; pixel < numPixels; pixel += numThreads) {
    int steps = 0;
    double rayPos[3] = {cam.pos[0], cam.pos[1], cam.pos[2]};
    double rayDelta[3];
    double x = (double) (pixel % screen_width) / screen_width;
    double y = (double) (pixel / screen_width) / screen_height;
    cam.getDeltaFrom2D(x, y, rayDelta);
    // printf("x, y: %f %f\n", x, y);
    // printf("Ray delta: %f %f %f\n", rayDelta[0], rayDelta[1], rayDelta[2]);
    bool hit;
    while (true) {
      double dst = static_cast<BasicSphere*>(fractal)->DE(rayPos);
      if (dst > DST_MAX) {
        hit = false;
        break;
      }
      if (dst < DST_MIN) {
        hit = true;
        break;
      }
      rayPos[0] += rayDelta[0] * dst;
      rayPos[1] += rayDelta[1] * dst;
      rayPos[2] += rayDelta[2] * dst;
      steps += 1;
    }
    if (hit) {
      buffer[pixel * 4 + 0] = (unsigned char) (steps * 4); // r
      buffer[pixel * 4 + 1] = (unsigned char) (steps * 4); // g
      buffer[pixel * 4 + 2] = (unsigned char) (steps * 4); // b
      buffer[pixel * 4 + 3] = 0x00; // a
    } else {
      buffer[pixel * 4 + 0] = 0xFF; // r
      buffer[pixel * 4 + 1] = 0x00; // g
      buffer[pixel * 4 + 2] = 0xFF; // b
      buffer[pixel * 4 + 3] = 0x00; // a
    }
  }
}

void renderScreen(int numPixels, Fractal* fractal, Camera* cam, unsigned char* buffer, int screen_width, int screen_height) {
  int numThreads = 256;
  drawPixels<<<1, numThreads>>>( // function assumes 1 block, don't change
  numPixels,
  fractal,
  buffer,
  numThreads,
  *cam, // copying data because it needs to know xFov and yFov
  SDL_GetTicks(),
  screen_width,
  screen_height);
  cudaDeviceSynchronize();
  gpuErrchk(cudaPeekAtLastError()); // to see stdout before handling
}
