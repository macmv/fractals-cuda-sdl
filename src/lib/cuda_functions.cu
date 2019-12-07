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
    int numBlocks,
    Camera cam,
    unsigned int millis,
    int screen_width,
    int screen_height) {
  int steps;
  bool hit;
  double dst;
  double rayPos[3];
  double rayDelta[3];
  double x, y;
  for (int pixel = threadIdx.x + blockIdx.x * numThreads; pixel < numPixels; pixel += numThreads * numBlocks) {
    rayPos[0] = cam.pos[0];
    rayPos[1] = cam.pos[1];
    rayPos[2] = cam.pos[2];
    x = (double) (pixel % screen_width) / screen_width;
    y = (double) (pixel / screen_width) / screen_height;
    cam.getDeltaFrom2D(x, y, rayDelta);
    // printf("x, y: %f %f\n", x, y);
    // printf("Ray delta: %f %f %f\n", rayDelta[0], rayDelta[1], rayDelta[2]);
    steps = 0;
    while (true) {
      // clock_t start_time = clock();
      // clock_t stop_time;
      dst = static_cast<BasicSphere*>(fractal)->DE(rayPos);
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
      // stop_time = clock();
      // printf("Time: %f microseconds\n", (int) (stop_time - start_time) / 1987.0);
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
  int numThreads = 1024;
  int numBlocks = 32;
  drawPixels<<<numBlocks, numThreads>>>(
  numPixels,
  fractal,
  buffer,
  numThreads,
  numBlocks,
  *cam, // copying data because it needs to know xFov and yFov
  SDL_GetTicks(),
  screen_width,
  screen_height);
  cudaDeviceSynchronize();
  gpuErrchk(cudaPeekAtLastError()); // to see stdout before handling
}
