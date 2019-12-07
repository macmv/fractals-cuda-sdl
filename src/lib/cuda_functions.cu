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
    cam.getDeltaFrom2D((double) (pixel % screen_width) / screen_width, (double) (pixel / screen_width) / screen_width, rayDelta);
    bool hit;
    // printf("rayPos: %f, %f, %f\n", rayPos[0], rayPos[1], rayPos[2]);
    // printf("rayDelta: %f, %f, %f\n", rayDelta[0], rayDelta[1], rayDelta[2]);
    while (true) {
      // double dst = 0;
      double dst = static_cast<BasicSphere*>(fractal)->DE(rayPos);
      // printf("dst: %f\n", dst);
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
      buffer[pixel * 4 + 0] = (unsigned char) (steps * 16); // r
      buffer[pixel * 4 + 1] = (unsigned char) (steps * 16); // g
      buffer[pixel * 4 + 2] = (unsigned char) (steps * 16); // b
      buffer[pixel * 4 + 3] = 0x00; // a
    } else {
      buffer[pixel * 4 + 0] = 0x00; // r
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
