#include "camera.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

Camera::Camera() {
  printf("Initializing camera...\n");
  cudaMallocManaged(&pos, sizeof(double) * 3);
  cudaMallocManaged(&dir, sizeof(double) * 2);
  cudaDeviceSynchronize();
  pos[0] = -5;
  pos[1] = 0;
  pos[2] = 0;
  dir[0] = 0;
  dir[1] = 0;
  gpuErrchk(cudaPeekAtLastError());
  printf("Initialized camera\n");
}

__device__
void Camera::getDeltaFrom2D(double x, double y, double* delta) {
  delta[0] = cos((dir[0] + (x - 0.5) * fov) / 180 * M_PI);
  delta[1] = dir[1] + (y - 0.5) * (fov * (16.0/9)) / 180;
  delta[2] = sin((dir[0] + (x - 0.5) * fov) / 180 * M_PI);
}

void Camera::free() {
  cudaFree(pos);
  cudaFree(dir);
}
