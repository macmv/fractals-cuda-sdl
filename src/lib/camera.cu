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
  cudaMalloc(&pos, sizeof(pos[0]) * 3);
  cudaMalloc(&dir, sizeof(pos[0]) * 2);
  cudaDeviceSynchronize();
  gpuErrchk(cudaPeekAtLastError());
  printf("Initialized camera\n");
}

__device__
void Camera::getDeltaFrom2D(double x, double y, double* delta) {
  delta[0] = x;
  delta[1] = y;
  delta[2] = pos[0];
}

void Camera::free() {
  cudaFree(pos);
  cudaFree(dir);
}
