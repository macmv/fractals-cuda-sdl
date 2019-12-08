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
  SDL_ShowCursor(SDL_DISABLE);
  printf("Initialized camera\n");
}

__device__
void Camera::getDeltaFrom2D(double x, double y, double* delta) {
  // if fov is 70 deg, xDeg will be from -35 to 35 across whole screen
  double xDeg = (x - 0.5) * fov;
  double yDeg = (y - 0.5) * (fov * (screenWidth / screenHeight));
  double cosY = cos((dir[1] + yDeg) / 180 * M_PI);
  double sinY = sin((dir[1] + yDeg) / 180 * M_PI);
  delta[0] = cos((dir[0] * cosY + xDeg) / 180 * M_PI);
  delta[1] = sin((dir[0] * cosY + xDeg) / 180 * M_PI);
  delta[2] = sinY;
}

void Camera::free() {
  cudaFree(pos);
  cudaFree(dir);
}

void Camera::rotate(double x, double y) {
  dir[0] += x;
  dir[1] += y;
  printf("rotation: %f %f\n", dir[0], dir[1]);
}

void Camera::translate(double x, double y, double z) {
  pos[0] += cos(dir[0] / 180 * M_PI) * x;
  pos[0] += cos((dir[0] + 90) / 180 * M_PI) * z;
  pos[1] += y;
  pos[2] += sin(dir[0] / 180 * M_PI) * x;
  pos[2] += sin((dir[0] + 90) / 180 * M_PI) * z;
}

void Camera::update(SDL_Window* window) {
  double speed = 0.01;
  if (keys.find(SDLK_w) != keys.end()) {
    translate(speed, 0, 0);
  }
  if (keys.find(SDLK_s) != keys.end()) {
    translate(-speed, 0, 0);
  }
  if (keys.find(SDLK_d) != keys.end()) {
    translate(0, 0, speed);
  }
  if (keys.find(SDLK_a) != keys.end()) {
    translate(0, 0, -speed);
  }
  if (keys.find(SDLK_SPACE) != keys.end()) {
    translate(0, speed, 0);
  }
  if (keys.find(SDLK_LSHIFT) != keys.end()) {
    translate(0, -speed, 0);
  }
  SDL_GetWindowSize(window, &screenWidth, &screenHeight);
  int x, y;
  SDL_GetMouseState(&x, &y);
  SDL_WarpMouseInWindow(window, screenWidth / 2, screenHeight / 2);
  if (x != 0 || y != 0) {
    int delta_x = screenWidth / 2 - x;
    int delta_y = screenHeight / 2 - y;
    rotate(delta_x / 20.0, delta_y / 20.0);
  }
  // for (auto key = keys.begin(); key != keys.end(); ++key) {
  //   printf("Key: %s\n", SDL_GetKeyName(*key));
  // }
}

void Camera::keyDown(SDL_Keycode key) {
  keys.insert(key);
}

void Camera::keyUp(SDL_Keycode key) {
  keys.erase(key);
}
