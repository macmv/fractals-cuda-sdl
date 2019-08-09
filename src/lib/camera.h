#include <SDL2/SDL.h>
#include <vector>

#ifndef CAMERA_H_
#define CAMERA_H_

class Camera {
public:
  double* pos; // [0, 1, 2] -> x, y, z respectively
  double* dir; // [0] is 0-1 x rotation, where 0 is 0 deg and 1 is 360 deg, [1] is 0-1 y rot where 1 is straight up and 0 is straight down
  const double xFov = 1 / 16.0; // in fractions of 360
  const double yFov = 1 / 16.0; // in fractions of 360

public:
  Camera();
  __device__ void getDeltaFrom2D(double x, double y, double* delta); // delta is a len 3 array that is a normalized direction vector
  void free();
};

#endif
