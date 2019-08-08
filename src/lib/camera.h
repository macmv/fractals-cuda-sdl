#include <SDL2/SDL.h>
#include <mathfu/vector.h>

#ifndef CAMERA_H_
#define CAMERA_H_

using namespace mathfu;

class Camera {
private:
  Vector<double, 3> pos;
  Vector<double, 3> dir; // euler rotation in degrees
  double xFov; // in degress
  double yFov; // in degrees
  SDL_Surface* surf;

public:
  Camera(SDL_Surface* surf1);
  Vector<double, 3> getAngle(int pixel);
  __device__ Vector<double, 3> getPoint(int x, int y);
};

#endif
