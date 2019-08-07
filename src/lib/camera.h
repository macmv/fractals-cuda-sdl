#include <SDL2/SDL.h>
#include <mathfu/vector.h>

#ifndef CAMERA_H_
#define CAMERA_H_

using namespace mathfu;

class Camera {
  private: Vector<double, 3> pos;
  private: Vector<double, 3> dir; // euler rotation in degrees
  private: double xFov; // in degress
  private: double yFov; // in degrees
  private: SDL_Surface* surf;

  public: Camera(SDL_Surface* surf1);
  public: Vector<double, 3> getAngle(int pixel);
};

#endif
