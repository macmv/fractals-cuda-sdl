#include <SDL2/SDL.h>
#include <vector>
#include <math.h>
#include <set>

#ifndef CAMERA_H_
#define CAMERA_H_

class Camera {
public:
  double* pos; // [0, 1, 2] -> x, y, z respectively
  double* dir; // [0] is 0-1 x rotation, where 0 is 0 deg and 1 is 360 deg, [1] is 0-1 y rot where 1 is straight up and 0 is straight down
  const double fov = 70; // in degrees
  std::set<SDL_Keycode> keys;

public:
  Camera();
  __device__ void getDeltaFrom2D(double x, double y, double* delta); // delta is a len 3 array that is a normalized direction vector
  void free();
  void rotate(double x, double y);
  void translate(double x, double y, double z);
  void update(SDL_Window* window);
  void keyDown(SDL_Keycode key);
  void keyUp(SDL_Keycode key);
};

#endif
