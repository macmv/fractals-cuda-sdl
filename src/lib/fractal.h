#include <SDL2/SDL.h>
#include <math.h>

#ifndef FRACTAL_H_
#define FRACTAL_H_

class Fractal {
protected:
  double pos[3] = {0, 0, 0};

public:
  __device__ static double DE(double* rayPos);
};

class BasicSphere : public Fractal {
public:
  __device__ static double DE(double* rayPos);
};

#endif
