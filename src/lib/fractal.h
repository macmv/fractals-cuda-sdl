#include <mathfu/vector.h>
#include <SDL2/SDL.h>
#include <math.h>

#ifndef FRACTAL_H_
#define FRACTAL_H_

using namespace mathfu;

typedef double(*DEfunc)(double x, double y, double z);

class Fractal {
protected:
  Vector<double, 3>* pos;

public:
  __device__ double DE(double x, double y, double z);
};

class BasicSphere: public Fractal {
public:
  __device__ double DE(double x, double y, double z);
};

#endif
