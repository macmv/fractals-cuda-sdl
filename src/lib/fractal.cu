#include "fractal.h"

using namespace mathfu;

__device__
double Fractal::DE(double x, double y, double z) {
  return 0;
}

__device__
double BasicSphere::DE(double x, double y, double z) {
  return 0.9;
}
