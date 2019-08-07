#include "fractal.h"

using namespace mathfu;

double Fractal::DE(Vector<double, 3>* target) {
  return 0;
}

double BasicSphere::DE(Vector<double, 3>* target) {
  return target->Distance(*target, *pos);
}
