#include "fractal.h"

__device__
double Fractal::DE(double* rayPos) {
  return -2;
}

__device__
double BasicSphere::DE(double* rayPos) {
  return sqrt(-rayPos[0] * -rayPos[0] + -rayPos[1] * -rayPos[1] + -rayPos[2] * -rayPos[2]) - 1;
}
