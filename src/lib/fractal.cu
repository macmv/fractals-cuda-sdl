#include "fractal.h"

__device__
double Fractal::DE(double* rayPos) {
  return -2;
}

__device__
double BasicSphere::DE(double* rayPos) {
  return sqrt(pow(0 - rayPos[0], 2) + pow(0 - rayPos[1], 2) + pow(0 - rayPos[2], 2)) - 1;
}
