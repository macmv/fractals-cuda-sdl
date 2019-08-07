class Fractal;

#include <mathfu/vector.h>

#ifndef FRACTAL_H_
#define FRACTAL_H_

using namespace mathfu;

class Fractal {
  private: Vector<double, 3> pos;
  public: Fractal();

  public: double DE(Vector<double, 3> target);
};

class BasicSphere: public Fractal {
  private: Vector<double, 3> pos;
  public: BasicSphere();

  public: double DE(Vector<double, 3> target);
};

#endif
