#include <mathfu/vector.h>

#ifndef FRACTAL_H_
#define FRACTAL_H_

using namespace mathfu;

class Fractal {
  protected: Vector<double, 3>* pos;

  public: double DE(Vector<double, 3>* target);
};

class BasicSphere: public Fractal {
  public: double DE(Vector<double, 3>* target);
};

#endif
