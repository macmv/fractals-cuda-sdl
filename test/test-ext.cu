#include <iostream>
using namespace std;

class Polygon {
public:
  int width, height;
public:
  void set_values(int a, int b) {
    width = a;
    height = b;
  }
  virtual int area() {
    return 0;
  }
};

class Rectangle: public Polygon {
public:
  int area () {
    return width * height;
  }
};

class Triangle: public Polygon {
public:
  int area () {
    return (width * height / 2);
  }
};

int get_area(Polygon& poly) {
  printf("t: %s\n", typeid(poly).name());
  printf("w: %d\n", poly.width);
  return poly.area();
}

int main () {
  Polygon* rect = new Rectangle();
  Polygon* trgl = new Triangle();
  Polygon* poly = new Polygon();
  rect->set_values(4, 5);
  trgl->set_values(5, 5);
  poly->set_values(6, 5);
  printf("size for rect: %d\n", get_area(*rect));
  printf("size for trgl: %d\n", get_area(*trgl));
  printf("size for poly: %d\n", get_area(*poly));
  return 0;
}
