#include "main.h"

int main(int argc, char** args) {
  Render* render = new Render();

  while (render->isAlive()) {
    render->update();
    render->render();
  }
  render->quit();

  return 0;
}
