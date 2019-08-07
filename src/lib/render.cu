#include "render.h"

Render::Render() {
  fractal = new BasicSphere();
  printf("Initializing SDL...\n");
  alive = initSDL();
  printf("Initialized SDL!\n");

  int size = getNumPixels() * sizeof(float);
  cudaMalloc(&sharedBuffer, size);
}

bool Render::initSDL() {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    return false;
  }
  window = SDL_CreateWindow("Fractals", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
  if (window == NULL) {
    printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
    return false;
  }
  return true;
}

bool Render::isAlive() {
  return alive;
}

void Render::update() {
  while(SDL_PollEvent(&e) != 0) {
    if(e.type == SDL_QUIT) {
      alive = false;
    }
  }
}

void Render::render() {
  renderScreen();
}

void Render::drawPixel() {

}

void Render::quit() {
  SDL_DestroyWindow(window);
  SDL_Quit();
}

int Render::getNumPixels() {
  return SCREEN_WIDTH * SCREEN_HEIGHT;
}

Camera* Render::getCamera() {
  return camera;
}
float* Render::getSharedBuffer() {
  return sharedBuffer;
}

Fractal* Render::getFractal() {
  return fractal;
}
