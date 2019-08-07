#include "render.h"

Render::Render() {
  fractal = new BasicSphere();
  printf("Initializing SDL...\n");
  alive = initSDL();
  printf("Initialized SDL!\n");

  int size = getNumPixels() * sizeof(unsigned char) * 3; // because rgb
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
  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  if (renderer == NULL) {
    printf("SDL_Render could not be created! SDL_Error: %s\n", SDL_GetError());
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

void Render::updateSurface() {
  int size = getNumPixels() * sizeof(unsigned char) * 3; // because rgb
  for (int pixel = 0; pixel < size; pixel += 3) {
    SDL_SetRenderDrawColor(renderer,
      sharedBuffer[pixel],
      sharedBuffer[pixel] + 1,
      sharedBuffer[pixel] + 2,
      0xFF);
    SDL_RenderDrawPoint(renderer, pixel % SCREEN_WIDTH, pixel / SCREEN_WIDTH);
  }
  SDL_RenderPresent(renderer);
}

void Render::render() {
  renderScreen(getNumPixels(), fractal, sharedBuffer);
  updateSurface();
}

void Render::quit() {
  SDL_DestroyWindow(window);
  SDL_Quit();
}

int Render::getNumPixels() {
  return SCREEN_WIDTH * SCREEN_HEIGHT;
}

Camera* Render::getCamera() {
  return cam;
}

Fractal* Render::getFractal() {
  return fractal;
}
