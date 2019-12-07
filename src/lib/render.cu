#include "render.h"

Render::Render() {
  fractal = new BasicSphere();
  printf("Initializing SDL...\n");
  alive = initSDL();
  printf("Initialized SDL\n");

  printf("Allocating memory...\n");
  bufferSize = getNumPixels() * sizeof(unsigned char) * 4; // because rgba
  printf("Size: %d\n", bufferSize);
  tempBuffer = (unsigned char*) malloc(bufferSize);
  cudaMalloc((void**) &sharedBuffer, bufferSize);
  cudaMemset(sharedBuffer, 0, bufferSize);
  cudaDeviceSynchronize();
  printf("Allocated memory\n");

  cam = new Camera();
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
  tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);
  return true;
}

bool Render::isAlive() {
  return alive;
}

void Render::update() {
  while (SDL_PollEvent(&e) != 0) {
    if (e.type == SDL_QUIT) {
      alive = false;
    } else if (e.type == SDL_KEYDOWN) {
      cam->keyDown(e.key.keysym.sym);
    } else if (e.type == SDL_KEYUP) {
      cam->keyUp(e.key.keysym.sym);
    }
  }
  cam->update();
}

void Render::updateWindow() {
  cudaMemcpy(tempBuffer, sharedBuffer, bufferSize, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  SDL_UpdateTexture(tex, NULL, tempBuffer, SCREEN_WIDTH * 4);
  SDL_RenderCopy(renderer, tex, NULL, NULL);
  SDL_RenderPresent(renderer);
  SDL_UpdateWindowSurface(window);
}

void Render::render() {
  renderScreen(getNumPixels(), fractal, cam, sharedBuffer, SCREEN_WIDTH, SCREEN_HEIGHT);
  updateWindow();
}

void Render::quit() {
  cam->free();
  free(tempBuffer);
  cudaFree(sharedBuffer);
  SDL_DestroyTexture(tex);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

int Render::getNumPixels() {
  return SCREEN_WIDTH * SCREEN_HEIGHT;
}
