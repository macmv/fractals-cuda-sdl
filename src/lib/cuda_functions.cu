#include "cuda_functions.h"

__global__ void drawPixel(int numPixels, Fractal* fractal/*, shared buffer */) {
  for (int i = threadIdx.x; i < numPixels; i += threadIdx.x) {
    SDL_RenderDrawPoint(render, SCREEN_WIDTH / 2, i);
  }
}

void renderScreen(Render* render) {
  printf("Drawing screen!");
  SDL_Window* window = SDL_CreateWindow("Fractals", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN );
  drawPixel<<<1, 128>>>(
    render->getNumPixels(),
    render->getFractal()
  /* shared buffer from render */);
  cudaDeviceSynchronize();
  // update render surface to match shared buffer
}
