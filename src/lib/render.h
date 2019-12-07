#include <SDL2/SDL.h>
#include <stdio.h>
#include "fractal.h"
#include "camera.h"
#include "cuda_functions.h"

#ifndef RENDER_H_
#define RENDER_H_

const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;

class Render {
  // things are set to NULL so we can see if there is an issue initializing them in initSDL();
private:
  Fractal* fractal;
  SDL_Window* window = NULL;
  bool alive = true; // when set to false, everything will get closed down
  SDL_Event e;
  Camera* cam;
  SDL_Renderer* renderer = NULL;
  unsigned char* tempBuffer = NULL; // used to copy sharedBuffer to sdl surface
  int bufferSize;
  SDL_Texture* tex;
public:
  unsigned char* sharedBuffer = NULL; // allocated on GPU, is the output of the ray marching

public:
  Render();
  bool isAlive();
  void update();
  void render();
  void quit();
  void updateWindow();
  int getNumPixels();
private:
  bool initSDL();
};

#endif
