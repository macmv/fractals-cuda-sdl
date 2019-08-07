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
  private: Fractal* fractal;
  private: SDL_Window* window = NULL;
  private: bool alive = true;
  private: SDL_Event e;
  private: Camera* cam;
  private: SDL_Renderer* renderer;
  public: unsigned char* sharedBuffer;

  public: Render();
  private: bool initSDL();
  public: bool isAlive();
  public: void update();
  public: void render();
  public: void updateSurface();
  public: void quit();
  public: int getNumPixels();
  public: Camera* getCamera();
  public: Fractal* getFractal();
};

#endif
