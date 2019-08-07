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
  // things are set to NULL so we can see if there is an issue initializing them in bool initSDL();
  private: Fractal* fractal;
  private: SDL_Window* window = NULL;
  private: bool alive = true; // when set to false, everything will get closed down
  private: SDL_Event e;
  private: Camera* cam;
  private: SDL_Renderer* renderer = NULL;
  public: unsigned char* sharedBuffer = NULL; // allocated on GPU, is the output of the ray marching
  private: unsigned char* tempBuffer = NULL; // used to copy sharedBuffer to sdl surface
  private: int bufferSize;
  private: SDL_Texture* tex;

  public: Render();
  private: bool initSDL();
  public: bool isAlive();
  public: void update();
  public: void render();
  public: void updateWindow();
  public: void quit();
  public: int getNumPixels();
  public: Camera* getCamera();
  public: Fractal* getFractal();
};

#endif
