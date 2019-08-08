#include <stdio.h>
#include <mathfu/vector.h>
#include <SDL2/SDL.h>
#include "fractal.h"

#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_

__global__ void drawPixel(int numPixels, Fractal* fractal, unsigned char* buffer, int numThreads, Camera* cam, unsigned int millis);

void renderScreen(int numPixels, Fractal* fractal, Camera* cam, unsigned char* buffer);

#endif
