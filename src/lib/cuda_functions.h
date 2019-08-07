#include <stdio.h>
#include <mathfu/vector.h>
#include <SDL2/SDL.h>
#include "fractal.h"

#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_

__global__ void drawPixel(int numPixels);

void renderScreen(int numPixels, Fractal* fractal, unsigned char* buffer);

#endif
