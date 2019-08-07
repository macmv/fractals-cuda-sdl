#include <stdio.h>
#include <mathfu/vector.h>
#include <SDL2/SDL.h>
#include "render.h"

#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_

__global__ void drawPixel(int numPixels);

__global__ void drawPixel(int numPixels);

void renderScreen(Render* render);

#endif
