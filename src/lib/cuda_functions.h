#include <stdio.h>
#include <mathfu/vector.h>
#include <SDL2/SDL.h>
#include "fractal.h"
#include "camera.h"

#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_

const double DST_MAX = 10;
const double DST_MIN = 0.001;

__global__
void drawPixels(int numPixels,
    Fractal* fractal,
    unsigned char* buffer,
    int numThreads,
    Camera cam,
    unsigned int millis,
    int screen_width,
    int screen_height);

void renderScreen(int numPixels,
    Fractal* fractal,
    Camera* cam,
    unsigned char* buffer,
    int screen_width,
    int screen_height);

#endif
