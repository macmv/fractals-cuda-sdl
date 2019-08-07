CU = nvcc
RM = rm -f
CPPFLAGS = -g $(shell root-config --cflags)
CPPFLAGS += -Imathfu/include -Imathfu/dependencies/vectorial/include -Imathfu/dependencies/fplutil/include -Imathfu/dependencies/googletest/include
CPPFLAGS += -lSDL2

SRCS=src/main.cu src/lib/render.cu src/lib/fractal.cu src/lib/camera.cu src/lib/cuda_functions.cu
OBJS=$(subst .cc,.o,$(SRCS))

all: main

main: $(OBJS)
	$(CU) -o build/main $(OBJS) $(CPPFLAGS) 

main.o: main.cu main.h

render.o: render.cu render.h

fractal.o: fractal.cu fractal.h

camera.o: camera.cu camera.h

cuda_functions.o: cuda_functions.cu cuda_functions.h

%.cu:
	$(CU) $(CPPFLAGS) -c $<

clean:
	$(RM) $(OBJS)