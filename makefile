
# WARNING: THIS TOOK AN HOUR!!!! DON'T EVER TRY AND EDIT THIS!!!!

CC = nvcc
DEPS = ./mathfu/include
DEPS += $(foreach path,$(shell echo ./mathfu/dependencies/*),$(path)/include)
CFLAGS = -D_REENTRANT -lSDL2 $(foreach dep,$(DEPS),-I$(dep))
# SRC_FILES = src/main.cu
SRC_FILES = $(foreach path,$(shell echo ./src/**/*.cu),$(path))

all: src/main.cu compile
	mkdir -p build
	$(CC) -o build/main objects/src/main.cu.o $(CFLAGS)

compile: $(SRC_FILES)
	mkdir -p objects/src/lib
	echo $(CC) -c $(SRC_FILES) $(CFLAGS)
	$(CC) -c $(SRC_FILES) $(CFLAGS)

test: src/test.cpp all
	$(CC) -o build/test src/test.cpp
