
TARGET = fractal
SRC_DIR = src
OBJ_DIR = obj

CU_FILES  = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/**/*.cu)
H_FILES   = $(wildcard $(SRC_DIR)/*.h) $(wildcard $(SRC_DIR)/**/*.h)
OBJ_FILES = $(wildcard $(OBJ_DIR)/*.o) $(wildcard $(OBJ_DIR)/**/*.o)

OBJS = $(patsubst %.cu,$(OBJ_DIR)/%.o,$(CU_FILES))
OBJS := $(subst /src/,/,$(OBJS))

NVCC = nvcc
NVCCFLAGS = -lSDL2
INCLUDES = -Imathfu/include -Imathfu/dependencies/vectorial/include -Imathfu/dependencies/fplutil/include -Imathfu/dependencies/googletest/include

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $?

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(CU_FILES)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc -o $@ $<
