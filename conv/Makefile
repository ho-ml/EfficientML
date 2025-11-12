# Compiler
CC = g++
CC_FLAGS = -O3
SRC = $(wildcard src/*.cpp)

# Include directories
INCLUDE_DIRS = -I./include

# Source Files
TARGET = benchmark
SRC += benchmark.cpp

# Object file
OBJS = $(SRC:.cpp=.o)

# Targets (build)
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(CUDA_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)