.SUFFIXES: .glsl .spv

CFLAGS = -std=c99 -pedantic -Wall -Wextra -g -O2
LDFLAGS = -lvulkan -lm -lglfw
TARGET = vk

OBJS = \
	main.o
SHADERS = \
	vert.spv \
	frag.spv

$(TARGET): $(OBJS) $(SHADERS)
	cc -o $@ $(LDFLAGS) $(OBJS)

.c.o:
	cc -c $(CFLAGS) $^

.glsl.spv:
	glslc -o $@ -fshader-stage=$$(basename -s .glsl $^) $^

clean:
	rm -rf $(OBJS) $(TARGET) $(SHADERS)
