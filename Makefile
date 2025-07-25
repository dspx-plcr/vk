.SUFFIXES: .glsl .spv

CFLAGS = -std=gnu99 -pedantic -Wall -Wextra -g
LDFLAGS = -lvulkan -lm -lglfw
TARGET = vk

OBJS = \
	main.o
SHADERS = \
	vert.spv \
	frag.spv

$(TARGET): $(OBJS) $(SHADERS)
	cc -o $@ $(LDFLAGS) $(OBJS)

main.c: linmath.h
.c.o:
	cc -c $(CFLAGS) $^

.glsl.spv:
	glslc -o $@ -fshader-stage=$$(basename -s .glsl $^) $^

clean:
	rm -rf $(OBJS) $(TARGET) $(SHADERS)
