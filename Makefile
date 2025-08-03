.SUFFIXES: .glsl .spv

GLSL = glslc
GFLAGS = --target-env=vulkan1.1 \
	-fshader-stage=$$(echo $^ | sed -e 's/\.[^.]*\.glsl//')
CFLAGS = -std=gnu99 -pedantic -Wall -Wextra -g
LDFLAGS = -lvulkan -lm -lglfw
TARGET = vk

OBJS = \
	main.o
SHADERS = \
	vert.model.spv \
	frag.model.spv \
	vert.ui.spv \
	frag.ui.spv

$(TARGET): $(OBJS) $(SHADERS)
	cc -o $@ $(LDFLAGS) $(OBJS)

main.c: linmath.h
.c.o:
	cc -c $(CFLAGS) $^

.glsl.spv:
	$(GLSL) -o $@ $(GFLAGS) $^

clean:
	rm -rf $(OBJS) $(TARGET) $(SHADERS)
