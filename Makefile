.SUFFIXES: .glsl .spv

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
	glslc -o $@ \
		-fshader-stage=$$(echo $^ | sed -e 's/.\(model\|ui\).glsl//') $^

clean:
	rm -rf $(OBJS) $(TARGET) $(SHADERS)
