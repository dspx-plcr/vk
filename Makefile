CFLAGS = -std=c99 -pedantic -Wall -Wextra -g -Os
LDFLAGS = -lvulkan -lm
TARGET = vk

OBJS = \
	main.o

$(TARGET): $(OBJS)
	cc -o $@ $(LDFLAGS) $^

.c.o:
	cc -c $(CFLAGS) $^

clean:
	rm -rf $(OBJS) $(TARGET)
