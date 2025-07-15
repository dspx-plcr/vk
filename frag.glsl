#version 450

layout(location=0) in vec4 vertCol;
layout(location=0) out vec4 fragCol;

void
main()
{
    fragCol = vertCol;
}
