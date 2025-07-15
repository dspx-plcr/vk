#version 450
layout(location=0) in vec3 pos;
layout(location=0) out vec4 vertCol;

void
main()
{
    gl_Position = vec4(pos, 1.0);
    vertCol = vec4(0.5, 0.0, 0.0, 1.0);
}
