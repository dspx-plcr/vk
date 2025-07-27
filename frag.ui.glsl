#version 450

layout(binding=0, set=0) uniform usampler2D smplr;

layout(location=0) in vec2 fragPos;
layout(location=0) out vec4 fragCol;

void
main()
{
	fragCol = texture(smplr, fragPos);
}
