#version 450

layout(push_constant, std430) uniform UBO {
	vec2 uibox;
	vec2 texelbox;
	vec2 iconoff;
};

layout(location=0) in vec2 pos;
layout(location=0) out vec2 fragPos;

void
main()
{
	gl_Position = vec4(2*pos / uibox - 1.0, 0.0, 1.0);
	fragPos = 1.0/2 * (pos - iconoff) / texelbox;
}
