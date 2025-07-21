#version 450

layout(binding=0, set=0) uniform TUBO {
	mat4 transform;
};
layout(binding=1, set=0) uniform MUBO {
	mat4 model;
};
layout(location=0) in vec3 pos;
layout(location=1) in vec3 norm;
layout(location=0) out vec4 vertCol;
layout(location=1) out vec3 vertNorm;
layout(location=2) out vec3 fragPos;

void
main()
{
	gl_Position = transform * vec4(pos, 1.0);
	vertCol = vec4(0.5, 0.0, 0.0, 1.0);
	vertNorm = norm;
	fragPos = (model * vec4(pos, 1.0)).xyz;
}
