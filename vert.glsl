#version 450
#extension GL_EXT_debug_printf : enable

layout(binding=0, set=0) uniform UBO {
	mat4 transform;
};
layout(location=0) in vec3 pos;
layout(location=1) in vec3 norm;
layout(location=0) out vec4 vertCol;

void
main()
{
	gl_Position = transform * vec4(pos, 1.0);
	debugPrintfEXT("position [%v4f]\n", gl_Position);
	vertCol = vec4(0.5, 0.0, 0.0, 1.0);
}
