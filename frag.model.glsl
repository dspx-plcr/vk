#version 450

layout(location=0) in vec4 vertCol;
layout(location=1) in vec3 vertNorm;
layout(location=2) in vec3 fragPos;
layout(location=0) out vec4 fragCol;

void
ambient(in vec3 col, out vec3 res)
{
	float ambientStrength = 0.1;
	vec3 lightColor = vec3(1, 1, 1);
	vec3 ambient = ambientStrength * lightColor;
	res = ambient * col;
}

void
diffuse(in vec3 col, in vec3 norm, in vec3 frag, out vec3 res)
{
	vec3 lightpos = vec3(0.3*1920/2, 0.7*1080/2, 1.0*1000/2);
	vec3 lightdir = normalize(lightpos - frag);
	float diff = max(dot(norm, lightdir), 0.0);
	vec3 lightColor = vec3(1, 1, 1);
	res = diff * lightColor;
}

void
diffuse2(in vec3 col, in vec3 norm, in vec3 frag, out vec3 res)
{
	vec3 lightpos = vec3(-0.3*1920/2, -0.3*1080/2, 0.3*1000/2);
	vec3 lightdir = normalize(lightpos - frag);
	float diff = max(dot(norm, lightdir), 0.0);
	vec3 lightColor = vec3(1, 1, 1);
	res = diff * lightColor;
}

void
main()
{
	vec3 am, df;
	ambient(vertCol.rgb, am);
	diffuse(vertCol.rgb, normalize(vertNorm), fragPos, df);
	vec3 l1 = (am + df) * vertCol.rgb;
	ambient(vertCol.rgb, am);
	diffuse2(vertCol.rgb, normalize(vertNorm), fragPos, df);
	vec3 l2 = (am + df) * vertCol.rgb;
	fragCol = vec4(l1 + l2, 1.0);
}
