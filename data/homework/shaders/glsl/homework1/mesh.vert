#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inColor;
layout (location = 4) in vec4 inTangent;

layout (set = 0, binding = 0) uniform UBOScene
{
	mat4 projection;
	mat4 view;
	vec4 lightPos;
	vec4 viewPos;
} uboScene;

layout (set = 2, binding = 0) uniform UboInstance
{
	mat4 model;
} uboInstance;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec3 outViewVec;
layout (location = 4) out vec3 outLightVec;
layout (location = 5) out vec4 outTangent;

void main()
{
	outNormal = inNormal;
	outTangent = inTangent;
	outColor = inColor;
	outUV = inUV;

	// Calculate skinned matrix from weights and joint indices of the current vertex
	gl_Position = uboScene.projection * uboScene.view * uboInstance.model * vec4(inPos.xyz, 1.0);
	outNormal = normalize(transpose(inverse(mat3(uboScene.view * uboInstance.model))) * inNormal);

	vec4 pos = uboScene.view * vec4(inPos, 1.0);
	vec3 lPos = mat3(uboScene.view) * uboScene.lightPos.xyz;
	outLightVec = lPos - pos.xyz;
	outViewVec = uboScene.viewPos.xyz - pos.xyz;
}