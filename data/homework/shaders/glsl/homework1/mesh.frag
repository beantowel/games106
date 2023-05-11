#version 450

layout (set = 1, binding = 0) uniform sampler2D baseColorMap;
layout (set = 1, binding = 1) uniform sampler2D normalMap;
layout (set = 1, binding = 2) uniform sampler2D metallicRoughnessMap;
layout (set = 1, binding = 3) uniform sampler2D emissiveMap;
layout (set = 1, binding = 4) uniform sampler2D brdfLUT;
layout (set = 1, binding = 5) uniform samplerCube prefilteredMap;
layout (set = 1, binding = 6) uniform MaterialData
{
	vec4 baseColorFactor;
	vec4 emissiveFactor;
	float metallicFactor;
	float roughnessFactor;
} materialData;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;
layout (location = 5) in vec4 inTangent;

layout (location = 0) out vec4 outFragColor;

#define PI 3.1415926535897932384626433832795

// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2)/(PI * denom*denom);
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;
	float GL = dotNL / (dotNL * (1.0 - k) + k);
	float GV = dotNV / (dotNV * (1.0 - k) + k);
	return GL * GV;
}

// Fresnel function ----------------------------------------------------
vec3 F_Schlick(float cosTheta, float metallic, vec3 color)
{
	vec3 F0 = mix(vec3(0.04), color, metallic); // * material.specular
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 F_SchlickR(float cosTheta, float roughness, float metallic, vec3 color)
{
	vec3 F0 = mix(vec3(0.04), color, metallic); // * material.specular
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 BRDF(vec3 L, vec3 V, vec3 N, float metallic, float roughness, vec3 albedo)
{
	// Precalculate vectors and dot products
	vec3 H = normalize (V + L);
	float dotNV = clamp(dot(N, V), 0.0, 1.0);
	float dotNL = clamp(dot(N, L), 0.0, 1.0);
	float dotNH = clamp(dot(N, H), 0.0, 1.0);

	// Light color fixed
	vec3 lightColor = vec3(1.0);

	if (dotNL > 0.0) {
		// D = Normal distribution (Distribution of the microfacets)
		float D = D_GGX(dotNH, roughness);
		// G = Geometric shadowing term (Microfacets shadowing)
		float G = G_SchlicksmithGGX(dotNL, dotNV, roughness);
		// F = Fresnel factor (Reflectance depending on angle of incidence)
		vec3 F = F_Schlick(dotNV, metallic, albedo);
		vec3 spec = D * F * G / (4.0 * dotNL * dotNV + 0.001);
		return spec * dotNL * lightColor;
	}

	return vec3(0.0);
}

vec3 prefilteredReflection(vec3 R, float roughness)
{
	const float MAX_REFLECTION_LOD = 9.0; // todo: param/const
	float lod = roughness * MAX_REFLECTION_LOD;
	float lodf = floor(lod);
	float lodc = ceil(lod);
	vec3 a = textureLod(prefilteredMap, R, lodf).rgb;
	vec3 b = textureLod(prefilteredMap, R, lodc).rgb;
	return mix(a, b, lod - lodf);
}

vec3 PBR(vec3 L, vec3 V, vec3 N, vec4 albedo, float metallic, float roughness)
{
	vec3 R = reflect(L, N);

	vec3 Lo = vec3(0.0);
	Lo += BRDF(L, V, N, metallic, roughness, albedo.xyz);

	// GI
	vec2 brdfIntg = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
	vec3 reflection = prefilteredReflection(R, roughness).rgb;
	// 不想玩辣!
	// vec3 irradiance = texture(samplerIrradiance, N).rgb;
	vec3 irradiance = PI * vec3(0.5);
	vec3 diffuse = irradiance * albedo.xyz;
	vec3 F = F_SchlickR(max(dot(N, V), 0.0), roughness, metallic, albedo.xyz);
	vec3 specular = reflection * (F * brdfIntg.x + brdfIntg.y);
	// Ambient part
	vec3 kD = 1.0 - F;
	kD *= 1.0 - metallic;
	vec3 ambient = (kD * diffuse + specular);

	return ambient + Lo;
}

void main()
{
	vec4 albedo = texture(baseColorMap, inUV) * materialData.baseColorFactor;
	vec3 normal = texture(normalMap, inUV).xyz;
	vec3 emissive = texture(emissiveMap, inUV).xyz * materialData.emissiveFactor.xyz;
	vec4 metalRough = texture(metallicRoughnessMap, inUV);
	float metallic = metalRough.x;
	float roughness = metalRough.y;

	vec3 N = normalize(inNormal);
	vec3 T = normalize(inTangent.xyz);
	vec3 B = cross(inNormal, inTangent.xyz) * inTangent.w;
	mat3 TBN = mat3(T, B, N);
	N = TBN * normalize(normal * 2.0 - vec3(1.0));

	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);

	vec3 c = PBR(L, V, N, albedo, metallic, roughness) + emissive;
	outFragColor = vec4(c, albedo.w);
	// outFragColor = vec4(albedo.www, 1.0);
	// outFragColor = vec4(emissive + 0.5, 1.0);
	// outFragColor = vec4(metalRough.xyz, 1.0);
	// outFragColor = vec4(normal, 1.0);
}