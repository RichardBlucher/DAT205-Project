#version 430 core
out vec4 fragColor;

layout(std430, binding = 3) buffer predPosBuffer 
{ 
    vec2 predPos[]; 
};

layout(std430, binding = 4) buffer CellCounts    
{ 
    uint cellCounts[]; 
};

layout(std430, binding = 5) buffer CellParticles 
{ 
    uint cellParticles[]; 
};

uniform uint  particleCount;
uniform uint  gridWidth;
uniform uint  gridHeight;
uniform uint  MAX_PARTICLES_PER_CELL;

uniform float smoothRadius;
uniform float kernelVolume;

uniform vec2  worldMin;
uniform vec2  worldMax;
uniform float targetDensity;

uniform float windowWidth;
uniform float windowHeight;

uniform float densityColorScaling;

float kernel(float radius, float dist)
{
    float value  = max(0.0, radius - dist);
    return value * value * value / kernelVolume;
}

ivec2 PositionToCellCoord(vec2 point)
{
    return ivec2(floor((point - worldMin) / smoothRadius));
}

void main()
{
    vec2 worldPos = mix(worldMin, worldMax, vec2(gl_FragCoord.x / windowWidth, gl_FragCoord.y / windowHeight)); // convert pixel pos to world pos

    ivec2 base = PositionToCellCoord(worldPos);
    float density = 0.0;
    // Works just as for the particles , just for every pixel in the screen now
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            ivec2 cell = base + ivec2(dx, dy);
            if (cell.x < 0 || cell.y < 0)                 
                continue;
            if (cell.x >= int(gridWidth) || cell.y >= int(gridHeight))                 
                continue;

            uint key   = uint(cell.x + cell.y * int(gridWidth));
            uint count = min(cellCounts[key], MAX_PARTICLES_PER_CELL);

            for (uint k = 0; k < count; k++)
            {
                uint j     = cellParticles[key * MAX_PARTICLES_PER_CELL + k];
                float dist = length(predPos[j] - worldPos);
                if (dist < smoothRadius)
                    density += kernel(smoothRadius, dist);
            }
        }
    }
    // map density to color
    float t = clamp(density / (targetDensity * densityColorScaling), 0.0, 1.0);

    vec3 low  = vec3(0.05, 0.1, 0.4);
    vec3 mid  = vec3(0.2,  0.5, 1.0);
    vec3 high = vec3(1.0,  1.0, 1.0);

    vec3 col = t < 0.5
        ? mix(low, mid,  t * 2.0)
        : mix(mid, high, t * 2.0 - 1.0);

    fragColor = vec4(col, 1.0);
}