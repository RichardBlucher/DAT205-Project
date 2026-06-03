#version 330 core
out vec4 color;

void main()
{
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float dist = dot(coord, coord); // draw dot (square)
    if (dist > 1.0) discard; // make dot circle
    color = vec4(1.0, 1.0, 1.0, 1.0);
}