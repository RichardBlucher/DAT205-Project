#ifdef _WIN32
extern "C" __declspec(dllexport) unsigned int NvOptimusEnablement = 0x00000001;
#endif

#include <iostream>
using namespace std;

#include <GL/glew.h>
#include <SDL.h>

#include <labhelper.h>
#include <imgui.h>
#include <perf.h>

#include <math.h>

#include <glm/glm.hpp>
using namespace glm;

#include <chrono>
#include <vector>

struct Ball {
    vec2 pos;
    vec2 vel;
};

std::vector<Ball> balls;
std::vector<float> densities;
// Window
SDL_Window* g_window = nullptr;
int windowWidth = 800;
int windowHeight = 600;

// Time
float currentTime = 0.0f;
float previousTime = 0.0f;
float deltaTime = 0.0f;

// Simple "ball"
vec2 ballPos(0.0f, 0.0f);
float speed = 1.5f;
const float mass = 1.0f;

// Constants
vec2 gravity(0.0f, -0.0f);
float dampening = 0.8f;

// Simulation constants
float smoothRadius = 0.3f;

float targetDensity = 2.5f;
float pressureMultiplier = 0.5f;


///////////////////////////////////////////////////////////////////////////////
GLuint vao = 0;
GLuint vbo = 0;
GLuint shaderProgram = 0;

void initSimpleStuff()
{
    // Triangle vertices (x, y)
    float vertices[] = {
        -0.2f, -0.2f,
         0.2f, -0.2f,
         0.0f,  0.2f
    };

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 1000 * sizeof(vec2), nullptr, GL_DYNAMIC_DRAW);

    // Simple shader
    const char* vs = R"(
    #version 330 core
    layout(location = 0) in vec2 pos;

    void main() {
        gl_Position = vec4(pos, 0.0, 1.0);
        gl_PointSize = 10.0;
    }
)";

    const char* fs = R"(
    #version 330 core
    out vec4 color;

    void main()
    {
        vec2 coord = gl_PointCoord * 2.0 - 1.0; // map [0,1] -> [-1,1]
        float dist = dot(coord, coord);         // distance^2 from center

        float alpha = 1.0 - smoothstep(0.9, 1.0, dist);

        if(dist > 1.0)
            discard; // outside circle

        color = vec4(1.0, 1.0, 1.0, alpha);
    }
)";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vs, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fs, nullptr);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
}
void initialize()
{
    g_window = labhelper::init_window_SDL("Bouncing Balls");

    glDisable(GL_DEPTH_TEST);

    initSimpleStuff();

    // Create some balls
    for (int i = 0; i < 1000; i++)
    {
        Ball b;
        b.pos = vec2(
            (rand() / float(RAND_MAX)) * 2.0f - 1.0f,
            (rand() / float(RAND_MAX)) * 2.0f - 1.0f
        );

        /*b.vel = vec2(
            (rand() / float(RAND_MAX)) * 0.5f - 0.25f,
            (rand() / float(RAND_MAX)) * 0.5f - 0.25f
        );*/
        b.vel = vec2(0.0f, 0.0f);

        balls.push_back(b);
    }
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

float kernel(float radius, float dist)
{
    float volume = M_PI * radius * radius * radius * radius / 2.0f;
    float value = std::max(0.0f, radius - dist);
    return value * value * value / volume;
}

float kernelDerivative(float radius, float dist)
{
    if (dist > radius)
    {
        return 0;
    }
    return -3.0f * (radius - dist) * (radius - dist);

}

float calculateDensity(vec2 samplePoint)
{
    float density = 0.0f;
    

    for (auto& b : balls)
    {
        float dist = length(b.pos - samplePoint);
        float influence = kernel(smoothRadius, dist);
        density += mass * influence;

        
    }
    return density;
}

void updateDensities()
{
    size_t n = balls.size();
    densities.resize(n);
    size_t i = 0;
    for (auto& b : balls)
    {
        float density = calculateDensity(b.pos);
        
        densities[i] = density;
        i++;
    }
}

float densityToPressure(float density)
{
    float densityError = density - targetDensity;
    float pressure = densityError * pressureMultiplier;
    return pressure;
}

float sharedPressure(float densityA, float densityB)
{
    float pressureA = densityToPressure(densityA);
    float pressureB = densityToPressure(densityB);
    return (pressureA + pressureB) / 2.0f;
}

vec2 calculateDensityGradient(int idxBall)
{
    vec2 gradient(0.0f, 0.0f);
    
    size_t i = 0;
    for (auto& b : balls)
    {

        float dist = length(b.pos - balls[idxBall].pos);
        
        if (dist > 0.0f)
        {
            vec2 dir = (b.pos - balls[idxBall].pos) / dist;
            
            float slope = kernelDerivative(smoothRadius, dist);
            float density = densities[i];

            gradient += sharedPressure(density, densities[idxBall]) * dir * slope * mass / density; //force
            //cout << gradient.x << "\n";
        }

    }
    
    return gradient;
}



void updateBalls()
{
    updateDensities();
    size_t i = 0;
    for (auto& b : balls)
    {

        b.vel += gravity * deltaTime;
        b.vel += calculateDensityGradient(i) / densities[i] * deltaTime;
        
        b.pos += b.vel * deltaTime;

        // Bounce on edges (-1 to 1 in OpenGL)
        if (b.pos.x < -1.0f || b.pos.x > 1.0f)
            b.vel.x *= -1.0f * dampening;

        if (b.pos.y < -1.0f || b.pos.y > 1.0f)
            b.vel.y *= -1.0f * dampening;
        i++;
    }
}
///////////////////////////////////////////////////////////////////////////////

void display()
{
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    updateBalls();

    // Collect positions
    std::vector<vec2> positions;
    for (auto& b : balls)
        positions.push_back(b.pos);

    // Upload to GPU
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
        positions.size() * sizeof(vec2),
        positions.data());

    glUseProgram(shaderProgram);
    glBindVertexArray(vao);

    glPointSize(10.0f); // ball size
    glDrawArrays(GL_POINTS, 0, positions.size());
}

///////////////////////////////////////////////////////////////////////////////

bool handleEvents()
{
    SDL_Event event;
    bool quit = false;

    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_QUIT)
            quit = true;

        if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)
            quit = true;
    }

    // Keyboard input (continuous)
    const uint8_t* state = SDL_GetKeyboardState(nullptr);

    if (state[SDL_SCANCODE_W]) ballPos.y += speed * deltaTime;
    if (state[SDL_SCANCODE_S]) ballPos.y -= speed * deltaTime;
    if (state[SDL_SCANCODE_A]) ballPos.x -= speed * deltaTime;
    if (state[SDL_SCANCODE_D]) ballPos.x += speed * deltaTime;

    return quit;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    initialize();

    auto startTime = std::chrono::system_clock::now();
    bool quit = false;

    while (!quit)
    {
        // Time update
        std::chrono::duration<float> t =
            std::chrono::system_clock::now() - startTime;

        previousTime = currentTime;
        currentTime = t.count();
        deltaTime = currentTime - previousTime;

        // Events
        quit = handleEvents();

        // Render
        display();
        

        SDL_GL_SwapWindow(g_window);
        //cout << calculateDensity(vec2(0.5f, 0.5f)) << "\n";
        cout << deltaTime << "\n";
    }

    labhelper::shutDown(g_window);
    return 0;
}