#ifdef _WIN32
extern "C" __declspec(dllexport) unsigned int NvOptimusEnablement = 0x00000001;
#endif

#include <GL/glew.h>
#include <SDL.h>

#include <labhelper.h>

#include <glm/glm.hpp>
using namespace glm;

#include <chrono>

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

///////////////////////////////////////////////////////////////////////////////

void initialize()
{
    g_window = labhelper::init_window_SDL("2D Test");

    glDisable(GL_DEPTH_TEST); // not needed for 2D
}

///////////////////////////////////////////////////////////////////////////////

void display()
{
    glViewport(0, 0, windowWidth, windowHeight);

    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw a single point (our "ball")
    glPointSize(15.0f);

    glBegin(GL_POINTS);
    glVertex2f(ballPos.x, ballPos.y);
    glEnd();
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
    }

    labhelper::shutDown(g_window);
    return 0;
}