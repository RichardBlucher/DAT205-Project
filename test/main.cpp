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

#include <algorithm>
#include <climits>

// Simulation constants
float smoothRadius = 0.05f;
float smoothRadiusSq = smoothRadius * smoothRadius;
float kernelVolume = 3.141592 * pow(smoothRadius, 4) / 2.0f;

float targetDensity = 6.5f;
float pressureMultiplier = 0.0f;

int n_particles = 50000;
std::vector<float> densities(n_particles, 0.0f);
// Particles
struct Particle
{
    vec2 position;
    vec2 velocity;
};

std::vector<Particle> particles(n_particles);

std::vector<vec2> positions(n_particles);
std::vector<vec2> velocities(n_particles);


std::vector<vec2> predictedPositions(n_particles);
float predictionFactor = 0.0f;
// Window
SDL_Window* g_window = nullptr;
int windowWidth = 1280;
int windowHeight = 720;

// Time
float currentTime = 0.0f;
float previousTime = 0.0f;
float deltaTime = 0.0f;
float simulationMs = 0.0f;



// Constants
vec2 gravity(0.0f, -0.0f);
float dampening = 1.0f;
const float mass = 1.0f;
float viscosityStrength = 0.1f;

// Debug
static bool paused = true;

// For plots
const int graphSize = 200;

std::vector<float> avgDensityHistory(graphSize, 0.0f);
std::vector<float> avgVelocityHistory(graphSize, 0.0f);

int graphOffset = 0;

// Mouse interactions
vec2 mousePos(0.0f);

bool mouseDown = false;

float mouseForceRadius = 0.3f;
float mouseForceStrength = 0.0f;

///////////////////////////////////////////////////////////////////////////////
// Compute shader stuff
#include <fstream>
#include <sstream>

GLuint forcesProgram = 0;

GLuint particleBuffer;


GLuint densityProgram = 0;
GLuint densityBuffer;
GLuint pressureBuffer;

GLuint predictProgram = 0;
GLuint predPosBuffer;



// Uniform grid
GLuint clearGridProgram = 0;
GLuint buildGridProgram = 0;
GLuint cellCountBuffer;

GLuint cellParticleBuffer;
GLuint overflowBuffer;

float worldMinX = -1.0f, worldMinY = -1.0f;
float worldMaxX = 1.0f, worldMaxY = 1.0f;

uint gridWidth = uint(ceil((worldMaxX - worldMinX) / smoothRadius)) + 1;
uint gridHeight = uint(ceil((worldMaxY - worldMinY) / smoothRadius)) + 1;
unsigned int numCells = gridWidth * gridHeight;

unsigned int MAX_PARTICLES_PER_CELL = 100 * ceil(float(n_particles) / numCells);

std::string loadFile(const char* path)
{
    std::ifstream file(path);

    if (!file.is_open())
    {
        std::cout << "Failed to open: "
            << path << std::endl;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

GLuint createComputeShader(const char* path)
{
    std::string sourceStr = loadFile(path);
    const char* source = sourceStr.c_str();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);

    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);


    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);

        std::cout << "Compute shader compile error:\n"
            << infoLog << std::endl;
    }

    GLuint program = glCreateProgram();

    glAttachShader(program, shader);
    glLinkProgram(program);

    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);

    if (!linked)
    {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);

        std::cout << "Compute program link error:\n"
            << infoLog << std::endl;
    }

    glDeleteShader(shader);

    return program;
}

void runForcesShader()
{
    glUseProgram(forcesProgram);

    // variables sent to compute shader
    GLint dtLoc =
        glGetUniformLocation(forcesProgram, "dt");

    glUniform1f(dtLoc, deltaTime);

    GLint countLoc = glGetUniformLocation(forcesProgram, "particleCount");
    glUniform1ui(countLoc, n_particles);

    GLint gravityLoc = glGetUniformLocation(forcesProgram, "gravity");
    glUniform2f(gravityLoc, gravity.x, gravity.y);

    glUniform1f(glGetUniformLocation(forcesProgram, "dampening"), dampening);


    // mouse force
    GLint mousePosLoc = glGetUniformLocation(forcesProgram, "mousePos");
    glUniform2f(mousePosLoc, mousePos.x, mousePos.y);
    
    GLint mouseForceRadiusLoc = glGetUniformLocation(forcesProgram, "mouseForceRadius");
    glUniform1f(mouseForceRadiusLoc, mouseForceRadius);

    GLint mouseForceStrengthLoc = glGetUniformLocation(forcesProgram, "mouseForceStrength");
    glUniform1f(mouseForceStrengthLoc, mouseForceStrength);
    
    GLint mouseDownLoc = glGetUniformLocation(forcesProgram, "mouseDown");
    glUniform1ui(mouseDownLoc, mouseDown);

    // pressure force
    glUniform1f(glGetUniformLocation(forcesProgram, "smoothRadius"), smoothRadius);
    glUniform1f(glGetUniformLocation(forcesProgram, "smoothRadiusSq"), smoothRadiusSq);
    glUniform1f(glGetUniformLocation(forcesProgram, "kernelVolume"), kernelVolume);

    // viscosity force
    glUniform1f(glGetUniformLocation(forcesProgram, "viscosityStrength"), viscosityStrength);

    glUniform1ui(glGetUniformLocation(forcesProgram, "gridWidth"), gridWidth);
    glUniform1ui(glGetUniformLocation(forcesProgram, "gridHeight"), gridHeight);

    glUniform2f(glGetUniformLocation(forcesProgram, "worldMin"), worldMinX, worldMinY);
    glUniform2f(glGetUniformLocation(forcesProgram, "worldMax"), worldMaxX, worldMaxY);


    glUniform1ui(glGetUniformLocation(forcesProgram, "MAX_PARTICLES_PER_CELL"), MAX_PARTICLES_PER_CELL);

    glDispatchCompute(
        (n_particles + 255) / 256,
        1,
        1
    );

    glMemoryBarrier(
        GL_SHADER_STORAGE_BARRIER_BIT |
        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT
    );
}

void runDensityShader()
{
    glUseProgram(densityProgram);

    // variables sent to density shader (i like this way better, change other one if i have time)
    glUniform1ui(glGetUniformLocation(densityProgram, "particleCount"), n_particles);

    glUniform1f(glGetUniformLocation(densityProgram, "smoothRadius"), smoothRadius);
    glUniform1f(glGetUniformLocation(densityProgram, "smoothRadiusSq"), smoothRadiusSq);
    glUniform1f(glGetUniformLocation(densityProgram, "kernelVolume"), kernelVolume);

    glUniform1f(glGetUniformLocation(densityProgram, "targetDensity"), targetDensity);

    glUniform1f(glGetUniformLocation(densityProgram, "pressureMultiplier"), pressureMultiplier);

    glUniform1ui(glGetUniformLocation(densityProgram, "gridWidth"), gridWidth);
    glUniform1ui(glGetUniformLocation(densityProgram, "gridHeight"), gridHeight);


    glUniform1ui(glGetUniformLocation(densityProgram, "MAX_PARTICLES_PER_CELL"), MAX_PARTICLES_PER_CELL);

    glUniform2f(glGetUniformLocation(densityProgram, "worldMin"), worldMinX, worldMinY);
    glUniform2f(glGetUniformLocation(densityProgram, "worldMax"), worldMaxX, worldMaxY);

    glDispatchCompute(
        (n_particles + 255) / 256,
        1,
        1
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void runPredictShader()
{
    glUseProgram(predictProgram);

    // variables sent to predict shader (i like this way better, change other one if i have time)
    glUniform1ui(glGetUniformLocation(predictProgram, "particleCount"), n_particles);

    GLint dtLoc =
        glGetUniformLocation(predictProgram, "dt");

    glUniform1f(dtLoc, deltaTime);

    glUniform1f(glGetUniformLocation(predictProgram, "predictionFactor"), predictionFactor);


    glDispatchCompute(
        (n_particles + 255) / 256,
        1,
        1
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
///////////////////////////////////////////////////////////////////////////////


void runClearGridShader()
{
    glUseProgram(clearGridProgram);

    glUniform1ui(glGetUniformLocation(clearGridProgram, "numCells"), numCells);



    glDispatchCompute(
        (numCells + 255) / 256,
        1,
        1
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void runBuildGridShader()
{
    
    glUseProgram(buildGridProgram);

    glUniform1ui(glGetUniformLocation(buildGridProgram, "particleCount"), n_particles);

    glUniform1ui(glGetUniformLocation(buildGridProgram, "gridWidth"), gridWidth);

    glUniform1ui(glGetUniformLocation(buildGridProgram, "gridHeight"), gridHeight);

    glUniform1f(glGetUniformLocation(buildGridProgram, "smoothRadius"), smoothRadius);

    glUniform1ui(
        glGetUniformLocation(buildGridProgram, "MAX_PARTICLES_PER_CELL"),
        MAX_PARTICLES_PER_CELL
    );

    glUniform2f(glGetUniformLocation(buildGridProgram, "worldMin"), worldMinX, worldMinY);
    glUniform2f(glGetUniformLocation(buildGridProgram, "worldMax"), worldMaxX, worldMaxY);


    glDispatchCompute(
        (n_particles + 255) / 256,
        1,
        1
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

///////////////////////////////////////////////////////////////////////////////
GLuint vao = 0;
GLuint shaderProgram = 0;

void initShaderStuff()
{


    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, particleBuffer);

    // Simple shader
    const char* vs = R"(
    #version 330 core
    layout(location = 0) in vec2 pos;
    
    void main() {
        gl_Position = vec4(pos, 0.0, 1.0);
        gl_PointSize = 3.0;
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

        color = vec4(1.0, 1.0, 1.0, 1.0);
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

    glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Particle),
        (void*)0
    );
    glEnableVertexAttribArray(0);

}
void initialize()
{
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_PROFILE_MASK,
        SDL_GL_CONTEXT_PROFILE_CORE
    );

    g_window = labhelper::init_window_SDL("SPH with " + std::to_string(n_particles) + " particles", windowWidth, windowHeight);
    SDL_GL_SetSwapInterval(0); // disable vsync
    glDisable(GL_DEPTH_TEST);

    std::cout << "OpenGL version: "
        << glGetString(GL_VERSION)
        << std::endl;

    

    // Create some balls
    for (int i = 0; i < n_particles; i++)
    {

        particles[i].position = vec2(
            (rand() / float(RAND_MAX)) * 2.0f - 1.0f,
            (rand() / float(RAND_MAX)) * 2.0f - 1.0f
        );

        /*particles[i].velocity = vec2(
            (rand() / float(RAND_MAX)) * 0.5f - 0.25f,
            (rand() / float(RAND_MAX)) * 0.5f - 0.25f
        );*/
        particles[i].velocity = vec2(0.0f, 0.0f);


    }
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    ////// Compute shader init
    

    forcesProgram =
        createComputeShader("forces3.comp");

    densityProgram = createComputeShader("density3.comp");

    predictProgram = createComputeShader("predict.comp");

    

    clearGridProgram = createComputeShader("clearGrid.comp");
    buildGridProgram = createComputeShader("buildGrid.comp");


    // Particles
    glGenBuffers(1, &particleBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, particleBuffer);

    glBufferData(
        GL_ARRAY_BUFFER,
        particles.size() * sizeof(Particle),
        particles.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        particleBuffer
    ); // compute shader writes directly to render buffer

    

    // Densities
    std::vector<float> densities(n_particles, 0.0f);

    glGenBuffers(1, &densityBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, densityBuffer);

    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        densities.size() * sizeof(float),
        densities.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        densityBuffer
    );

    // Pressure
    std::vector<float> pressures(n_particles, 0.0f);

    glGenBuffers(1, &pressureBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pressureBuffer);

    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        pressures.size() * sizeof(float),
        pressures.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        2,
        pressureBuffer
    );

    // Predicted positions
    std::vector<vec2> predPos(n_particles);

    glGenBuffers(1, &predPosBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, predPosBuffer);

    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        predPos.size() * sizeof(vec2),
        predPos.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        3,
        predPosBuffer
    );

    

    /////// Uniform grid
    // Cell counts
    std::vector<unsigned int> emptyCounts(
        numCells,
        0
    );

    glGenBuffers(1, &cellCountBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,
        cellCountBuffer);

    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        numCells * sizeof(unsigned int),
        emptyCounts.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        4,
        cellCountBuffer
    );

    // Cell particles
    std::vector<unsigned int> emptyParticles(
        numCells* MAX_PARTICLES_PER_CELL,
        0
    );

    glGenBuffers(1, &cellParticleBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,
        cellParticleBuffer);

    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        emptyParticles.size() * sizeof(unsigned int),
        emptyParticles.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        5,
        cellParticleBuffer
    );

    // Cell particles overflow debug

    unsigned int zero = 0;

    glGenBuffers(1, &overflowBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, overflowBuffer);

    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        sizeof(unsigned int),
        &zero,
        GL_DYNAMIC_DRAW
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        6,
        overflowBuffer
    );


    initShaderStuff();
}
















///////////////////////////////////////////////////////////////////////////////

void display()
{
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);



    auto t0 = chrono::high_resolution_clock::now();
    if (!paused)
    {
        runPredictShader();
        runClearGridShader();

        runBuildGridShader();
        
        runDensityShader();
        runForcesShader();
        
        //// Overflow check
        static float overflowTimer = 0.0f;

        overflowTimer += deltaTime;

        if (overflowTimer >= 1.0f)
        {
            overflowTimer = 0.0f;

            unsigned int overflows = 0;

            glBindBuffer(
                GL_SHADER_STORAGE_BUFFER,
                overflowBuffer
            );

            glGetBufferSubData(
                GL_SHADER_STORAGE_BUFFER,
                0,
                sizeof(unsigned int),
                &overflows
            );

            std::cout
                << "Overflows: "
                << overflows
                << std::endl;
        }
        
    }




    auto t1 = chrono::high_resolution_clock::now();

    simulationMs = chrono::duration<float, milli>(t1 - t0).count();



    // Upload to GPU
    
    

    glUseProgram(shaderProgram);
    glBindVertexArray(vao);

    
    glDrawArrays(GL_POINTS, 0, n_particles);

    
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
        if (event.type == SDL_MOUSEBUTTONDOWN)
        {
            if (event.button.button == SDL_BUTTON_LEFT)
                mouseDown = true;
        }

        if (event.type == SDL_MOUSEBUTTONUP)
        {
            if (event.button.button == SDL_BUTTON_LEFT)
                mouseDown = false;
        }
    }



    int mx, my;
    SDL_GetMouseState(&mx, &my);
    // convet to OpenGL coords
    mousePos.x = (mx / float(windowWidth)) * 2.0f - 1.0f;
    mousePos.y = 1.0f - (my / float(windowHeight)) * 2.0f;

    return quit;
}




void rebuildGrid()
{
    gridWidth = uint(ceil((worldMaxX - worldMinX) / smoothRadius)) + 1;
    gridHeight = uint(ceil((worldMaxY - worldMinY) / smoothRadius)) + 1;
    numCells = gridWidth * gridHeight;
    MAX_PARTICLES_PER_CELL = MAX_PARTICLES_PER_CELL = 100 * ceil(float(n_particles) / numCells);
    smoothRadiusSq = smoothRadius * smoothRadius;
    kernelVolume = 3.141592 * pow(smoothRadius, 4) / 2.0f;

    // Reallocate cellCountBuffer
    std::vector<unsigned int> emptyCounts(numCells, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellCountBuffer);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        numCells * sizeof(unsigned int),
        emptyCounts.data(),
        GL_DYNAMIC_DRAW
    );

    // Reallocate cellParticleBuffer
    std::vector<unsigned int> emptyParticles(
        numCells * MAX_PARTICLES_PER_CELL, 0
    );
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellParticleBuffer);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        emptyParticles.size() * sizeof(unsigned int),
        emptyParticles.data(),
        GL_DYNAMIC_DRAW
    );
}

void gui()
{
    // ----------------- Set variables --------------------------
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
        ImGui::GetIO().Framerate);
    // ----------------------------------------------------------

    ImGui::Text("Simulation: %.3f ms", simulationMs);
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    if (ImGui::CollapsingHeader("Physics"))
    {
        // gravity
        ImGui::SliderFloat2("Gravity", &gravity.x, -1.0f, 1.0f);

        // density / pressure
        ImGui::SliderFloat("Target Density", &targetDensity, 0.1f, 20.0f);
        ImGui::SliderFloat("Pressure Multiplier", &pressureMultiplier, 0.0f, 5.0f);

        // kernel radius
        float prevRadius = smoothRadius;
        ImGui::SliderFloat("Smooth Radius", &smoothRadius, 0.0f, 0.1f);
        if (smoothRadius != prevRadius)
            rebuildGrid();

        // damping
        ImGui::SliderFloat("Dampening", &dampening, 0.0f, 1.0f);

        // damping
        ImGui::SliderFloat("Viscosity", &viscosityStrength, 0.0f, 1.0f);

        // prediction factor for positions
        ImGui::SliderFloat("Prediction Factor", &predictionFactor, 0.0f, 10.0f);
    }
    if (ImGui::CollapsingHeader("Mouse Force"))
    {
        ImGui::SliderFloat("Strength", &mouseForceStrength, -5.0f, 5.0f);

        ImGui::SliderFloat("Radius", &mouseForceRadius, 0.0f, 1.0f);
    }

    // Debug
    
    ImGui::Checkbox("Pause", &paused);

    if (paused && ImGui::Button("Step"))
    {
        runPredictShader();
        runClearGridShader();

        runBuildGridShader();
        runDensityShader();
        runForcesShader();
    }

    

    labhelper::perf::drawEventsWindow();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    initialize();

    auto startTime = std::chrono::system_clock::now();
    bool quit = false;

    while (!quit)
    {
        //update currentTime
        std::chrono::duration<float> timeSinceStart = std::chrono::system_clock::now() - startTime;
        previousTime = currentTime;
        currentTime = timeSinceStart.count();
        deltaTime = currentTime - previousTime;
        deltaTime = std::min(deltaTime, 0.01f); // just for not breaking the simulation if it gets slow

        // check events (keyboard among other)
        quit = handleEvents();

        // Inform imgui of new frame
        labhelper::newFrame(g_window);

        // render to window
        display();

        // Render overlay GUI.
        gui();

        // Finish the frame and render the GUI
        labhelper::finishFrame();

        // Swap front and back buffer. This frame will now been displayed.
        SDL_GL_SwapWindow(g_window);
    }

    labhelper::shutDown(g_window);
    return 0;
}