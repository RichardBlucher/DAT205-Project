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
float smoothRadius = 0.15f;

float targetDensity = 6.5f;
float pressureMultiplier = 0.1f;

int n_particles = 1000;

// Particles


std::vector<vec2> positions(n_particles);
std::vector<vec2> velocities(n_particles);
std::vector<float> densities(n_particles);
std::vector<float> pressures(n_particles);

std::vector<vec2> predictedPositions(n_particles);
float predictionFactor = 1.0f;
// Window
SDL_Window* g_window = nullptr;
int windowWidth = 800;
int windowHeight = 600;

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
static bool paused = false;

// For plots
const int graphSize = 200;

std::vector<float> avgDensityHistory(graphSize, 0.0f);
std::vector<float> avgVelocityHistory(graphSize, 0.0f);

int graphOffset = 0;

// Mouse interactions
vec2 mousePos(0.0f);

bool mouseDown = false;

float mouseForceRadius = 0.3f;
float mouseForceStrength = 2.0f;

///////////////////////////////////////////////////////////////////////////////
// Compute shader stuff
#include <fstream>
#include <sstream>

GLuint computeProgram = 0;

GLuint particleBuffer;
GLuint velSSBO;

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

void runComputeShader()
{
    glUseProgram(computeProgram);

    GLint dtLoc =
        glGetUniformLocation(computeProgram, "dt");

    glUniform1f(dtLoc, deltaTime);

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

///////////////////////////////////////////////////////////////////////////////
struct Entry
{
    int particleIndex;
    unsigned int cellKey;

    // for sorting by cellKey
    bool operator<(const Entry& other) const
    {
        return cellKey < other.cellKey;
    }
};

// Global containers
const unsigned int TABLE_SIZE = 2 * n_particles;

std::vector<Entry> spacialLookup(n_particles);
std::vector<int> startIndices(TABLE_SIZE, INT_MAX);

std::pair<int, int> PositionToCellCoord(const vec2& point, float radius) // maps positions to cells (radius should be approx interaction radius)
{
    int cellX = static_cast<int>(floor(point.x / radius));
    int cellY = static_cast<int>(floor(point.y / radius));
    return { cellX, cellY };
}

unsigned int HashCell(int cellX, int cellY) // turns cell coords to hash (large int)
{
    unsigned int a = (unsigned int)cellX * 15823u;
    unsigned int b = (unsigned int)cellY * 9737333u;
    return a + b;
}

unsigned int GetKeyFromHash(unsigned int hash) // hash to key (some hashes might have same key but worth it anyway)
{
    return hash % TABLE_SIZE;
}

void updateSpacialLookup(const std::vector<vec2>& points, float radius)
{
    std::fill(startIndices.begin(), startIndices.end(), INT_MAX); // reset

    for (int i = 0; i < n_particles; i++)
    {
        auto [cellX, cellY] = PositionToCellCoord(points[i], radius);
        unsigned int cellKey = GetKeyFromHash(HashCell(cellX, cellY));

        spacialLookup[i] = { i, cellKey }; // assignt each particle to a cell
    }

    std::sort(spacialLookup.begin(), spacialLookup.end()); // sort by cell key

    for (int i = 0; i < n_particles; i++)
    {
        unsigned int key = spacialLookup[i].cellKey;
        unsigned int prevKey = (i == 0) ? UINT_MAX : spacialLookup[i - 1].cellKey;

        if (key != prevKey)
        {
            startIndices[key] = i; // sets start indices
        }
    }
}

std::vector<std::pair<int, int>> cellOffsets = {
    {-1,-1}, {0,-1}, {1,-1},
    {-1, 0}, {0, 0}, {1, 0},
    {-1, 1}, {0, 1}, {1, 1}
}; // needed for neighboring cells

template<typename Func>
void ForEachPointWithinRadius(const vec2& samplePoint, const std::vector<vec2>& pointSet, float radius, Func func)
{
    std::pair<int, int> centre = PositionToCellCoord(samplePoint, radius); // find centre cell
    int centreX = centre.first;
    int centreY = centre.second;

    float sqrRadius = radius * radius;

    for (const auto& offset : cellOffsets) // check centre and neighbors
    {
        int offsetX = offset.first;
        int offsetY = offset.second;

        unsigned int key = GetKeyFromHash(
            HashCell(centreX + offsetX, centreY + offsetY)
        );

        int startIndex = startIndices[key];
        if (startIndex == INT_MAX)
            continue; // if INT_MAX, no particle in cell

        for (int i = startIndex; i < spacialLookup.size(); i++)
        {
            if (spacialLookup[i].cellKey != key)
                break; // iterate only paricles in this cell (if key changes, no longer same cell)

            int particleIndex = spacialLookup[i].particleIndex;

            vec2 diff = pointSet[particleIndex] - samplePoint;
            float sqrDst = dot(diff, diff);

            if (sqrDst <= sqrRadius) // distance check
            {
                func(particleIndex, sqrDst); // do whatever func() does (ex. calculate forces)
            }
        }
    }
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
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_PROFILE_MASK,
        SDL_GL_CONTEXT_PROFILE_CORE
    );

    g_window = labhelper::init_window_SDL("SPH", windowWidth, windowHeight);
    SDL_GL_SetSwapInterval(0); // disable vsync
    glDisable(GL_DEPTH_TEST);

    std::cout << "OpenGL version: "
        << glGetString(GL_VERSION)
        << std::endl;

    

    // Create some balls
    for (int i = 0; i < n_particles; i++)
    {

        positions[i] = vec2(
            (rand() / float(RAND_MAX)) * 2.0f - 1.0f,
            (rand() / float(RAND_MAX)) * 2.0f - 1.0f
        );

        /*b.vel = vec2(
            (rand() / float(RAND_MAX)) * 0.5f - 0.25f,
            (rand() / float(RAND_MAX)) * 0.5f - 0.25f
        );*/
        velocities[i] = vec2(0.0f, 0.0f);


    }
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    ////// Compute shader init
    computeProgram =
        createComputeShader("integrate.comp");

    // Positions
    glGenBuffers(1, &particleBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, particleBuffer);

    glBufferData(
        GL_ARRAY_BUFFER,
        positions.size() * sizeof(vec2),
        positions.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        particleBuffer
    ); // comute shader writes directly to render buffer

    // Velocities
    glGenBuffers(1, &velSSBO);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, velSSBO);

    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        velocities.size() * sizeof(vec2),
        velocities.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        velSSBO
    );

    initShaderStuff();
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
    float volume = M_PI * radius * radius * radius * radius / 2.0f; // for normalisation
    return -3.0f * (radius - dist) * (radius - dist) / volume;

}

float calculateDensity(const vec2& samplePoint, const std::vector<vec2>& pointSet)
{
    float density = 0.0f;

    ForEachPointWithinRadius(samplePoint, pointSet, smoothRadius,
        [&](int j, float sqrDst)
        {
            float dist = sqrt(sqrDst);
            float influence = kernel(smoothRadius, dist);
            density += mass * influence;
        });

    return density;
}
float densityToPressure(float density)
{
    float densityError = density - targetDensity;
    float pressure = densityError * pressureMultiplier;//std::max(0.0f, densityError) * pressureMultiplier; // maybe not clamp at 0?
    return pressure;
}

void updateDensities()
{

    for (int i = 0; i < n_particles; i++)
    {
        float density = calculateDensity(positions[i], predictedPositions);

        densities[i] = density;
        pressures[i] = densityToPressure(density);
    }
}


float sharedPressure(int iA, int iB)
{
    float pressureA = pressures[iA];
    float pressureB = pressures[iB];
    return (pressureA + pressureB) / 2.0f;
}

vec2 calculatePressureForce(int i)
{
    vec2 force(0.0f);

    const vec2& pos_i = predictedPositions[i];

    ForEachPointWithinRadius(pos_i, predictedPositions, smoothRadius,
        [&](int j, float sqrDst)
        {
            if (j == i) return; // skip self

            // Avoid sqrt unless needed
            float dist = sqrt(sqrDst);
            if (dist < 1e-6f) return;

            vec2 dir = (predictedPositions[j] - pos_i) / dist;

            float slope = kernelDerivative(smoothRadius, dist);
            float density_j = densities[j];

            force += sharedPressure(i, j) * dir * slope * mass / density_j;
        });

    return force;
}

vec2 calculateViscosityForce(int i)
{
    vec2 force(0.0f);
    const vec2& pos_i = predictedPositions[i];

    ForEachPointWithinRadius(pos_i, predictedPositions, smoothRadius,
        [&](int j, float sqrDst)
        {
            if (j == i) return;

            float dist = sqrt(sqrDst);
            if (dist < 1e-6f) return;

            vec2 velDiff = velocities[j] - velocities[i];

            float influence = kernel(smoothRadius, dist);

            force += velDiff * influence;
        });


    return viscosityStrength * force;
}

vec2 calculateMouseForce(int i)
{
    if (!mouseDown)
        return vec2(0.0f);

    vec2 offset = mousePos - predictedPositions[i];

    float dist = length(offset);

    if (dist > mouseForceRadius || dist < 1e-5f)
        return vec2(0.0f);

    vec2 dir = offset / dist;

    // Smooth falloff
    float strength =
        1.0f - dist / mouseForceRadius;

    return dir * strength * mouseForceStrength;
}

void updateStatistics()
{
    float avgDensity = 0.0f;
    float avgVelocity = 0.0f;

    for (int i = 0; i < n_particles; i++)
    {
        avgDensity += densities[i];
        avgVelocity += length(velocities[i]);
    }

    avgDensity /= n_particles;
    avgVelocity /= n_particles;

    avgDensityHistory[graphOffset] = avgDensity;
    avgVelocityHistory[graphOffset] = avgVelocity;

    graphOffset = (graphOffset + 1) % graphSize;
}

void updateBalls()
{
    for (int i = 0; i < n_particles; i++)
    {
        predictedPositions[i] =
            positions[i] + velocities[i] * deltaTime * predictionFactor;
    }
    updateSpacialLookup(predictedPositions, smoothRadius);
    updateDensities();
    for (int i = 0; i < n_particles; i++)
    {

        velocities[i] += gravity * deltaTime;
        velocities[i] += calculatePressureForce(i) / std::max(densities[i], 0.0001f) * deltaTime;
        velocities[i] += calculateViscosityForce(i) * deltaTime;
        velocities[i] += calculateMouseForce(i) * deltaTime;

        positions[i] += velocities[i] * deltaTime;

        // Bounce on edges (-1 to 1 in OpenGL)
        if (positions[i].x < -1.0f) {
            positions[i].x = -1.0f;
            velocities[i].x *= -dampening;
        }
        if (positions[i].x > 1.0f) {
            positions[i].x = 1.0f;
            velocities[i].x *= -dampening;
        }


        if (positions[i].y < -1.0f) {
            positions[i].y = -1.0f;
            velocities[i].y *= -dampening;
        }
        if (positions[i].y > 1.0f) {
            positions[i].y = 1.0f;
            velocities[i].y *= -dampening;
        }
    }
}
///////////////////////////////////////////////////////////////////////////////

void display()
{
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);



    auto t0 = chrono::high_resolution_clock::now();
    if (!paused)
    {
        //updateBalls();
        runComputeShader();
        
        updateStatistics();
    }




    auto t1 = chrono::high_resolution_clock::now();

    simulationMs = chrono::duration<float, milli>(t1 - t0).count();



    // Upload to GPU
    
    

    glUseProgram(shaderProgram);
    glBindVertexArray(vao);

    glPointSize(10.0f); // ball size
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
        ImGui::SliderFloat("Smooth Radius", &smoothRadius, 0.05f, 1.0f);

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
    if (ImGui::Button("Reset Simulation"))
    {
        for (int i = 0; i < n_particles; i++)
        {
            positions[i] = vec2(
                (rand() / float(RAND_MAX)) * 2.0f - 1.0f,
                (rand() / float(RAND_MAX)) * 2.0f - 1.0f
            );
            velocities[i] = vec2(0.0f);
        }
    }

    ImGui::Checkbox("Pause", &paused);

    if (paused && ImGui::Button("Step"))
    {
        updateBalls();
    }

    // plotting 
    ImGui::Separator();
    ImGui::Text("Simulation Statistics");

    float currentDensity =
        avgDensityHistory[(graphOffset - 1 + graphSize) % graphSize];

    float currentVelocity =
        avgVelocityHistory[(graphOffset - 1 + graphSize) % graphSize];

    ImGui::Text("Avg Density: %.3f", currentDensity);
    ImGui::Text("Avg Velocity: %.3f", currentVelocity);

    ImGui::PlotLines(
        "Average Density",
        avgDensityHistory.data(),
        graphSize,
        graphOffset,
        nullptr,
        0.0f,
        targetDensity * 5.0f,
        ImVec2(0, 80)
    );

    ImGui::PlotLines(
        "Average Velocity",
        avgVelocityHistory.data(),
        graphSize,
        graphOffset,
        nullptr,
        0.0f,
        5.0f,
        ImVec2(0, 80)
    );

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