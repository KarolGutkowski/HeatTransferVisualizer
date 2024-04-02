#include <iostream>
#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include "glfw_utils/glfw_utils.h"
#include "gpu_calculation/gpu_calculation.h"
#include "cuda_gl_interop.h"

#define WINDOW_HEIGHT 600
#define WINDOW_WIDTH 600

// this should be multiples of 4, since opengl wants allignment 
// so f.e. 10 wont work but 8, 12, 32, 100 will
#define SIMULATION_SIZE 52


void run_window_loop(GLFWwindow* window);
GLuint generatePBO();
void addDataToPBO(uchar3* data, GLuint pbo, size_t size);
uchar3* allocateCPUGrid();
void loadGlad();
GLuint generateTexture();
float* generate_starting_temperature_data(int w, int h);
void draw_texture(GLuint tex, int w, int h);

int main(void)
{
    GLFWwindow* window = initialize_GLFWWindow(WINDOW_WIDTH, WINDOW_HEIGHT);
    loadGlad();
    run_window_loop(window);
    glfwTerminate();
    return 0;
}



void run_window_loop(GLFWwindow* window)
{
    auto temperature_data = generate_starting_temperature_data(SIMULATION_SIZE, SIMULATION_SIZE);

    float alpha = 2000;
    float delta_x, delta_y;

    delta_x = delta_y = 0.05f/ SIMULATION_SIZE;

    float delta_t = (delta_x * delta_x)/ (4*alpha);
    auto pbo = generatePBO();
    auto cpu_grid = allocateCPUGrid();

    auto tex = generateTexture();

    addDataToPBO(cpu_grid, pbo, 3* sizeof(char)* SIMULATION_SIZE * SIMULATION_SIZE);

    cudaGraphicsResource* cuda_pbo;
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

    while (!glfwWindowShouldClose(window))
    {
        clear_buffers();
        uchar3* pixels;

        cudaGraphicsMapResources(1, &cuda_pbo, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&pixels, NULL, cuda_pbo);

        calculate_heat_equation_at_time(delta_t, delta_x, delta_y, alpha, 
            pixels, temperature_data, SIMULATION_SIZE, SIMULATION_SIZE);

        cudaGraphicsUnmapResources(1, &cuda_pbo, 0);

        draw_texture(tex, SIMULATION_SIZE, SIMULATION_SIZE);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    free(temperature_data);
}


GLuint generatePBO()
{
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    return pbo;
}

void addDataToPBO(uchar3* data, GLuint pbo, size_t size)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, data, GL_STREAM_DRAW);
}

void loadGlad()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initalize GLAD" << std::endl;
        exit(EXIT_SUCCESS);
    }
}

void draw_texture(GLuint tex, int w , int h)
{
    glBindTexture(GL_TEXTURE_2D, tex);

    glClear(GL_COLOR_BUFFER_BIT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0, 1.0);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(-1.0, 1.0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

uchar3* allocateCPUGrid()
{
    uchar3* cpu_grid = (uchar3*)malloc(sizeof(uchar3) * SIMULATION_SIZE * SIMULATION_SIZE);

    if (!cpu_grid)
    {
        std::cout << "failed to allocate cpu grid" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < SIMULATION_SIZE * SIMULATION_SIZE; i++)
    {
        cpu_grid[i].x = 0;
        cpu_grid[i].y = 0;
        cpu_grid[i].z = 0;
    }

    return cpu_grid;
}


GLuint generateTexture()
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return tex;
}

float* generate_starting_temperature_data(int w, int h)
{
    float* temps = (float*)malloc(sizeof(float) * w * h);
    if (temps == NULL)
    {
        printf("Failed to allocate initial temperatures\n");
        exit(EXIT_FAILURE);
    }

    float hotTemp = 2900.0f;
    float coldtemp = 100.0f;

    for (int i = 0; i < w * h; i++)
    {
        if (i < w)
            temps[i] = hotTemp;
        else if (i % w == w - 1)
            temps[i] = hotTemp;
        else 
            temps[i] = 0.0f;
    }

    return temps;
}