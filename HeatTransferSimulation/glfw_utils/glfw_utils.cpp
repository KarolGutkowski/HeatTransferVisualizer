#include "glfw_utils.h"
#include <iostream>


void clear_buffers()
{
    glClear(GL_COLOR_BUFFER_BIT);
}

GLFWwindow* initialize_GLFWWindow(int width, int height)
{
    /* Initialize the library */
    if (!glfwInit())
        exit(EXIT_FAILURE);


    /* Create a windowed mode window and its OpenGL context */
    GLFWwindow* window = glfwCreateWindow(width, height, "Heat Transfer Simulation", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);


    return window;
}

