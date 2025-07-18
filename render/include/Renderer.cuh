#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <iostream>

static GLuint pbo = 0;
extern GLFWwindow* window;
static cudaGraphicsResource* cuda_pbo_resource;
static uchar4* m_devPtr;

class Renderer
{
public:
    Renderer(int x, int y, int z) : m_width(x), m_height(y), z_layer(z) {}

    void runRenderer();
    void runCuda();

    float* getDataPtr() {return display_data;};

    void display();

private:
    void init();

    float* display_data = nullptr;

    int m_width = 100;
    int m_height = 100;
    int z_layer = 0;
};