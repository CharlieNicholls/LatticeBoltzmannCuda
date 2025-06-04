#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <iostream>

static int m_width = 800;
static int m_height = 600;
static GLuint pbo = 0;
extern GLFWwindow* window;
static cudaGraphicsResource* cuda_pbo_resource;
static uchar4* m_devPtr;

class renderer
{
public:
    void runRenderer();
    uchar4* getRenderPtr();
    void runCuda();

private:
    void initPBO();
    void display();
};