#include "renderer.cuh"

GLFWwindow* window = nullptr;

__global__ void fillImageKernel(uchar4* ptr, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        ptr[idx] = make_uchar4(x % 256, y % 256, 128, 255);
    }
}

void renderer::runRenderer()
{
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    window = glfwCreateWindow(m_width, m_height, "CUDA + OpenGL PBO Render", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glewInit();
    cudaSetDevice(0);

    initPBO();

    while (!glfwWindowShouldClose(window)) {
        display();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    return;
}

uchar4* renderer::getRenderPtr()
{
    uchar4* m_devPtr;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&m_devPtr, &size, cuda_pbo_resource);

    return m_devPtr;
}

void renderer::runCuda() {
    uchar4* devPtr = getRenderPtr();
    size_t size;

    // Map PBO to CUDA
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cuda_pbo_resource);

    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);
    fillImageKernel<<<grid, block>>>(devPtr, m_width, m_height);

    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void renderer::initPBO()
{
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * 4, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void renderer::display() 
{
    runCuda();

    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}