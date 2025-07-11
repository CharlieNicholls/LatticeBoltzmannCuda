#include "renderer.cuh"

GLFWwindow* window = nullptr;

__global__ void fillImageKernel(uchar4* ptr, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        ptr[idx] = make_uchar4(255, 0, 0, 255);
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

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    cudaSetDevice(0);

    glViewport(0, 0, m_width, m_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, m_width, 0, m_height, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    initPBO();

    while (!glfwWindowShouldClose(window)) {
        // Clear the screen
        //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
    uchar4* devPtr;
    size_t size;

    // Map PBO to CUDA
    cudaError_t err = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    if (err != cudaSuccess)
        std::cerr << "cudaGraphicsMapResources failed: " << cudaGetErrorString(err) << std::endl;

    err = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cuda_pbo_resource);
    if (err != cudaSuccess)
        std::cerr << "cudaGraphicsResourceGetMappedPointer failed: " << cudaGetErrorString(err) << std::endl;

    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);
    fillImageKernel<<<grid, block>>>(devPtr, m_width, m_height);

    cudaDeviceSynchronize();

    err = cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    if (err != cudaSuccess)
        std::cerr << "cudaGraphicsUnmapResources failed: " << cudaGetErrorString(err) << std::endl;
}

void renderer::initPBO()
{
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * 4, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorString(err) << std::endl;
}

void renderer::display() 
{
    runCuda();

    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}