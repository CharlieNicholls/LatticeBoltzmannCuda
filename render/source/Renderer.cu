#include "Renderer.cuh"

#include <float.h>

GLFWwindow* window = nullptr;

#define SCALE 5

__global__ void displayArray(uchar4* ptr, int width, int height, float2* max_values, float* data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        int share = (y/SCALE) * (width/SCALE) + (x/SCALE);

        int pixelValue = (int)(255.0 * (data[share] - max_values->x)/(max_values->y - max_values->x));

        if(pixelValue > 255)
        {
            printf("%f, %f\n", max_values->y, data[share]);
        }

        ptr[idx] = make_uchar4(pixelValue, pixelValue, pixelValue, 255);
    }
}

__global__ void convert_float_to_float2(float* data, float2* outData, int dataLen)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < dataLen)
    {
        outData[x].x = data[x];
        outData[x].y = data[x];
    }
}

__global__ void find_min_max_post(float2* data, float2* outData, int dataLen)
{
    extern __shared__ float2 block_data_post[];

    float2 result;

    result.x = FLT_MAX;
    result.y = FLT_TRUE_MIN;

    int tx = threadIdx.x;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < dataLen)
    {
        block_data_post[tx] = data[x];
    }
    __syncthreads();

    if(tx == 0)
    {
        for(int i = 0; i < blockDim.x; ++i)
        {
            if(block_data_post[i].x < result.x)
            {
                result.x = block_data_post[i].x;
            }

            if(block_data_post[i].y > result.y)
            {
                result.y = block_data_post[i].y;
            }
        }

        outData[blockIdx.x] = result;
    }
    __syncthreads();
}

void Renderer::runRenderer()
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

    init();

    return;
}

void Renderer::runCuda() {
    uchar4* devPtr;
    size_t size;

    // Map PBO to CUDA
    cudaError_t err = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    if (err != cudaSuccess)
        std::cerr << "cudaGraphicsMapResources failed: " << cudaGetErrorString(err) << std::endl;

    err = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cuda_pbo_resource);
    if (err != cudaSuccess)
        std::cerr << "cudaGraphicsResourceGetMappedPointer failed: " << cudaGetErrorString(err) << std::endl;

    dim3 threads(32, 32);
    dim3 blocks((m_width + threads.x - 1) / threads.x, (m_height + threads.y - 1) / threads.y);

    if(display_data != nullptr)
    {
        int threadNum = threads.x * threads.y;
        int blockNum = blocks.x * blocks.y;

        float2* max_data;
        float2* max_value;

        int dataLength = m_width/SCALE * m_height/SCALE;

        cudaMalloc(&max_data, dataLength * sizeof(float2));

        convert_float_to_float2<<<blockNum, threadNum>>>(display_data, max_data, dataLength);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            std::cerr << "convert_float_to_float2 failed: " << cudaGetErrorString(err) << std::endl;

        while(blockNum * threadNum != 1)
        {
            cudaMalloc(&max_value, blockNum * sizeof(float2));

            find_min_max_post<<<blockNum, threadNum, threadNum * sizeof(float2)>>>(max_data, max_value, dataLength);

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
                std::cerr << "find_min_max_post failed: " << cudaGetErrorString(err) << std::endl;

            cudaFree(max_data);

            max_data = max_value;

            threadNum = blockNum;
            blockNum = 1;

            dataLength = threadNum * blockNum;
        }

        displayArray<<<blocks, threads>>>(devPtr, m_width, m_height, max_data, display_data);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            std::cerr << "displayArray failed: " << cudaGetErrorString(err) << std::endl;

        cudaFree(max_data);
        cudaFree(max_value);
    }

    err = cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    if (err != cudaSuccess)
        std::cerr << "cudaGraphicsUnmapResources failed: " << cudaGetErrorString(err) << std::endl;
}

void Renderer::init()
{
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * 4, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorString(err) << std::endl;

    cudaMalloc(&display_data, m_width/SCALE * m_height/SCALE * sizeof(float));
}

void Renderer::display()
{
    if(!glfwWindowShouldClose(window))
    {
        runCuda();

        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glDrawPixels(m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}