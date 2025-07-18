#include <cuda_runtime.h>
#include <cuda.h>

#include "device_launch_parameters.h"

class LatticeData;

namespace CudaFunctions
{
    __global__ void render_output_data(LatticeData lattice, float* data);
}

namespace RunCudaFunctions
{
    void run_render_output_data(dim3 blocks, dim3 threads, LatticeData lattice, float* data, int layer);
}