#include <cuda_runtime.h>
#include <cuda.h>

#include "device_launch_parameters.h"

namespace RunCudaTestFunctions
{
    void run_test_coordinates(dim3 threads, dim3 blocks, cudaPitchedPtr latticePtr, int y_resolution, bool* result);
}