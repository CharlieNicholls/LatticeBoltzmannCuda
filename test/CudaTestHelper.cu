#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>

#include "Lattice.h"
#include "LatticePoint.h"
#include "device_launch_parameters.h"
#include "SimCalcFuncs.cuh"

__device__ bool res = false;

__global__ void test_coordinates(cudaPitchedPtr latticePtr, int y_resolution, bool* result)
{
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    size_t pitch = latticePtr.pitch;
    size_t slicePitch = pitch * y_resolution;

    char* curr_slice = (char*)latticePtr.ptr + z * slicePitch;

    LatticePoint* lattice_points = (LatticePoint*)(curr_slice + y * pitch);

    LatticePoint point = lattice_points[x];

    if(point.x != x || point.y != y || point.z != z)
    {
        *result = false;
    }
}

namespace RunCudaTestFunctions
{
    void run_test_coordinates(dim3 threads, dim3 blocks, cudaPitchedPtr latticePtr, int y_resolution, bool* result)
    {
        test_coordinates<<<threads, blocks>>>(latticePtr, y_resolution, result);
    }

    void run_test_streaming(dim3 threads, dim3 blocks, LatticeData lattice, LatticeData templattice)
    {
        CudaFunctions::calculate_streaming<<<threads, blocks>>>(lattice, templattice);
    }

    void run_prime_points(dim3 threads, dim3 blocks, LatticeData lattice)
    {
        CudaFunctions::prime_points<<<threads, blocks>>>(lattice);
    }
}