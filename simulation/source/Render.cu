#include "Render.cuh"
#include "LatticePoint.h"
#include "Lattice.h"

__device__ const constexpr int DIRECTIONS[27][3] = {{0, 0, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {-1, -1, 0}, {-1, 0, -1}, {-1, 0, 1}, {-1, 1, 0}, {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1}, {1, -1, 0}, {1, 0, -1}, {1, 0, 1}, {1, 1, 0}, {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1}, {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}};
__device__ const constexpr int INVERSE_DIRECTIONS[27] = {0, 6, 5, 4, 3, 2, 1, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 26, 25, 24, 23, 22, 21, 20, 19};
__device__ const constexpr float SPLIT[27] = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0};

namespace CudaFunctions
{
    __device__ LatticePoint* get_lattice_at_coords_renderer(LatticeData lattice, int x, int y, int z)
    {
        cudaPitchedPtr latticePtr = lattice.latticePtr;
        dim3 resolution = lattice.latticeDimensions;

        if(x < 0 || y < 0 || z < 0 || x >= resolution.x || y >= resolution.y || z >= resolution.z)
        {
            return nullptr;
        }

        size_t pitch = latticePtr.pitch;
        size_t slicePitch = pitch * latticePtr.ysize;

        char* curr_slice = (char*)latticePtr.ptr + x * slicePitch;

        LatticePoint* lattice_points = (LatticePoint*)(curr_slice + y * pitch);

        return &lattice_points[z];
    }

    __global__ void render_output_data(LatticeData lattice, float* data, int layer)
    {
        int z = layer;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int x = blockDim.x * blockIdx.x + threadIdx.x;

        float output = 0.0;

        LatticePoint* curr_point = get_lattice_at_coords_renderer(lattice, x, y, layer);

        for(int i = 0; i < 27; ++i)
        {
            output += curr_point->particle_distribution[i];
        }

        data[y * lattice.latticeDimensions.y + x] = output;
    }
}

namespace RunCudaFunctions
{
    void run_render_output_data(dim3 blocks, dim3 threads, LatticeData lattice, float* data, int layer)
    {
        blocks.z = 1;
        threads.z = 1;
        CudaFunctions::render_output_data<<<blocks, threads>>>(lattice, data, layer);
    }
}