#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#include "Lattice.h"
#include "LatticePoint.h"
#include "FlowSurface.h"
#include "device_launch_parameters.h"
#include "SimCalcFuncs.cuh"
#include "FlowCriterion.cuh"

__device__ const constexpr int DIRECTIONS[27][3] = {{0, 0, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {-1, -1, 0}, {-1, 0, -1}, {-1, 0, 1}, {-1, 1, 0}, {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1}, {1, -1, 0}, {1, 0, -1}, {1, 0, 1}, {1, 1, 0}, {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1}, {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}};
__device__ const constexpr int INVERSE_DIRECTIONS[27] = {0, 6, 5, 4, 3, 2, 1, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 26, 25, 24, 23, 22, 21, 20, 19};
__device__ const constexpr float SPLIT[27] = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0};

namespace CudaFunctions
{
    __device__ int getReflectionDirection(LatticePoint* point, const int index)
    {
        constexpr unsigned int clearLookup[6] = {0x1F, 0x3E0, 0x7C00, 0xF8000, 0x1F00000, 0x3E000000};

        int compressedIndex = index/6;
        int compressedLoc = index % 6;

        int result = point->d_reflections->reflection_directions[compressedIndex] & clearLookup[compressedLoc];

        result = result >> compressedLoc * 5;
        
        return result;
    }

    __device__ LatticePoint* get_lattice_at_coords(LatticeData lattice, int x, int y, int z)
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

    __device__ LatticePoint* get_lattice_point(LatticeData lattice, bool prime_point = false)
    {
        int z = blockDim.z * blockIdx.z + threadIdx.z;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int x = blockDim.x * blockIdx.x + threadIdx.x;

        LatticePoint* result = get_lattice_at_coords(lattice, x, y, z);

        if(prime_point)
        {
            result->x = x;
            result->y = y;
            result->z = z;
        }

        return result;
    }

    __global__ void prime_points(LatticeData lattice)
    {
        LatticePoint* current_point = get_lattice_point(lattice);

        current_point->x = blockDim.x * blockIdx.x + threadIdx.x;
        current_point->y = blockDim.y * blockIdx.y + threadIdx.y;
        current_point->z = blockDim.z * blockIdx.z + threadIdx.z;
    }

    __global__ void calculate_streaming(LatticeData lattice, LatticeData templattice)
    {
        LatticePoint* current_point = get_lattice_point(templattice, true);

        for(int i = 0; i < 27; ++i)
        {
            LatticePoint* neighbour = get_lattice_at_coords(lattice, current_point->x + DIRECTIONS[i][0], current_point->y + DIRECTIONS[i][1], current_point->z + DIRECTIONS[i][2]);

            if(neighbour != nullptr)
            {
                current_point->particle_distribution[INVERSE_DIRECTIONS[i]] = neighbour->particle_distribution[INVERSE_DIRECTIONS[i]];
            }
            else
            {
                current_point->particle_distribution[INVERSE_DIRECTIONS[i]] = 0.0;
            }
        }
    }

    __global__ void calculate_collision(LatticeData lattice, float timescale)
    {
        LatticePoint* current_point = get_lattice_point(lattice);

        float density = 0.0;
        float ux = 0.0;
        float uy = 0.0;
        float uz = 0.0;

        for(int i = 0; i < 27; ++i)
        {
            density += current_point->particle_distribution[i];
            ux += current_point->particle_distribution[i] * DIRECTIONS[i][0] * SPLIT[i];
            uy += current_point->particle_distribution[i] * DIRECTIONS[i][1] * SPLIT[i];
            uz += current_point->particle_distribution[i] * DIRECTIONS[i][2] * SPLIT[i];
        }
        
        if(density != 0.0)
        {
            ux /= density;
            uy /= density;
            uz /= density;
        }

        {   
            float common_term = ((DIRECTIONS[0][0] * ux) + (DIRECTIONS[0][1] * uy) + (DIRECTIONS[0][2] * uz));

            float eq = (8.0/27.0) * density * (1 + (3.0 * common_term) + (4.5 * common_term * common_term) - (1.5 * ((ux * ux) + (uy * uy) + (uz * uz))));

            atomicAdd(&(current_point->particle_distribution[0]), (eq - current_point->particle_distribution[0])/timescale);
        }

        for(int i = 1; i < 7; ++i)
        {
            float common_term = ((DIRECTIONS[i][0] * ux) + (DIRECTIONS[i][1] * uy) + (DIRECTIONS[i][2] * uz));

            float eq = (2.0/27.0) * density * (1 + (3.0 * common_term) + (4.5 * common_term * common_term) - (1.5 * ((ux * ux) + (uy * uy) + (uz * uz))));

            atomicAdd(&(current_point->particle_distribution[i]), (eq - current_point->particle_distribution[i])/timescale);
        }

        for(int i = 7; i < 19; ++i)
        {
            float common_term = ((DIRECTIONS[i][0] * ux) + (DIRECTIONS[i][1] * uy) + (DIRECTIONS[i][2] * uz));

            float eq = (1.0/54.0) * density * (1 + (3.0 * common_term) + (4.5 * common_term * common_term) - (1.5 * ((ux * ux) + (uy * uy) + (uz * uz))));

            atomicAdd(&(current_point->particle_distribution[i]), (eq - current_point->particle_distribution[i])/timescale);
        }

        for(int i = 19; i < 27; ++i)
        {
            float common_term = ((DIRECTIONS[i][0] * ux) + (DIRECTIONS[i][1] * uy) + (DIRECTIONS[i][2] * uz));

            float eq = (1.0/216.0) * density * (1 + (3.0 * common_term) + (4.5 * common_term * common_term) - (1.5 * ((ux * ux) + (uy * uy) + (uz * uz))));

            atomicAdd(&(current_point->particle_distribution[i]), (eq - current_point->particle_distribution[i])/timescale);
        }
    }

    __global__ void calculate_reflections(LatticeData lattice, LatticeData templattice, ReflectionData* reflection = nullptr)
    {
        LatticePoint* current_point = get_lattice_point(lattice);

        if(current_point->isReflected)
        {
            for(int i = 1; i < 81; ++i)
            {
                int reflection_direction = getReflectionDirection(current_point, i);

                if(reflection_direction == 0)
                {
                    continue;
                }

                LatticePoint* neighbour = get_lattice_at_coords(templattice, current_point->x + DIRECTIONS[reflection_direction][0], current_point->y + DIRECTIONS[reflection_direction][1], current_point->z + DIRECTIONS[reflection_direction][2]);

                if(neighbour != nullptr)
                {
                    atomicAdd(&(neighbour->particle_distribution[reflection_direction]), current_point->particle_distribution[i/3] * current_point->d_reflections->reflection_weight[i]);

                    if(reflection != nullptr)
                    {
                        atomicAdd(&(reflection->x), (current_point->particle_distribution[i/3] * current_point->d_reflections->reflection_weight[i]) * ((DIRECTIONS[reflection_direction][0] * SPLIT[reflection_direction]) - (DIRECTIONS[i][0] * SPLIT[i])));
                        atomicAdd(&(reflection->y), (current_point->particle_distribution[i/3] * current_point->d_reflections->reflection_weight[i]) * ((DIRECTIONS[reflection_direction][1] * SPLIT[reflection_direction]) - (DIRECTIONS[i][1] * SPLIT[i])));
                        atomicAdd(&(reflection->z), (current_point->particle_distribution[i/3] * current_point->d_reflections->reflection_weight[i]) * ((DIRECTIONS[reflection_direction][2] * SPLIT[reflection_direction]) - (DIRECTIONS[i][2] * SPLIT[i])));
                    }
                }
            }
        }

        if(current_point->isInternal)
        {
            LatticePoint* current_point_temp = get_lattice_point(templattice);

            for(int i = 0; i < 27; ++i)
            {
                current_point_temp->particle_distribution[i] = 0.0;
            }
        }
    }

    __global__ void generate_flow(LatticeData lattice, FlowData* flowData)
    {
        int z = blockDim.z * blockIdx.z + threadIdx.z;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int x = blockDim.x * blockIdx.x + threadIdx.x;

        if(flowData->pointCriterion(x, y, z))
        {
            LatticePoint* current_point = get_lattice_point(lattice);

            for(int i = 0; i < 27; ++i)
            {
                atomicAdd(&(current_point->particle_distribution[i]), flowData->inducedFlow.particle_distribution[i]);
            }
        }
    }
}

namespace RunCudaFunctions
{
    void run_calculate_streaming(dim3 blocks, dim3 threads, LatticeData lattice, LatticeData templattice)
    {
        CudaFunctions::calculate_streaming<<<blocks, threads>>>(lattice, templattice);
    }

    void run_calculate_collision(dim3 blocks, dim3 threads, LatticeData lattice, float timescale)
    {
        CudaFunctions::calculate_collision<<<blocks, threads>>>(lattice, timescale);
    }

    void run_calculate_reflections(dim3 blocks, dim3 threads, LatticeData lattice, LatticeData templattice)
    {
        CudaFunctions::calculate_reflections<<<blocks, threads>>>(lattice, templattice);
    }

    void run_calculate_reflections_data(dim3 blocks, dim3 threads, LatticeData lattice, LatticeData templattice, ReflectionData* reflection)
    {
        CudaFunctions::calculate_reflections<<<blocks, threads>>>(lattice, templattice, reflection);
    }

    void run_prime_points(dim3 blocks, dim3 threads, LatticeData lattice)
    {
        CudaFunctions::prime_points<<<blocks, threads>>>(lattice);
    }

    void run_generate_flow(dim3 blocks, dim3 threads, LatticeData lattice, FlowData* flowData)
    {
        CudaFunctions::generate_flow<<<blocks, threads>>>(lattice, flowData);
    }
}