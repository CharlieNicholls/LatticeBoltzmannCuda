#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#include "Lattice.h"
#include "LatticePoint.h"
#include "FlowSurface.h"
#include "device_launch_parameters.h"
#include "SimCalcFuncs.cuh"
#include "FlowCriterion.cuh"

namespace CudaFunctions
{
    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;

        unsigned long long int old = *address_as_ull, assumed;

        do{ assumed = old;
            old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +__longlong_as_double(assumed)));
        } while (assumed != old);

        return __longlong_as_double(old);
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

    __global__ void calculate_equilibrium(LatticeData lattice)
    {
        LatticePoint* current_point = get_lattice_point(lattice);

        double density = 0.0;
        double ux = 0.0;
        double uy = 0.0;
        double uz = 0.0;

        constexpr int directions[27][3] = {{0, 0, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {-1, -1, 0}, {-1, 0, -1}, {-1, 0, 1}, {-1, 1, 0}, {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1}, {1, -1, 0}, {1, 0, -1}, {1, 0, 1}, {1, 1, 0}, {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1}, {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}};
        constexpr double split[27] = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0};

        for(int i = 0; i < 27; ++i)
        {
            density += current_point->particle_distribution[i];
            ux += current_point->particle_distribution[i] * directions[i][0] * split[i];
            uy += current_point->particle_distribution[i] * directions[i][1] * split[i];
            uz += current_point->particle_distribution[i] * directions[i][2] * split[i];
        }
        
        if(density == 0.0)
        {
            for(int i = 0; i < 27; ++i)
            {
                current_point->equilibrium[i] = 0.0;
            }

            return;
        }

        ux /= density;
        uy /= density;
        uz /= density;

        double sq = (1.5 * ((ux * ux) + (uy * uy) + (uz * uz)));

        {   
            double common_term = ((directions[0][0] * ux) + (directions[0][1] * uy) + (directions[0][2] * uz));
            current_point->equilibrium[0] = (8.0/27.0) * density * (1 + (3.0 * common_term) + (4.5 * common_term * common_term) - sq);
        }

        for(int i = 1; i < 7; ++i)
        {
            double common_term = ((directions[i][0] * ux) + (directions[i][1] * uy) + (directions[i][2] * uz));

            current_point->equilibrium[i] = (2.0/27.0) * density * (1 + (3.0 * common_term) + (4.5 * common_term * common_term) - sq);
        }

        for(int i = 7; i < 19; ++i)
        {
            double common_term = ((directions[i][0] * ux) + (directions[i][1] * uy) + (directions[i][2] * uz));

            current_point->equilibrium[i] = (1.0/54.0) * density * (1 + (3.0 * common_term) + (4.5 * common_term * common_term) - sq);
        }

        for(int i = 19; i < 27; ++i)
        {
            double common_term = ((directions[i][0] * ux) + (directions[i][1] * uy) + (directions[i][2] * uz));

            current_point->equilibrium[i] = (1.0/216.0) * density * (1 + (3.0 * common_term) + (4.5 * common_term * common_term) - sq);
        }
    }

    __global__ void calculate_streaming(LatticeData lattice, LatticeData templattice)
    {
        LatticePoint* current_point = get_lattice_point(templattice, true);

        constexpr int directions[27][3] = {{0, 0, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {-1, -1, 0}, {-1, 0, -1}, {-1, 0, 1}, {-1, 1, 0}, {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1}, {1, -1, 0}, {1, 0, -1}, {1, 0, 1}, {1, 1, 0}, {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1}, {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}};
        constexpr int inverse_directions[27] = {0, 6, 5, 4, 3, 2, 1, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 26, 25, 24, 23, 22, 21, 20, 19};

        for(int i = 0; i < 27; ++i)
        {
            LatticePoint* neighbour = get_lattice_at_coords(lattice, current_point->x + directions[i][0], current_point->y + directions[i][1], current_point->z + directions[i][2]);

            if(neighbour != nullptr)
            {
                current_point->particle_distribution[inverse_directions[i]] = neighbour->particle_distribution[inverse_directions[i]];
            }
            else
            {
                current_point->particle_distribution[inverse_directions[i]] = 0.0;
            }
        }
    }

    __global__ void calculate_collision(LatticeData lattice, double timescale)
    {
        LatticePoint* current_point = get_lattice_point(lattice);

        for(int i = 0; i < 27; ++i)
        {
            double adder = (current_point->equilibrium[i] - current_point->particle_distribution[i])/timescale;

            if(adder > 0.5)
            {
                printf("Large collsion value change detected: %d, %d, %d, %d, %f, %f", current_point->x, current_point->y, current_point->z, i, current_point->particle_distribution[i], current_point->equilibrium[i]);
            }

            atomicAdd(&(current_point->particle_distribution[i]), (current_point->equilibrium[i] - current_point->particle_distribution[i])/timescale);
        }
    }

    __global__ void calculate_reflections(LatticeData lattice, LatticeData templattice, ReflectionData* reflection = nullptr)
    {
        LatticePoint* current_point = get_lattice_point(lattice);

        constexpr int directions[27][3] = {{0, 0, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {-1, -1, 0}, {-1, 0, -1}, {-1, 0, 1}, {-1, 1, 0}, {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1}, {1, -1, 0}, {1, 0, -1}, {1, 0, 1}, {1, 1, 0}, {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1}, {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}};
        constexpr double split[27] = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0};

        if(current_point->isReflected)
        {
            for(int i = 1; i < 81; ++i)
            {
                int reflection_direction = current_point->reflection_directions[i];

                if(reflection_direction == 0)
                {
                    continue;
                }

                LatticePoint* neighbour = get_lattice_at_coords(templattice, current_point->x + directions[reflection_direction][0], current_point->y + directions[reflection_direction][1], current_point->z + directions[reflection_direction][2]);

                if(neighbour != nullptr)
                {
                    atomicAdd(&(neighbour->particle_distribution[reflection_direction]), current_point->particle_distribution[i/3] * current_point->reflection_weight[i]);

                    if(reflection != nullptr)
                    {
                        atomicAdd(&(reflection->x), (current_point->particle_distribution[i/3] * current_point->reflection_weight[i]) * ((directions[reflection_direction][0] * split[reflection_direction]) - (directions[i][0] * split[i])));
                        atomicAdd(&(reflection->y), (current_point->particle_distribution[i/3] * current_point->reflection_weight[i]) * ((directions[reflection_direction][1] * split[reflection_direction]) - (directions[i][1] * split[i])));
                        atomicAdd(&(reflection->z), (current_point->particle_distribution[i/3] * current_point->reflection_weight[i]) * ((directions[reflection_direction][2] * split[reflection_direction]) - (directions[i][2] * split[i])));
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

    void run_calculate_collision(dim3 blocks, dim3 threads, LatticeData lattice, double timescale)
    {
        CudaFunctions::calculate_equilibrium<<<blocks, threads>>>(lattice);

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