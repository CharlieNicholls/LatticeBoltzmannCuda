#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <cstring>
#include <CGAL/bounding_box.h>
#include <queue>
#include <math.h>
#include <vector>

#include "Lattice.h"
#include "LatticePoint.h"
#include "Model.h"

#include "device_launch_parameters.h"
#include "SimCalcFuncs.cuh"

Lattice::Lattice(int x, int y, int z, dim3 blocks, dim3 threads, FluidData fluid, double spacing)
{
    m_xResolution = x;
    m_yResolution = y;
    m_zResolution = z;

    m_blocks = blocks;
    m_threads = threads;

    m_fluid = fluid;

    m_latticeSpacing = spacing;

    createExtent();
    allocateLatticeArray();

    m_dataPackage = LatticeData(latticePtr, dim3(x, y, z));
}

Lattice::~Lattice()
{
    cudaFree(latticePtr.ptr);
}

void Lattice::createExtent()
{
    latticeExtent = make_cudaExtent(sizeof(LatticePoint) * m_xResolution, m_yResolution, m_zResolution);
}

void Lattice::allocateLatticeArray()
{
    cudaMalloc3D(&latticePtr, latticeExtent);
}

void Lattice::load_data(LatticePoint* lattice_array)
{
    cudaMemcpy3DParms params = {0};
    params.srcPtr = make_cudaPitchedPtr(lattice_array, sizeof(LatticePoint) * m_xResolution, m_xResolution, m_yResolution);;
    params.srcPtr.xsize = sizeof(LatticePoint) * m_xResolution;
    params.srcPtr.ysize = m_yResolution;
    params.srcPtr.pitch = sizeof(LatticePoint) * m_xResolution;
    params.dstPtr = latticePtr;
    params.extent = latticeExtent;
    params.kind = cudaMemcpyHostToDevice;

    cudaError_t err = cudaMemcpy3D(&params);
    if (err != cudaSuccess) {
        printf("Lattice::load_data failed: %s\n", cudaGetErrorString(err));
    }
}

LatticePoint* Lattice::retrieve_data()
{
    LatticePoint* lattice_array = new LatticePoint[m_xResolution * m_yResolution * m_zResolution];

    cudaMemcpy3DParms params = {0};
    params.srcPtr = latticePtr;
    params.dstPtr.xsize = sizeof(LatticePoint) * m_xResolution;
    params.dstPtr.ysize = m_yResolution;
    params.dstPtr.pitch = sizeof(LatticePoint) * m_xResolution;
    params.dstPtr = make_cudaPitchedPtr(lattice_array, sizeof(LatticePoint) * m_xResolution, m_xResolution, m_yResolution);
    params.extent = latticeExtent;
    params.kind = cudaMemcpyDeviceToHost;

    cudaError_t err = cudaMemcpy3D(&params);
    if (err != cudaSuccess) {
        printf("Lattice::retrieve_data failed: %s\n", cudaGetErrorString(err));
        delete[] lattice_array;
        return nullptr;
    }

    size_t pitch = latticePtr.pitch;
    size_t slicePitch = pitch * latticePtr.ysize;

    LatticePoint* lattice_points = lattice_array;

    int z = 0, y = 1, x = 0;
    size_t index = z * (m_xResolution * m_yResolution) + y * m_xResolution + x;

    return lattice_array;
}

void Lattice::simulateStreaming()
{
    cudaPitchedPtr temporaryLatticePtr;

    cudaError_t err = cudaMalloc3D(&temporaryLatticePtr, latticeExtent);
    if (err != cudaSuccess) {
        printf("Lattice::simulateStreaming malloc: %s\n", cudaGetErrorString(err));
    }

    LatticeData temporary_data(temporaryLatticePtr, getDimensions());

    RunCudaFunctions::run_calculate_streaming(m_blocks, m_threads, m_dataPackage, temporary_data);

    err = cudaFree(latticePtr.ptr);
    if (err != cudaSuccess) {
        printf("Lattice::simulateStreaming free: %s\n", cudaGetErrorString(err));
    }

    latticePtr = temporaryLatticePtr;

    m_dataPackage.latticePtr = temporaryLatticePtr;
}

void Lattice::simulateCollision()
{
    RunCudaFunctions::run_calculate_collision(m_blocks, m_threads, m_dataPackage, m_fluid.m_characteristicTimescale);
}

void Lattice::simulateReflections()
{
    cudaPitchedPtr temporaryLatticePtr;

    cudaMalloc3D(&temporaryLatticePtr, latticeExtent);

    LatticeData temporary_data(temporaryLatticePtr, getDimensions());

    cudaMemcpy3DParms params = {0};
    params.srcPtr = m_dataPackage.latticePtr;
    params.dstPtr = temporaryLatticePtr;
    params.extent = latticeExtent;
    params.kind = cudaMemcpyDeviceToDevice;

    cudaError_t err = cudaMemcpy3D(&params);
    if (err != cudaSuccess) {
        printf("Lattice::simulateReflections memcpy failed: %s\n", cudaGetErrorString(err));
        return;
    }

    RunCudaFunctions::run_calculate_reflections(m_blocks, m_threads, m_dataPackage, temporary_data);

    err = cudaFree(latticePtr.ptr);
    if (err != cudaSuccess) {
        printf("Lattice::simulateReflections free failed: %s\n", cudaGetErrorString(err));
        return;
    }

    latticePtr = temporaryLatticePtr;

    m_dataPackage.latticePtr = temporaryLatticePtr;
}

void Lattice::simulateFlow()
{
    if(m_flowData != nullptr)
    {
        RunCudaFunctions::run_generate_flow(m_blocks, m_threads, m_dataPackage, d_flowData);
    }
}

void Lattice::simulateLattice()
{
    simulateStreaming();
    simulateCollision();
}

void Lattice::insertModel(std::string filename)
{
    m_simModel = new Model();

    m_simModel->importModel(filename);

    CGAL::Simple_cartesian<double>::Iso_cuboid_3 modelBound = m_simModel->bounding_box();

    if( modelBound.xmin() < 0.0 || 
        modelBound.ymin() < 0.0 || 
        modelBound.zmin() < 0.0 ||
        modelBound.xmax() > m_xResolution * m_latticeSpacing ||
        modelBound.ymax() > m_yResolution * m_latticeSpacing ||
        modelBound.zmax() > m_zResolution * m_latticeSpacing)
    {
        return;
    }

    LatticePoint* data_array = retrieve_data();

    for(int x = 0; x < m_xResolution; ++x)
    {
        for(int y = 0; y < m_yResolution; ++y)
        {
            for(int z = 0; z < m_zResolution; ++z)
            {
                data_array[z + (y * m_zResolution) + (x * m_zResolution * m_yResolution)].isInternal = m_simModel->isPointInsideModel(CGAL::Simple_cartesian<double>::Point_3(x * m_latticeSpacing, y * m_latticeSpacing, z * m_latticeSpacing));
            }
        }
    }

    load_data(data_array);
}

std::array<std::pair<double, int>, 27> Lattice::distributeVector(Point_3 vector)
{
    constexpr double directions[27][3] = {  {0.0, 0.0, 0.0},
                                            {-1.0, 0.0, 0.0}, 
                                            {0.0, -1.0, 0.0}, 
                                            {0.0, 0.0, -1.0}, 
                                            {0.0, 0.0, 1.0}, 
                                            {0.0, 1.0, 0.0}, 
                                            {1.0, 0.0, 0.0}, 
                                            {-1.0/1.4142135623730951, -1.0/1.4142135623730951, 0.0}, 
                                            {-1.0/1.4142135623730951, 0.0, -1.0/1.4142135623730951}, 
                                            {-1.0/1.4142135623730951, 0.0, 1.0/1.4142135623730951}, 
                                            {-1.0/1.4142135623730951, 1.0/1.4142135623730951, 0.0}, 
                                            {0.0, -1.0/1.4142135623730951, -1.0/1.4142135623730951}, 
                                            {0.0, -1.0/1.4142135623730951, 1.0/1.4142135623730951}, 
                                            {0.0, 1.0/1.4142135623730951, -1.0/1.4142135623730951}, 
                                            {0.0, 1.0/1.4142135623730951, 1.0/1.4142135623730951}, 
                                            {1.0/1.4142135623730951, -1.0/1.4142135623730951, 0.0}, 
                                            {1.0/1.4142135623730951, 0.0, -1.0/1.4142135623730951}, 
                                            {1.0/1.4142135623730951, 0.0, 1.0/1.4142135623730951}, 
                                            {1.0/1.4142135623730951, 1.0/1.4142135623730951, 0.0}, 
                                            {-1.0/1.7320508075688772, -1.0/1.7320508075688772, -1.0/1.7320508075688772}, 
                                            {-1.0/1.7320508075688772, -1.0/1.7320508075688772, 1.0/1.7320508075688772}, 
                                            {-1.0/1.7320508075688772, 1.0/1.7320508075688772, -1.0/1.7320508075688772}, 
                                            {-1.0/1.7320508075688772, 1.0/1.7320508075688772, 1.0/1.7320508075688772}, 
                                            {1.0/1.7320508075688772, -1.0/1.7320508075688772, -1.0/1.7320508075688772}, 
                                            {1.0/1.7320508075688772, -1.0/1.7320508075688772, 1.0/1.7320508075688772}, 
                                            {1.0/1.7320508075688772, 1.0/1.7320508075688772, -1.0/1.7320508075688772}, 
                                            {1.0/1.7320508075688772, 1.0/1.7320508075688772, 1.0/1.7320508075688772}};

    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>> dot_products;
    
    auto dist_func = [&](int index)
    {
        return 2.0 - sqrt(((vector.x - directions[index][0]) * (vector.x - directions[index][0])) +
                          ((vector.y - directions[index][1]) * (vector.y - directions[index][1])) +
                          ((vector.z - directions[index][2]) * (vector.z - directions[index][2])));
    };

    auto top_pop = [&]()
    {
        std::pair<double, int> result = dot_products.top();
        dot_products.pop();
        return result;
    };

    for(int i = 0; i < 27; ++i)
    {
        dot_products.push({dist_func(i), i});
    }

    std::array<std::pair<double, int>, 27> result;

    for(int i = 0; i < 27; ++i)
    {
        result[i] = top_pop();
    }

    return result;
}

void Lattice::preProcessModel()
{
    if(m_simModel == nullptr)
    {
        return;
    }

    constexpr int directions[27][3] = {{0, 0, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {-1, -1, 0}, {-1, 0, -1}, {-1, 0, 1}, {-1, 1, 0}, {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1}, {1, -1, 0}, {1, 0, -1}, {1, 0, 1}, {1, 1, 0}, {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1}, {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}};
    constexpr int inverse_directions[27] = {0, 6, 5, 4, 3, 2, 1, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 26, 25, 24, 23, 22, 21, 20, 19};

    LatticePoint* data_array = retrieve_data();

    for(int x = 0; x < m_xResolution; ++x)
    {
        for(int y = 0; y < m_yResolution; ++y)
        {
            for(int z = 0; z < m_zResolution; ++z)
            {
                LatticePoint* curr_element = &data_array[z + (y * m_zResolution) + (x * m_zResolution * m_yResolution)];

                if(curr_element->isInternal)
                {
                    for(int dir_index = 0; dir_index < 27; ++dir_index)
                    {
                        if( ((x + directions[dir_index][0]) >= m_xResolution) ||
                            ((y + directions[dir_index][1]) >= m_yResolution) ||
                            ((z + directions[dir_index][2]) >= m_zResolution) ||
                            ((x + directions[dir_index][0]) < 0) ||
                            ((y + directions[dir_index][1]) < 0) ||
                            ((z + directions[dir_index][2]) < 0))
                        {
                            continue;
                        }

                        if(!getElementAtDirection(x, y, z, dir_index, data_array)->isInternal)
                        {
                            Point_3 reflection_vector = m_simModel->reflectionVector(CoordsSystem::Point_3(m_latticeSpacing * (x + directions[dir_index][0]), m_latticeSpacing * (y + directions[dir_index][1]), m_latticeSpacing * (z + directions[dir_index][2])),
                                                                                     CoordsSystem::Point_3(m_latticeSpacing * x, m_latticeSpacing * y, m_latticeSpacing * z));

                            std::array<std::pair<double, int>, 27> resulting_dist = distributeVector(reflection_vector);
                            
                            int counter = 0;

                            for(int i = 0; i < 27; ++i)
                            {
                                if(!getElementAtDirection(x, y, z, resulting_dist[i].second, data_array)->isInternal)
                                {
                                    curr_element->reflection_directions[inverse_directions[dir_index] * 3 + counter] = resulting_dist[i].second;
                                    curr_element->reflection_weight[inverse_directions[dir_index] * 3 + counter] = resulting_dist[i].first;

                                    curr_element->isReflected = true;

                                    ++counter;
                                }

                                if(counter == 3)
                                {
                                    break;
                                }
                            }
                        }
                        double norm = 0.0;

                        for(int i = 0; i < 3; ++i)
                        {
                            if(curr_element->reflection_directions[inverse_directions[dir_index] * 3 + i] != 0)
                            {
                                norm += curr_element->reflection_weight[inverse_directions[dir_index] * 3 + i];
                            }
                        }

                        if(norm != 0.0)
                        {
                            for(int i = 0; i < 3; ++i)
                            {
                                if(curr_element->reflection_directions[inverse_directions[dir_index] * 3 + i] != 0)
                                {
                                    curr_element->reflection_weight[inverse_directions[dir_index] * 3 + i] /= norm;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    load_data(data_array);
}

void Lattice::setFlowData(FlowData* flowData)
{
    cudaFree(d_flowData);

    m_flowData = flowData;

    cudaMalloc(&d_flowData, sizeof(FlowData));
    cudaMemcpy(d_flowData, m_flowData, sizeof(FlowData), cudaMemcpyHostToDevice);
}

inline LatticePoint* Lattice::getElementAtDirection(int& x, int& y, int& z, int& dir_index, LatticePoint* data_array)
{
    constexpr int directions[27][3] = {{0, 0, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {-1, -1, 0}, {-1, 0, -1}, {-1, 0, 1}, {-1, 1, 0}, {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1}, {1, -1, 0}, {1, 0, -1}, {1, 0, 1}, {1, 1, 0}, {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1}, {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}};

    return &data_array[(z + directions[dir_index][2]) + ((y + directions[dir_index][1]) * m_zResolution) + ((x + directions[dir_index][0]) * m_zResolution * m_yResolution)];
}