#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <cstring>

#include "Lattice.h"
#include "LatticePoint.h"

#include "device_launch_parameters.h"
#include "SimCalcFuncs.cuh"

Lattice::Lattice(int x, int y, int z, dim3 blocks, dim3 threads, FluidData fluid)
{
    m_xResolution = x;
    m_yResolution = y;
    m_zResolution = z;

    m_blocks = blocks;
    m_threads = threads;

    m_fluid = fluid;

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

    cudaMemcpy3D(&params);
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
        printf("cudaMemcpy3D failed: %s\n", cudaGetErrorString(err));
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

    cudaMalloc3D(&temporaryLatticePtr, latticeExtent);

    LatticeData temporary_data(temporaryLatticePtr, getDimensions());

    RunCudaFunctions::run_calculate_streaming(m_blocks, m_threads, m_dataPackage, temporary_data);

    cudaFree(&latticePtr);

    latticePtr = temporaryLatticePtr;

    m_dataPackage.latticePtr = temporaryLatticePtr;
}

void Lattice::simulateCollision()
{
    RunCudaFunctions::run_calculate_collision(m_blocks, m_threads, m_dataPackage, m_fluid.m_characteristicTimescale);
}

void Lattice::simulateLattice()
{
    simulateStreaming();
    simulateCollision();
}