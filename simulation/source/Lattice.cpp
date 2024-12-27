#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>

#include "Lattice.h"
#include "LatticePoint.h"

#include "SimCalcFuncs.cuh"

Lattice::Lattice(int x, int y, int z)
{
    m_xResolution = x;
    m_yResolution = y;
    m_zResolution = z;

    createExtent();
    allocateLatticeArray();
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
    printf("abc");

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

LatticePoint* Lattice::retrieve_data(size_t& size)
{
    LatticePoint* lattice_array = new LatticePoint[m_xResolution * m_yResolution * m_zResolution];

    cudaMemcpy3DParms params = {0};
    params.srcPtr.ptr = latticePtr.ptr;
    params.srcPtr.xsize = sizeof(LatticePoint) * m_xResolution;
    params.srcPtr.ysize = m_yResolution;
    params.srcPtr.pitch = sizeof(LatticePoint) * m_xResolution;
    params.dstPtr.ptr = lattice_array;
    params.extent = latticeExtent;

    return lattice_array;
}