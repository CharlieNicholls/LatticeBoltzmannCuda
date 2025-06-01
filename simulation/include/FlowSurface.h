#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include "LatticePoint.h"
#include "FlowCriterion.cuh"

struct FlowData
{
    LatticePoint inducedFlow;
    bool (*pointCriterion)(int x, int y, int z);
    int val = 1;

    __host__ __device__ FlowData(bool (**criterion)(int, int, int), LatticePoint flow) 
    {
        inducedFlow = flow;

        bool (*criterionFunction)(int, int, int);
        cudaError_t err = cudaMemcpyFromSymbol(&criterionFunction, criterion, sizeof(bool (*)(int, int, int)));

        if (err != cudaSuccess) {
            printf("FlowData Constructor cudaMemcpyFromSymbol failed: %s\n", cudaGetErrorString(err));
            return;
        }

        pointCriterion = criterionFunction;
    }

    void setCriterion(bool (**criterion)(int, int, int))
    {
        bool (*criterionFunction)(int, int, int);
        cudaError_t err = cudaMemcpyFromSymbol(&criterionFunction, &criterion, sizeof(bool (*)(int, int, int)));
    
        if (err != cudaSuccess) {
            printf("FlowData setCriterion cudaMemcpyFromSymbol failed: %s\n", cudaGetErrorString(err));
            return;
        }

        pointCriterion = criterionFunction;
    }

    void setInducedFlow(LatticePoint flow)
    {
        inducedFlow = flow;
    }
};