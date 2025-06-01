#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include "device_launch_parameters.h"

namespace FlowCriterion
{
    __device__ bool testCriteria(int x, int y, int z);
    __device__ bool testPlaneCriteria(int x, int y, int z);

    extern __device__ bool (*ptestCriteria)(int, int, int);
    extern __device__ bool (*ptestPlaneCriteria)(int, int, int);
}