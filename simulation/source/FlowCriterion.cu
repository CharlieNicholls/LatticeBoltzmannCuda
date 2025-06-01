#include "FlowCriterion.cuh"

namespace FlowCriterion
{
    __device__ bool testCriteria(int x, int y, int z)
    {
        return x == 1 && y == 1 && z == 1;
    }

    __device__ bool testPlaneCriteria(int x, int y, int z)
    {
        return x == 0;
    }

    __device__ bool (*ptestCriteria)(int, int, int) = &FlowCriterion::testCriteria;
    __device__ bool (*ptestPlaneCriteria)(int, int, int) = &FlowCriterion::testPlaneCriteria;
}