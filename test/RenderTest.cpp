#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "Renderer.cuh"
#include "Lattice.h"
#include "device_launch_parameters.h"


TEST(RenderTest, RenderTest)
{
    dim3 threads(1, 1, 1);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(1, 1, 1, blocks, threads, fluid, 0.1);

    LatticePoint* latticeArray = new LatticePoint[1];

    for(int i = 0; i < 27; ++i)
    {
        latticeArray[0].particle_distribution[i] = 0.0;
    }
    latticeArray[0].particle_distribution[5] = 2.0;

    testLattice.load_data(latticeArray);

    float* d_data;

    cudaMalloc(&d_data, sizeof(float));

    testLattice.updateRenderData(d_data);

    float* h_data = new float[1];

    cudaMemcpy(h_data, d_data, sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_data[0], 2.0, 1e-6);
}