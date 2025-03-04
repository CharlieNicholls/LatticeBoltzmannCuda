#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <cassert>
#include <gtest/gtest.h>

#include "Lattice.h"
#include "LatticePoint.h"
#include "CudaTestHelper.cuh"
#include "device_launch_parameters.h"
#include "Constants.h"

TEST(CoordinatesTest, TestLoadAndRetrieve)
{
    dim3 threads(1, 2, 1);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(1, 2, 1, blocks, threads, fluid, 0.1);

    LatticePoint* latticeArray = new LatticePoint[1 * 2 * 1];

    latticeArray[0].particle_distribution[5] = 2.0;

    testLattice.load_data(latticeArray);

    LatticePoint* tempLatticeArray = testLattice.retrieve_data();

    EXPECT_NEAR(tempLatticeArray[0].particle_distribution[5], latticeArray[0].particle_distribution[5], 1e-9);
}

TEST(CoordinatesTest, TestCoordinatesOnCopy)
{
    dim3 threads(10, 10, 10);
    dim3 blocks(10, 10, 10);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(100, 100, 100, blocks, threads, fluid, 0.1);

    LatticePoint* latticeArray = new LatticePoint[100 * 100 * 100];

    int counter = 0;

    for(int z = 0; z < 100; ++z)
    {
        for(int y = 0; y < 100; ++y)
        {
            for(int x = 0; x < 100; ++x)
            {
                LatticePoint curr_point;

                curr_point.x = x;
                curr_point.y = y;
                curr_point.z = z;

                latticeArray[counter] = curr_point;

                ++counter;
            }
        }
    }

    testLattice.load_data(latticeArray);

    bool* coords_test_result;

    bool final_result = true;

    cudaMalloc((void **)&coords_test_result, sizeof(bool));

    cudaMemcpy(coords_test_result, &final_result, sizeof(bool), cudaMemcpyHostToDevice);

    RunCudaTestFunctions::run_test_coordinates(threads, blocks, testLattice.getCudaDataPointer(), 100, coords_test_result);

    cudaMemcpy(&final_result, coords_test_result, sizeof(bool), cudaMemcpyDeviceToHost);

    EXPECT_TRUE(final_result);
}

TEST(CoordinatesTest, TestCoordinatesWithModel)
{
    dim3 threads(1, 1, 1);
    dim3 blocks(10, 10, 10);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(10, 10, 10, blocks, threads, fluid, 0.1);

    LatticePoint* latticeArray = new LatticePoint[10 * 10 * 10];

    testLattice.load_data(latticeArray);

    testLattice.insertModel("../dataFiles/compat_cube.obj");

    LatticePoint* tempLatticeArray = testLattice.retrieve_data();

    for(int y = 0; y < 10; ++y)
    {
        for(int x = 0; x < 10; ++x)
        {
            EXPECT_FALSE(tempLatticeArray[(10 * y) + x].isReflected);
        }
    }

    for(int y = 1; y < 9; ++y)
    {
        for(int x = 1; x < 9; ++x)
        {
            EXPECT_TRUE(tempLatticeArray[500 + (10 * y) + x].isReflected);
        }
    }
}

TEST(CoordinatesTest, TestDistributeVector)
{
    dim3 threads(1, 1, 1);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(1, 1, 1, blocks, threads, fluid, 0.1);

    Point_3 vector(1.0, 0.1, 0.1);

    std::array<std::pair<double, int>, 3> expected{std::pair<double, int>{0.6726727939963124, 6}, std::pair<double, int>{0.5232166435699435, 18}, std::pair<double, int>{0.5232166435699435, 17}};

    std::array<std::pair<double, int>, 3> result = testLattice.distributeVector(vector);

    EXPECT_NEAR(result[0].first, expected[0].first, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_NEAR(result[1].first, expected[1].first, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_NEAR(result[2].first, expected[2].first, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_EQ(result[0].second, expected[0].second);
    EXPECT_EQ(result[1].second, expected[1].second);
    EXPECT_EQ(result[2].second, expected[2].second);
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}