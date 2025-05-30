#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <cassert>
#include <gtest/gtest.h>
#include <math.h>

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

    for(int x = 0; x < 100; ++x)
    {
        for(int y = 0; y < 100; ++y)
        {
            for(int z = 0; z < 100; ++z)
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
            EXPECT_FALSE(tempLatticeArray[(10 * y) + x].isInternal);
        }
    }

    for(int y = 1; y < 9; ++y)
    {
        for(int x = 1; x < 9; ++x)
        {
            EXPECT_TRUE(tempLatticeArray[500 + (10 * y) + x].isInternal);
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

    std::array<std::pair<double, int>, 3> expected{std::pair<double, int>{0.4134132615337746, 6}, std::pair<double, int>{0.2932933692331128, 18}, std::pair<double, int>{0.2932933692331128, 17}};

    std::array<std::pair<double, int>, 27> result = testLattice.distributeVector(vector);

    double norm = result[0].first + result[1].first + result[2].first;

    result[0].first /= norm;
    result[1].first /= norm;
    result[2].first /= norm;

    EXPECT_NEAR(result[0].first, expected[0].first, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_NEAR(result[1].first, expected[1].first, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_NEAR(result[2].first, expected[2].first, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_EQ(result[0].second, expected[0].second);
    EXPECT_EQ(result[1].second, expected[1].second);
    EXPECT_EQ(result[2].second, expected[2].second);
}

TEST(CoordinatesTest, TestReflection)
{
    dim3 threads(3, 3, 3);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(3, 3, 3, blocks, threads, fluid, 0.5);

    LatticePoint* latticeArray = new LatticePoint[3 * 3 * 3];

    testLattice.load_data(latticeArray);

    testLattice.insertModel("../dataFiles/cube_angled.obj");

    LatticeData latticeData_1(testLattice.getCudaDataPointer(), testLattice.getDimensions());

    testLattice.simulateCollision();

    testLattice.preProcessModel();

    LatticePoint* tempLatticeArray = testLattice.retrieve_data();

    EXPECT_NEAR(tempLatticeArray[13].reflection_weight[1*3 + 0], 0.367402, 1e-6);
    EXPECT_NEAR(tempLatticeArray[13].reflection_weight[1*3 + 1], 0.341121, 1e-6);
    EXPECT_NEAR(tempLatticeArray[13].reflection_weight[1*3 + 2], 0.291477, 1e-6);
    EXPECT_EQ(tempLatticeArray[13].reflection_directions[1*3 + 0], 17);
    EXPECT_EQ(tempLatticeArray[13].reflection_directions[1*3 + 1], 4);
    EXPECT_EQ(tempLatticeArray[13].reflection_directions[1*3 + 2], 26);
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}