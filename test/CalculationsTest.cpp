#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <cassert>
#include <gtest/gtest.h>

#include "Lattice.h"
#include "LatticePoint.h"
#include "SimCalcFuncs.cuh"
#include "CudaTestHelper.cuh"
#include "device_launch_parameters.h"
#include "Model.h"
#include "Constants.h"
#include "FlowSurface.h"
#include "FlowCriterion.cuh"

TEST(CalculationsTest, TestStreaming)
{
    dim3 threads(1, 2, 1);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(1, 2, 1, blocks, threads, fluid, 0.1);

    LatticeData latticeData_1(testLattice.getCudaDataPointer(), testLattice.getDimensions());

    LatticePoint* latticeArray = new LatticePoint[1 * 2 * 1];

    memset(latticeArray, 0, sizeof(LatticePoint) * 2);

    latticeArray[0].particle_distribution[5] = 2.0;

    testLattice.load_data(latticeArray);

    RunCudaTestFunctions::run_prime_points(blocks, threads, latticeData_1);

    testLattice.simulateStreaming();

    latticeArray = testLattice.retrieve_data();

    EXPECT_NEAR(latticeArray[1].particle_distribution[5], 2.0, 1e-9);
}

TEST(CalculationsTest, TestCollision)
{
    dim3 threads(1, 1, 1);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(1, 1, 1, blocks, threads, fluid, 0.1);

    LatticeData latticeData(testLattice.getCudaDataPointer(), testLattice.getDimensions());

    LatticePoint* latticeArray = new LatticePoint[1 * 1 * 1];

    memset(latticeArray, 0, sizeof(LatticePoint) * 1);

    latticeArray[0].particle_distribution[0] = 2.0;

    testLattice.load_data(latticeArray);

    RunCudaTestFunctions::run_prime_points(blocks, threads, latticeData);

    testLattice.simulateCollision();

    latticeArray = testLattice.retrieve_data();

    EXPECT_NEAR(latticeArray[0].particle_distribution[0], 0.592592593, 1e-9);
    EXPECT_NEAR(latticeArray[0].particle_distribution[1], 0.148148148, 1e-9);
    EXPECT_NEAR(latticeArray[0].particle_distribution[2], 0.148148148, 1e-9);
}

TEST(CalculationsTest, TestFull)
{
    dim3 threads(1, 2, 1);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(1, 2, 1, blocks, threads, fluid, 0.1);

    LatticeData latticeData_1(testLattice.getCudaDataPointer(), testLattice.getDimensions());

    LatticePoint* latticeArray = new LatticePoint[1 * 2 * 1];

    memset(latticeArray, 0, sizeof(LatticePoint) * 2);

    latticeArray[0].particle_distribution[5] = 2.0;

    testLattice.load_data(latticeArray);

    RunCudaTestFunctions::run_prime_points(blocks, threads, latticeData_1);

    testLattice.simulateLattice();

    latticeArray = testLattice.retrieve_data();
    
    EXPECT_NEAR(latticeArray[1].particle_distribution[0], -0.296296296, 1e-9);
    EXPECT_NEAR(latticeArray[1].particle_distribution[1], -0.074074074, 1e-9);
    EXPECT_NEAR(latticeArray[1].particle_distribution[2], 0.148148148, 1e-9);
}

TEST(CalculationsTest, TestReflections)
{
    dim3 threads(3, 3, 3);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(3, 3, 3, blocks, threads, fluid, 0.5);

    LatticeData latticeData_1(testLattice.getCudaDataPointer(), testLattice.getDimensions());

    LatticePoint* latticeArray = new LatticePoint[3 * 3 * 3];

    testLattice.insertModel("../dataFiles/cube_angled.obj");
    testLattice.preProcessModel();

    auto point_at_coords = [&](int x, int y, int z)
    {
        dim3 dims = testLattice.getDimensions();
        return z + (y * dims.z) + (x * dims.z * dims.y);
    };

    latticeArray = testLattice.retrieve_data();

    latticeArray[point_at_coords(1, 1, 1)].particle_distribution[1] = 1.0;

    testLattice.load_data(latticeArray);

    RunCudaTestFunctions::run_prime_points(blocks, threads, latticeData_1);

    testLattice.simulateReflections();

    latticeArray = testLattice.retrieve_data();

    EXPECT_NEAR(latticeArray[point_at_coords(2, 1, 2)].particle_distribution[17], 0.367402, 1e-6);
    EXPECT_NEAR(latticeArray[point_at_coords(1, 1, 2)].particle_distribution[4], 0.341121, 1e-6);
    EXPECT_NEAR(latticeArray[point_at_coords(2, 2, 2)].particle_distribution[26], 0.291477, 1e-6);

    EXPECT_NEAR(latticeArray[point_at_coords(1, 1, 1)].particle_distribution[1], 0.0, 1e-6);
}

TEST(CalculationsTest, TestReflectionsData)
{
    dim3 threads(3, 3, 3);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(3, 3, 3, blocks, threads, fluid, 0.5);

    LatticeData latticeData_1(testLattice.getCudaDataPointer(), testLattice.getDimensions());

    LatticePoint* latticeArray = new LatticePoint[3 * 3 * 3];

    testLattice.insertModel("../dataFiles/cube_angled.obj");
    testLattice.preProcessModel();

    auto point_at_coords = [&](int x, int y, int z)
    {
        dim3 dims = testLattice.getDimensions();
        return z + (y * dims.z) + (x * dims.z * dims.y);
    };

    latticeArray = testLattice.retrieve_data();

    latticeArray[point_at_coords(1, 1, 1)].particle_distribution[1] = 1.0;

    testLattice.load_data(latticeArray);

    ReflectionData* reflection = new ReflectionData;

    testLattice.setReflectionData(reflection);

    RunCudaTestFunctions::run_prime_points(blocks, threads, latticeData_1);

    testLattice.simulateReflections();

    latticeArray = testLattice.retrieve_data();

    EXPECT_NEAR(latticeArray[point_at_coords(2, 1, 2)].particle_distribution[17], 0.367402, 1e-6);
    EXPECT_NEAR(latticeArray[point_at_coords(1, 1, 2)].particle_distribution[4], 0.341121, 1e-6);
    EXPECT_NEAR(latticeArray[point_at_coords(2, 2, 2)].particle_distribution[26], 0.291477, 1e-6);

    EXPECT_NEAR(latticeArray[point_at_coords(1, 1, 1)].particle_distribution[1], 0.0, 1e-6);

    reflection = testLattice.retrieve_reflection_data();

    EXPECT_NEAR(reflection->x, 0.28086, 1e-6);
    EXPECT_NEAR(reflection->y, 0.097159, 1e-6);
    EXPECT_NEAR(reflection->z, 0.621981, 1e-6);
}

TEST(CalculationsTest, TestFlowGeneration)
{
    dim3 threads(3, 3, 3);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(3, 3, 3, blocks, threads, fluid, 0.5);

    LatticeData latticeData_1(testLattice.getCudaDataPointer(), testLattice.getDimensions());

    LatticePoint* latticeArray = new LatticePoint[3 * 3 * 3];

    auto point_at_coords = [&](int x, int y, int z)
    {
        dim3 dims = testLattice.getDimensions();
        return z + (y * dims.z) + (x * dims.z * dims.y);
    };

    testLattice.load_data(latticeArray);

    RunCudaTestFunctions::run_prime_points(blocks, threads, latticeData_1);

    LatticePoint flowRef;
    flowRef.particle_distribution[0] = 1.0;

    FlowData* inducedFlow = new FlowData(&FlowCriterion::ptestCriteria, flowRef);

    testLattice.setFlowData(inducedFlow);
    testLattice.simulateFlow();

    latticeArray = testLattice.retrieve_data();

    EXPECT_NEAR(latticeArray[point_at_coords(1, 1, 1)].particle_distribution[0], 1.0, 1e-6);
    EXPECT_NEAR(latticeArray[point_at_coords(0, 0, 0)].particle_distribution[0], 0.0, 1e-6);
}

TEST(CalculationsTest, TestPlaneFlowGeneration)
{
    dim3 threads(3, 3, 3);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(3, 3, 3, blocks, threads, fluid, 0.5);

    LatticeData latticeData_1(testLattice.getCudaDataPointer(), testLattice.getDimensions());

    LatticePoint* latticeArray = new LatticePoint[3 * 3 * 3];

    auto point_at_coords = [&](int x, int y, int z)
    {
        dim3 dims = testLattice.getDimensions();
        return z + (y * dims.z) + (x * dims.z * dims.y);
    };

    testLattice.load_data(latticeArray);

    RunCudaTestFunctions::run_prime_points(blocks, threads, latticeData_1);

    LatticePoint flowRef;
    flowRef.particle_distribution[0] = 1.0;

    FlowData* inducedFlow = new FlowData(&FlowCriterion::ptestPlaneCriteria, flowRef);

    testLattice.setFlowData(inducedFlow);
    testLattice.simulateFlow();

    latticeArray = testLattice.retrieve_data();

    for(int y = 0; y < 3; ++y)
    {
        for(int z = 0; z < 3; ++z)
        {
            EXPECT_NEAR(latticeArray[point_at_coords(0, y, z)].particle_distribution[0], 1.0, 1e-6);
            EXPECT_NEAR(latticeArray[point_at_coords(1, y, z)].particle_distribution[0], 0.0, 1e-6);
        }
    }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}