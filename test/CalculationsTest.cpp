#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <cassert>
#include <gtest/gtest.h>

#include "Lattice.h"
#include "LatticePoint.h"
#include "SimCalcFuncs.cuh"
#include "CudaTestHelper.cuh"
#include "device_launch_parameters.h"

TEST(CalculationsTest, TestStreaming)
{
    dim3 threads(1, 2, 1);
    dim3 blocks(1, 1, 1);

    FluidData fluid(1.0, 1.0);

    Lattice testLattice(1, 2, 1, blocks, threads, fluid);

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

    Lattice testLattice(1, 1, 1, blocks, threads, fluid);

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

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}