#include <cassert>
#include <gtest/gtest.h>
#include <array>
#include <math.h>
#include <cuda_runtime.h>
#include <CGAL/Simple_cartesian.h>

#include "Geometry.h"
#include "Constants.h"

#include <iostream>

TEST(GeometryTest, TestQuaternionRoatation)
{
    Point_3 axis(cos(M_PI * 0.25), sin(M_PI * 0.25), 0.0);

    Point_3 vector(1.0, 0.0, 0.0);

    Point_3 expected(0.0, 1.0, 0.0);

    Point_3 result = rotateVectorAroundAxis(axis, vector, 180.0 * (M_PI/180.0));

    std::cout << axis.x << " " << axis.y << " " << axis.z << std::endl;

    EXPECT_NEAR(result.x, expected.x, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_NEAR(result.y, expected.y, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_NEAR(result.z, expected.z, CONSTANTS::GEOMETRIC_TOLERANCE);
}