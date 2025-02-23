#include <cassert>
#include <gtest/gtest.h>
#include <array>
#include <math.h>
#include <cuda_runtime.h>
#include <CGAL/Simple_cartesian.h>

#include "Geometry.h"
#include "Constants.h"

TEST(GeometryTest, TestQuaternionRoatation)
{
    Point_3 axis(cos(M_PI * 0.25), sin(M_PI * 0.25), 0.0);

    Point_3 vector(1.0, 0.0, 0.0);

    Point_3 expected(0.0, 1.0, 0.0);

    Point_3 result = rotateVectorAroundAxis(axis, vector, 180.0 * (M_PI/180.0));

    EXPECT_NEAR(result.x, expected.x, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_NEAR(result.y, expected.y, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_NEAR(result.z, expected.z, CONSTANTS::GEOMETRIC_TOLERANCE);

    Point_3 expected_2(0.5, 0.5, -cos(M_PI * 0.25));

    Point_3 result_2 = rotateVectorAroundAxis(axis, vector, 90.0 * (M_PI/180.0));

    EXPECT_NEAR(result_2.x, expected_2.x, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_NEAR(result_2.y, expected_2.y, CONSTANTS::GEOMETRIC_TOLERANCE);
    EXPECT_NEAR(result_2.z, expected_2.z, CONSTANTS::GEOMETRIC_TOLERANCE);
}