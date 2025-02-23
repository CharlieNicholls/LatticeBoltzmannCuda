#include "Geometry.h"
#include "Constants.h"

#include <math.h>
#include <array>

double calculateVectorMagnitude(Point_3& vector)
{
    return sqrt((vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z));
}

std::array<double, 4> quaternionMult(std::array<double, 4> vector_1, std::array<double, 4> vector_2)
{
    double r_0 = (vector_1[0] * vector_2[0]) - (vector_1[1] * vector_2[1]) - (vector_1[2] * vector_2[2]) - (vector_1[3] * vector_2[3]);
    double r_1 = (vector_1[0] * vector_2[1]) + (vector_1[1] * vector_2[0]) + (vector_1[2] * vector_2[3]) - (vector_1[3] * vector_2[2]);
    double r_2 = (vector_1[0] * vector_2[2]) - (vector_1[1] * vector_2[3]) + (vector_1[2] * vector_2[0]) + (vector_1[3] * vector_2[1]);
    double r_3 = (vector_1[0] * vector_2[3]) + (vector_1[1] * vector_2[2]) - (vector_1[2] * vector_2[1]) + (vector_1[3] * vector_2[0]);

    return std::array<double, 4>{r_0, r_1, r_2, r_3};
}

Point_3 rotateVectorAroundAxis(Point_3 axis, Point_3 vector, double angle)
{
    {
        double magnitude = calculateVectorMagnitude(axis);

        axis.x /= magnitude;
        axis.y /= magnitude;
        axis.z /= magnitude;
    }

    angle /= 2.0;

    std::array<double, 4> q{cos(angle), sin(angle) * axis.x, sin(angle) * axis.y, sin(angle) * axis.z};
    std::array<double, 4> q_inverse{cos(-angle), sin(-angle) * axis.x, sin(-angle) * axis.y, sin(-angle) * axis.z};

    std::array<double, 4> point{0.0, vector.x, vector.y, vector.z};

    std::array<double, 4> result = quaternionMult(quaternionMult(q, point), q_inverse);

    return Point_3(result[1], result[2], result[3]);
}