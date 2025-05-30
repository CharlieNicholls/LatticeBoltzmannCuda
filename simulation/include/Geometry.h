#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <math.h>
#include <cuda_runtime.h>

//Will need to create solution for geometry data structures
struct Point_3
{
    double x, y, z;

    Point_3(double x_i, double y_i, double z_i) : x(x_i), y(y_i), z(z_i) {};

    bool operator==(const Point_3& a) const
    {
        return a.x == x &&
               a.y == y &&
               a.z == z;
    }

    void normaliseVector()
    {
        double norm = sqrt((x * x) + (y * y) + (z * z));

        x /= norm;
        y /= norm;
        z /= norm;
    }
};

Point_3 rotateVectorAroundAxis(Point_3 axis, Point_3 vector, double angle);
#endif