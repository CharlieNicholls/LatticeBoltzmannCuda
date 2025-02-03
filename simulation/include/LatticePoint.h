#pragma once

struct LatticePoint
{
    int x, y, z;
    double equilibrium[27];
    double particle_distribution[27];
    bool isReflected;
    double reflection_weight[27];
};