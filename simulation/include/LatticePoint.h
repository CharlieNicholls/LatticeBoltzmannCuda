#pragma once

struct LatticePoint
{
    int x, y, z;
    double equilibrium[27];
    double particle_distribution[27];
    bool isReflected;
    bool isInternal;
    int reflection_directions[81];
    double reflection_weight[81];
};