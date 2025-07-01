#pragma once

struct LatticePoint
{
    int x, y, z;
    float particle_distribution[27];
    bool isReflected;
    bool isInternal;
    int reflection_directions[14];
    float reflection_weight[81];
};