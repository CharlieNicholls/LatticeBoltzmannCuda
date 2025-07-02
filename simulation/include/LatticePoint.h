#pragma once

struct ReflectionValues
{
    int reflection_directions[14];
    float reflection_weight[81];
};

struct LatticePoint
{
    int x, y, z;
    float particle_distribution[27];
    bool isReflected;
    bool isInternal;
    ReflectionValues* reflections = nullptr;
    ReflectionValues* d_reflections = nullptr;
};