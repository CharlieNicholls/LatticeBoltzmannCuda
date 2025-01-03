#pragma once

struct LatticePoint
{
    int x, y, z;
    double density[27];
    double veclocity[27];
    double equilibrium[27];
    double particle_distribution[27];
    double streaming[27];
    double collision[27];
};