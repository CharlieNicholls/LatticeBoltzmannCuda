#pragma once

struct LatticePoint
{
    int x, y, z;
    double streaming[27];
    double collision[27];
};