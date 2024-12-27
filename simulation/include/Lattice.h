#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

class LatticePoint;

class Lattice
{
public:
    Lattice(int x, int y, int z);
    ~Lattice();

    void load_data(LatticePoint* lattice_array);
    LatticePoint* retrieve_data(size_t& size);

    cudaPitchedPtr getCudaDataPointer() { return latticePtr; };

private:
    void createExtent();
    void allocateLatticeArray();

    cudaPitchedPtr latticePtr;
    cudaExtent latticeExtent;

    unsigned short m_xResolution;
    unsigned short m_yResolution;
    unsigned short m_zResolution;
};