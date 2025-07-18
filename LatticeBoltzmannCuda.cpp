#include <stdio.h>
#include <iostream>

#include "Lattice.h"
#include "LatticePoint.h"
#include "device_launch_parameters.h"
#include "FlowSurface.h"
#include "FlowCriterion.cuh"
#include "SimCalcFuncs.cuh"
#include "Renderer.cuh"

int main(int argc, char *argv[])
{
    if(argc == 0)
    {
        return -1;
    }

    dim3 threads(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    dim3 blocks(atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));

    FluidData fluid(atof(argv[8]), atof(argv[9]));

    LatticePoint* data = new LatticePoint[threads.x * blocks.x * threads.y * blocks.y * threads.z * blocks.z];

    Lattice userLattice(threads.x * blocks.x, threads.y * blocks.y, threads.z * blocks.z, blocks, threads, fluid, atof(argv[10]));

    std::string filename(argv[1]);

    LatticePoint flowRef;

    {   
        flowRef.particle_distribution[0] = 0.001;
    }

    for(int i = 1; i < 7; ++i)
    {
        flowRef.particle_distribution[i] = 0.001;
    }

    for(int i = 7; i < 19; ++i)
    {
        flowRef.particle_distribution[i] = 0.001;
    }

    for(int i = 19; i < 27; ++i)
    {
        flowRef.particle_distribution[i] = 0.001;
    }
    flowRef.particle_distribution[6] = 0.002;

    data = userLattice.retrieve_data();

    for(int x = 0; x < threads.x * blocks.x * threads.y * blocks.y * threads.z * blocks.z; ++x)
    {
        for(int i = 0; i < 27; ++i)
        {
            data[x].particle_distribution[i] = flowRef.particle_distribution[i];
        }
    }

    userLattice.load_data(data);

    userLattice.insertModel(filename);
    userLattice.preProcessModel();

    FlowData* inducedFlow = new FlowData(&FlowCriterion::ptestPlaneCriteria, flowRef);
    userLattice.setFlowData(inducedFlow);

    ReflectionData* reflection = new ReflectionData;
    reflection->x = 0.0;
    reflection->y = 0.0;
    reflection->z = 0.0;
    userLattice.setReflectionData(reflection);

    LatticeData latticeData_1(userLattice.getCudaDataPointer(), userLattice.getDimensions());
    RunCudaFunctions::run_prime_points(blocks, threads, latticeData_1);

    auto print_data = [&]()
    {
        reflection = userLattice.retrieve_reflection_data();

        std::cout << reflection->x << " " << reflection->y << " " << reflection->z << std::endl;
    };

    Renderer render_controller(userLattice.getDimensions().x * 5, userLattice.getDimensions().y * 5, userLattice.getDimensions().z/2);

    render_controller.runRenderer();

    for(int step = 0; step < atoi(argv[11]); ++step)
    {
        print_data();
        userLattice.updateRenderData(render_controller.getDataPtr());
        render_controller.display();

        userLattice.simulateStreaming();
        userLattice.simulateReflections();
        userLattice.simulateCollision();
        userLattice.simulateFlow();
    }

    return 0;
}