cmake_minimum_required(VERSION 3.8.0)

project(LatticeBoltzmannCuda VERSION 0.1.0 LANGUAGES C CXX CUDA)

enable_testing()

add_subdirectory(simulation)
add_subdirectory(test)

add_executable(LatticeBoltzmannCuda LatticeBoltzmannCuda.cpp)

target_link_libraries(LatticeBoltzmannCuda PRIVATE GTest::gtest_main SimulationLib)