cmake_minimum_required(VERSION 3.18)

project(LatticeBoltzmannCuda VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(CGAL REQUIRED COMPONENTS Core)

if(CGAL_FOUND)
    include_directories(${CGAL_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "CGAL not found")
endif()

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)

find_package(glfw3 REQUIRED)

enable_testing()

add_subdirectory(simulation)
add_subdirectory(render)
add_subdirectory(test)

add_executable(LatticeBoltzmannCuda LatticeBoltzmannCuda.cpp)

target_include_directories(LatticeBoltzmannCuda PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CGAL_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
)

target_link_libraries(LatticeBoltzmannCuda PRIVATE
    GTest::gtest_main
    SimulationLib
    RenderLib
    CGAL::CGAL
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
    GLU
    dl
    pthread
)