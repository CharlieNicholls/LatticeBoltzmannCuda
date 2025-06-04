# CMakeLists.txt for the simulation module
add_subdirectory(include)
add_subdirectory(source)

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)

find_package(glfw3 REQUIRED)

# Create a library that includes both headers and sources
add_library(RenderLib ${SOURCE_FILES})
target_include_directories(RenderLib PUBLIC include OpenGL GLEW GLFW)

target_link_libraries(RenderLib PUBLIC OpenGL GLEW glfw)

set_target_properties(RenderLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)