# CMakeLists.txt for the simulation module
add_subdirectory(include)
add_subdirectory(source)

# Create a library that includes both headers and sources
add_library(SimulationLib ${SOURCE_FILES})
target_include_directories(SimulationLib PUBLIC include)