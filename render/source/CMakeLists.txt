# CMakeLists.txt for the source directory
file(GLOB_RECURSE SOURCE_FILES *.cpp *.cu)

# Export source files as a variable to the parent scope
set(SOURCE_FILES ${SOURCE_FILES} PARENT_SCOPE)