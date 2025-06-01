# CMakeLists.txt for the include directory
file(GLOB_RECURSE HEADER_FILES *.h *.hpp *.cuh)

# Export header files as a variable to the parent scope
set(HEADER_FILES ${HEADER_FILES} PARENT_SCOPE)