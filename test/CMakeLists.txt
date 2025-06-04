include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB TEST_SOURCES *.cpp)
file(GLOB CUDA_SOURCES *.cu)

file(GLOB OBJ_FILES dataFiles/*.obj)

file(COPY dataFiles DESTINATION ${CMAKE_BINARY_DIR})

include(GoogleTest)

foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE} ${CUDA_SOURCES})
    target_link_libraries(${TEST_NAME} PRIVATE GTest::gtest_main SimulationLib RenderLib)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()