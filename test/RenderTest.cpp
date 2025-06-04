#include <gtest/gtest.h>

#include "renderer.cuh"

TEST(RenderTest, RenderTest)
{
    renderer render_obj;

    render_obj.runRenderer();

    render_obj.runCuda();
}