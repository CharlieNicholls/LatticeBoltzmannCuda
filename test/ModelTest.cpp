#include <gtest/gtest.h>

#include "Model.h"

TEST(ModelTest, importModel)
{
    Model modelData;

    modelData.importModel("../dataFiles/triangle.obj");

    EXPECT_EQ(modelData.getMesh().number_of_vertices(), 3);
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}