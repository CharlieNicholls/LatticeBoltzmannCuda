#include <gtest/gtest.h>

#include "Model.h"

TEST(ModelTest, importModel)
{
    Model modelData;

    modelData.importModel("../dataFiles/triangle.obj");

    EXPECT_EQ(modelData.getMesh().number_of_vertices(), 3);
}

TEST(ModelTest, isModelClosed)
{
    Model modelData;

    modelData.importModel("../dataFiles/cube.obj");

    EXPECT_TRUE(modelData.isModelClosed());

    modelData.importModel("../dataFiles/triangle.obj");

    EXPECT_FALSE(modelData.isModelClosed());
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}