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

TEST(ModelTest, isPointInsideModel)
{
    Model modelData;

    modelData.importModel("../dataFiles/cube.obj");

    CGAL::Simple_cartesian<double>::Point_3 point(0.0, 0.0, 0.0);

    EXPECT_TRUE(modelData.isPointInsideModel(point));

    CGAL::Simple_cartesian<double>::Point_3 point_2(2.0, 0.0, 0.0);

    EXPECT_FALSE(modelData.isPointInsideModel(point_2));
}

TEST(ModelTest, checkErrorModel)
{
    Model modelData;

    modelData.importModel("../dataFiles/quad.obj");

    EXPECT_TRUE(modelData.checkError());
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}