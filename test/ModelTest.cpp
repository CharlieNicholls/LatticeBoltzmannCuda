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

TEST(ModelTest, intersectionNormal)
{
    Model modelData;

    modelData.importModel("../dataFiles/compat_cube.obj");

    CGAL::Simple_cartesian<double>::Point_3 point_1(0.5, 0.5, -1.0);
    CGAL::Simple_cartesian<double>::Point_3 point_2(0.5, 0.5,  1.0);

    Point_3 result = modelData.reflectionVector(point_1, point_2);
    Point_3 value(0.0, 0.0, -1.0);

    EXPECT_NEAR(result.x, value.x, 1e-9);
    EXPECT_NEAR(result.y, value.y, 1e-9);
    EXPECT_NEAR(result.z, value.z, 1e-9);

    point_2 = CGAL::Simple_cartesian<double>::Point_3(0.5, 0.5,  -2.0);

    result = modelData.reflectionVector(point_1, point_2);

    value = Point_3(0.0, 0.0, 0.0);

    EXPECT_EQ(result, value);
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}