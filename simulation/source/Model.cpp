#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/bounding_box.h>
#include <boost/optional.hpp>

#include <filesystem>

#include "Model.h"

typedef CGAL::Simple_cartesian<double> CoordsSystem;
typedef CGAL::Surface_mesh<CoordsSystem::Point_3> Mesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<CoordsSystem, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;
typedef boost::optional<AABB_tree::Intersection_and_primitive_id<CoordsSystem::Ray_3>::Type> Ray_intersection;

void Model::importModel(std::string filename)
{
    if(!std::filesystem::exists(filename))
    {
        return;
    }

    CGAL::IO::read_polygon_mesh(filename, modelMesh);

    meshValid = CGAL::is_triangle_mesh(modelMesh);

    if(modelTree != nullptr)
    {
        delete modelTree;
    }

    lazyAabbTreeConstruction();
}

bool Model::isModelClosed()
{
    return CGAL::is_closed(modelMesh);
}

bool Model::isPointInsideModel(CGAL::Simple_cartesian<double>::Point_3 point)
{
    if(!isModelClosed())
    {
        return false;
    }

    if(!CGAL::is_triangle_mesh(modelMesh))
    {
        meshValid = false;
        return false;
    }

    CGAL::Side_of_triangle_mesh<Mesh, CoordsSystem> tester(*modelTree, CoordsSystem());

    CGAL::Bounded_side result = tester(point);

    return (result == CGAL::ON_BOUNDED_SIDE || result == CGAL::ON_BOUNDARY);
}

void Model::lazyAabbTreeConstruction()
{
    modelTree = new AABB_tree(faces(modelMesh).first, faces(modelMesh).second, modelMesh);
}

CGAL::Simple_cartesian<double>::Iso_cuboid_3 Model::bounding_box()
{
    return CGAL::bounding_box(modelMesh.points().begin(), modelMesh.points().end());
}

Point_3 Model::reflectionVector(CoordsSystem::Point_3 point_1, CoordsSystem::Point_3 point_2)
{
    Ray_intersection intersection = modelTree->first_intersection(CoordsSystem::Ray_3(point_1, point_2));

    Point_3 vector{point_1.x() - point_2.x(), point_1.y() - point_2.y(), point_1.z() - point_2.z()};

    vector.normaliseVector();

    if(intersection)
    {
        CoordsSystem::Vector_3 normal = CGAL::Polygon_mesh_processing::compute_face_normal(intersection->second, modelMesh);
        Point_3 result = rotateVectorAroundAxis({normal.x(), normal.y(), normal.z()}, vector, M_PI);
        return result;
    }

    return Point_3{0.0, 0.0, 0.0};
}