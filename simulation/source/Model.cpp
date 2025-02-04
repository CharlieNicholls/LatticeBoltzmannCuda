#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/bounding_box.h>

#include "Model.h"

typedef CGAL::Simple_cartesian<double> CoordsSystem;
typedef CGAL::Surface_mesh<CoordsSystem::Point_3> Mesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<CoordsSystem, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;

void Model::importModel(std::string filename)
{
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