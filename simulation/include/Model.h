#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Simple_cartesian.h>
#include <string>

using Meshdata = CGAL::Surface_mesh<CGAL::Simple_cartesian<double>::Point_3>;

// Will need to work on removing the headers and typedefs from this header
typedef CGAL::Simple_cartesian<double> CoordsSystem;
typedef CGAL::Surface_mesh<CoordsSystem::Point_3> Mesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<CoordsSystem, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;

class Model
{
public:
    void importModel(std::string filename);

    bool isModelClosed();

    bool isPointInsideModel(CGAL::Simple_cartesian<double>::Point_3 point);

    const Meshdata& getMesh() const { return modelMesh; };

    bool checkError() { return !meshValid; };

    CGAL::Simple_cartesian<double>::Iso_cuboid_3 bounding_box();

    CoordsSystem::Vector_3 reflectionVector(CoordsSystem::Point_3 point_1, CoordsSystem::Point_3 point_2);

private:
    void lazyAabbTreeConstruction();

    AABB_tree* modelTree = nullptr;

    Meshdata modelMesh;
    bool meshValid = true;
};