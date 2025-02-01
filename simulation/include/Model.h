#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <string>

using Meshdata = CGAL::Surface_mesh<CGAL::Exact_predicates_inexact_constructions_kernel::Point_3>;

class Model
{
public:
    void importModel(std::string filename);

    const Meshdata& getMesh() const { return modelMesh; };

private:
    Meshdata modelMesh;
};