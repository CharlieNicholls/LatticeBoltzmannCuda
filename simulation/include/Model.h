#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Simple_cartesian.h>
#include <string>

using Meshdata = CGAL::Surface_mesh<CGAL::Simple_cartesian<double>::Point_3>;

class Model
{
public:
    void importModel(std::string filename);

    bool isModelClosed();

    bool isPointInsideModel(CGAL::Simple_cartesian<double>::Point_3 point);

    const Meshdata& getMesh() const { return modelMesh; };

    bool checkError() { return !meshValid; };

private:
    Meshdata modelMesh;
    bool meshValid = true;
};