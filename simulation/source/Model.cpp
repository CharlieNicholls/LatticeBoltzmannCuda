#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/boost/graph/helpers.h>

#include "Model.h"

void Model::importModel(std::string filename)
{
    CGAL::IO::read_polygon_mesh(filename, modelMesh);
}

bool Model::isModelClosed()
{
    return CGAL::is_closed(modelMesh);
}