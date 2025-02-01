#include "Model.h"
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <iostream>

void Model::importModel(std::string filename)
{
    CGAL::IO::read_polygon_mesh(filename, modelMesh);
}