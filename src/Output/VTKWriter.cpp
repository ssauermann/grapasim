#include "VTKWriter.h"

void VTKWriter::write(unsigned long iteration) {




}

void VTKWriter::plotParticle(Particle &particle) {

}

void VTKWriter::initializeVTKFile() {
    // Add the data we want to output for each molecule.
    // The iterator over PointData traverses the DataArrays just in the order
    // in which we add them here.
    PointData pointData;
    DataArray_t particleId(type::Float32, "id", 1);
    pointData.DataArray().push_back(moleculeId);

    DataArray_t node_rank(type::Int32, "node-rank", 1);
    pointData.DataArray().push_back(node_rank);
    DataArray_t forces(type::Float32, "forces", 3);
    pointData.DataArray().push_back(forces);

    CellData cellData; // we don't have cell data => leave it empty

    // 3 coordinates
    Points points;
    DataArray_t pointCoordinates(type::Float32, "points", 3);
    points.DataArray().push_back(pointCoordinates);

    Cells cells; // we don't have cells, => leave it empty
    // for some reasons, we have to add a dummy entry for paraview
    DataArray_t cells_data(type::Float32, "types", 0);
    cells.DataArray().push_back(cells_data);

    PieceUnstructuredGrid_t piece(pointData, cellData, points, cells, 0, 0);
    UnstructuredGrid_t unstructuredGrid(piece);
    _vtkFile = new VTKFile_t("UnstructuredGrid");
    _vtkFile->UnstructuredGrid(unstructuredGrid);

}
