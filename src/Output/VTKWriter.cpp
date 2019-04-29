#include <fstream>
#include "VTKWriter.h"
#include <assert.h>


#include "VTK/vtk-unstructured.h"
#include "VTK/vtk-punstructured.h"

void VTKWriter::write(unsigned long iteration) {
#ifdef VTK
    this->initializeVTKFile();

    for (auto &p : this->particles) {
        this->plotParticle(p);
    }

    std::stringstream fileNameStream;
    fileNameStream << this->filename;

    // TODO MPI
/*#ifdef ENABLE_MPI
    fileNameStream << "_node" << rank;

	if (rank == 0) {
		int numProcs = 0;
		MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &numProcs) );
		outputParallelVTKFile(numProcs,simstep, impl);
	}
#endif*/
    fileNameStream << "_" << iteration << ".vtu";
    auto fileName = fileNameStream.str();

    this->vtkFile->UnstructuredGrid()->Piece().NumberOfPoints(this->particles.size()); // sets the number of points
    std::ofstream file(fileName);
    VTKFile(file, *this->vtkFile); //actually writes the file
#else
    assert(false && "GraPaSim was not compiled with VTK support, but the VTK writer is used");
#endif
}

void VTKWriter::plotParticle(const Particle &particle) {
#ifdef VTK
    PointData::DataArray_sequence &pointDataArraySequence = this->vtkFile->UnstructuredGrid()->Piece().PointData().DataArray();
    PointData::DataArray_iterator data_iterator = pointDataArraySequence.begin();
    // id
    data_iterator->push_back(particle.id);
    data_iterator++;
    // mpi-node rank
    data_iterator->push_back(this->rank);
    data_iterator++;
    // forces
    for (int i = 0; i < 3; ++i) {
        data_iterator->push_back(i < DIMENSIONS ? particle.F[i] : 0);
    }
    data_iterator++;
    // velocities
    for (int i = 0; i < 3; ++i) {
        data_iterator->push_back(i < DIMENSIONS ? particle.v[i] : 0);
    }

    // Coordinates
    Points::DataArray_sequence &pointsArraySequence = this->vtkFile->UnstructuredGrid()->Piece().Points().DataArray();
    Points::DataArray_iterator coordinates_iterator = pointsArraySequence.begin();

    // positions
    for (int i = 0; i < 3; ++i) {
        coordinates_iterator->push_back(i < DIMENSIONS ? particle.x[i] : 0);
    }
#endif
}

void VTKWriter::initializeVTKFile() {
#ifdef VTK
    // Add the data we want to output for each molecule.
    // The iterator over PointData traverses the DataArrays just in the order
    // in which we add them here.
    PointData pointData;
    DataArray_t particleId(type::Float32, "id", 1);
    pointData.DataArray().push_back(particleId);

    DataArray_t node_rank(type::Int32, "node-rank", 1);
    pointData.DataArray().push_back(node_rank);
    DataArray_t forces(type::Float32, "forces", 3);
    pointData.DataArray().push_back(forces);
    DataArray_t velocities(type::Float32, "velocities", 3);
    pointData.DataArray().push_back(velocities);

    CellData cellData; // we don't have cell data => leave it empty

    // coordinates
    Points points;
    DataArray_t pointCoordinates(type::Float32, "points", 3);
    points.DataArray().push_back(pointCoordinates);

    Cells cells; // we don't have cells, => leave it empty
    // for some reasons, we have to add a dummy entry for paraview
    DataArray_t cells_data(type::Float32, "types", 0);
    cells.DataArray().push_back(cells_data);

    PieceUnstructuredGrid_t piece(pointData, cellData, points, cells, 0, 0);
    UnstructuredGrid_t unstructuredGrid(piece);
    this->vtkFile = std::unique_ptr<VTKFile_t, VTKFile_tDeleter>(new VTKFile_t("UnstructuredGrid"));
    this->vtkFile->UnstructuredGrid(unstructuredGrid);
#endif
}

void VTKWriter::initializeParallelVTKFile(const std::vector<std::string> &fileNames) {
#ifdef VTK
    PPointData p_pointData;
    DataArray_t p_particleId(type::Float32, "id", 1);
    p_pointData.PDataArray().push_back(p_particleId);
    DataArray_t p_node_rank(type::Int32, "node-rank", 1);
    p_pointData.PDataArray().push_back(p_node_rank);
    DataArray_t p_forces(type::Float32, "forces", 3);
    p_pointData.PDataArray().push_back(p_forces);
    DataArray_t p_velocities(type::Float32, "velocities", 3);
    p_pointData.PDataArray().push_back(p_velocities);


    PCellData p_cellData; // we don't have cell data => leave it empty

    // coordinates
    PPoints p_points;
    DataArray_t p_pointCoordinates(type::Float32, "points", 3);
    p_points.PDataArray().push_back(p_pointCoordinates);

    PCells p_cells; // we don't have cells, => leave it empty
    // for some reasons, we have to add a dummy entry for paraview
    DataArray_t p_cells_data(type::Float32, "types", 0);
    p_cells.PDataArray().push_back(p_cells_data);

    PUnstructuredGrid_t p_unstructuredGrid(p_pointData, p_cellData, p_points, p_cells);
    for (const auto &fileName : fileNames) {
        Piece p_piece(fileName);
        p_unstructuredGrid.Piece().push_back(p_piece);
    }

    this->parallelVTKFile = std::unique_ptr<VTKFile_t, VTKFile_tDeleter>(new VTKFile_t("PUnstructuredGrid"));
    this->parallelVTKFile->PUnstructuredGrid(p_unstructuredGrid);
#endif
}

// Required because VTKFile_t is incomplete type in header
void VTKFile_tDeleter::operator()(VTKFile_t *p) {
    delete p;
}
