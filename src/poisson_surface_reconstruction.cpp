#include "poisson_surface_reconstruction.h"
#include "fd_grad.h"
#include "fd_interpolate.h"
#include "index_utils.h"
// #include <Eigen/src/IterativeLinearSolvers/BiCGSTAB.h>
#include <algorithm>
#include <igl/copyleft/marching_cubes.h>

#include <iostream>

Eigen::VectorXd compute_vector_component_staggered(
    const int nx, const int ny, const int nz, const double h, const int dir,
    const Eigen::RowVector3d &corner, const Eigen::MatrixXd &P,
    const Eigen::MatrixXd &N);

void poisson_surface_reconstruction(const Eigen::MatrixXd &P,
                                    const Eigen::MatrixXd &N,
                                    Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
  ////////////////////////////////////////////////////////////////////////////
  // Construct FD gridgered grid values in v via sparse trilinear interp,
  // CONGRATULATIONS! You get this for free!
  ////////////////////////////////////////////////////////////////////////////
  // number of input points
  const int n = P.rows();
  // Grid dimensions
  int nx, ny, nz;
  // Maximum extent (side length of bounding box) of points
  double max_extent =
      (P.colwise().maxCoeff() - P.colwise().minCoeff()).maxCoeff();
  // padding: number of cells beyond bounding box of input points
  const double pad = 8;
  // choose grid spacing (h) so that shortest side gets 30+2*pad samples
  double h = max_extent / double(30 + 2 * pad);
  // Place bottom-left-front corner of grid at minimum of points minus padding
  Eigen::RowVector3d corner = P.colwise().minCoeff().array() - pad * h;
  // Grid dimensions should be at least 3
  nx = std::max(
      (P.col(0).maxCoeff() - P.col(0).minCoeff() + (2. * pad) * h) / h, 3.);
  ny = std::max(
      (P.col(1).maxCoeff() - P.col(1).minCoeff() + (2. * pad) * h) / h, 3.);
  nz = std::max(
      (P.col(2).maxCoeff() - P.col(2).minCoeff() + (2. * pad) * h) / h, 3.);
  // Compute positions of grid nodes
  Eigen::MatrixXd x(nx * ny * nz, 3);
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        // Convert subscript to index
        const auto ind = i + nx * (j + k * ny);
        x.row(ind) = corner + h * Eigen::RowVector3d(i, j, k);
      }
    }
  }
  Eigen::VectorXd g = Eigen::VectorXd::Zero(nx * ny * nz);

  // Compute the normals on a staggered grid.
  std::cout << "Computing the normals" << std::endl;
  Eigen::VectorXd v_x =
      compute_vector_component_staggered(nx, ny, nz, h, X_DIR, corner, P, N);
  Eigen::VectorXd v_y =
      compute_vector_component_staggered(nx, ny, nz, h, Y_DIR, corner, P, N);
  Eigen::VectorXd v_z =
      compute_vector_component_staggered(nx, ny, nz, h, Z_DIR, corner, P, N);
  Eigen::VectorXd v(v_x.size() + v_y.size() + v_z.size());
  v << v_x, v_y, v_z;
  assert(v.size() ==
         (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1));

  // Compute the grad matrix
  std::cout << "Computing the grad matrix G" << std::endl;
  Eigen::SparseMatrix<double> G(v.size(), g.size());
  fd_grad(nx, ny, nz, h, G);

  // Now we need to solve
  // G.T * G * g = G.T v
  std::cout << "Solving for indicator func g" << std::endl;
  Eigen::SparseMatrix<double> Gt = G.transpose();
  Eigen::SparseMatrix<double> GtG = Gt * G;

  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
  // I tried these solvers but they are slower
  // Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
  // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  // This seems comparable
  // Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
  //                        Eigen::Lower | Eigen::Upper>
  //   solver;

  solver.compute(GtG);
  Eigen::VectorXd b = Gt * v;
  g = solver.solve(b);
  std::cout << "success: " << (solver.info() == Eigen::Success) << std::endl;

  // Computing sigma of iso-level
  std::cout << "Computing iso-level sigma" << std::endl;
  Eigen::SparseMatrix<double> W(n, nx * ny * nz);
  fd_interpolate(nx, ny, nz, h, corner, P, W);
  double sigma = (W * g).mean();
  g.array() -= sigma;

  std::cout << "Reconstructing the mesh" << std::endl;
  ////////////////////////////////////////////////////////////////////////////
  // Run black box algorithm to compute mesh from implicit function: this
  // function always extracts g=0, so "pre-shift" your g values by -sigma
  ////////////////////////////////////////////////////////////////////////////
  igl::copyleft::marching_cubes(g, x, nx, ny, nz, V, F);
}

Eigen::VectorXd compute_vector_component_staggered(
    const int nx, const int ny, const int nz, const double h, const int dir,
    const Eigen::RowVector3d &corner, const Eigen::MatrixXd &P,
    const Eigen::MatrixXd &N) {
  assert(0 <= dir && dir <= 2 && "dir must be 0,1,2");

  const int n = N.rows();

  int dir_indicator[3]{};
  dir_indicator[dir] = 1;

  // Distribute the normals on the staggered matrix
  Eigen::SparseMatrix<double> W(n, (nx - dir_indicator[0]) *
                                       (ny - dir_indicator[1]) *
                                       (nz - dir_indicator[2]));
  Eigen::RowVector3d corner_staggered = corner;
  corner_staggered(dir) += 0.5 * h;
  fd_interpolate(nx - dir_indicator[0], ny - dir_indicator[1],
                 nz - dir_indicator[2], h, corner_staggered, P, W);
  return W.transpose() * N.col(dir);
}
