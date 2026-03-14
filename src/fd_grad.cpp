#include "fd_grad.h"
#include "fd_partial_derivative.h"
#include "index_utils.h"
#include "matrix_utils.h"

void fd_grad(const int nx, const int ny, const int nz, const double h,
             Eigen::SparseMatrix<double> &G) {
  Eigen::SparseMatrix<double> Dx((nx - 1) * ny * nz, nx * ny * nz);
  fd_partial_derivative(nx, ny, nz, h, X_DIR, Dx);
  Eigen::SparseMatrix<double> Dy(nx * (ny - 1) * nz, nx * ny * nz);
  fd_partial_derivative(nx, ny, nz, h, Y_DIR, Dy);
  Eigen::SparseMatrix<double> Dz(nx * ny * (nz - 1), nx * ny * nz);
  fd_partial_derivative(nx, ny, nz, h, Z_DIR, Dz);

  std::vector<const Eigen::SparseMatrix<double> *> matrices = {&Dx, &Dy, &Dz};
  vstackSparseMatrix(matrices, G);
}
