#include "fd_grad.h"
#include "fd_partial_derivative.h"
#include "matrix_utils.h"

void fd_grad(const int nx, const int ny, const int nz, const double h,
             Eigen::SparseMatrix<double> &G) {
  Eigen::SparseMatrix<double> Dx((nx - 1) * ny * nz, nx * ny * nz);
  fd_partial_derivative(nx, ny, nz, h, 0, Dx);
  Eigen::SparseMatrix<double> Dy(nx * (ny - 1) * nz, nx * ny * nz);
  fd_partial_derivative(nx, ny, nz, h, 1, Dy);
  Eigen::SparseMatrix<double> Dz(nx * ny * (nz - 1), nx * ny * nz);
  fd_partial_derivative(nx, ny, nz, h, 2, Dz);

  std::vector<const Eigen::SparseMatrix<double> *> matrices = {&Dx, &Dy, &Dz};
  vstackSparseMatrix(matrices, G);
}
