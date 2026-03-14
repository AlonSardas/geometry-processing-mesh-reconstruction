#include "fd_partial_derivative.h"
#include "index_utils.h"
#include <gtest/gtest.h>

int count_nonzeros_in_row(const Eigen::SparseMatrix<double> &M, int row);

TEST(PartialDerivativeTest, TestXDir) {
  int nx = 2;
  int ny = 2;
  int nz = 1;

  double h = 1.0;
  int dir = 0;

  Eigen::SparseMatrix<double> D((nx - 1) * ny * nz, nx * ny * nz);

  fd_partial_derivative(nx, ny, nz, h, dir, D);
  EXPECT_NEAR(D.coeff(get_1D_index(0, 0, 0, nx - 1, ny, nz),
                      get_1D_index(1, 0, 0, nx, ny, nz)),
              1, 1e-6);
  EXPECT_NEAR(D.coeff(get_1D_index(0, 0, 0, nx - 1, ny, nz),
                      get_1D_index(0, 0, 0, nx, ny, nz)),
              -1, 1e-6);
  EXPECT_NEAR(D.coeff(get_1D_index(0, 1, 0, nx - 1, ny, nz),
                      get_1D_index(1, 1, 0, nx, ny, nz)),
              1, 1e-6);
  EXPECT_NEAR(D.coeff(get_1D_index(0, 1, 0, nx - 1, ny, nz),
                      get_1D_index(0, 1, 0, nx, ny, nz)),
              -1, 1e-6);
}

TEST(PartialDerivativeTest, TestGridSize) {
  int nx = 2;
  int ny = 2;
  int nz = 2;

  double h = 3.2;
  int dir = 1;

  Eigen::SparseMatrix<double> D(nx * (ny - 1) * nz, nx * ny * nz);

  fd_partial_derivative(nx, ny, nz, h, dir, D);
  EXPECT_NEAR(D.coeff(get_1D_index(0, 0, 0, nx, ny - 1, nz),
                      get_1D_index(0, 1, 0, nx, ny, nz)),
              1.0 / h, 1e-6);
  EXPECT_NEAR(D.coeff(get_1D_index(0, 0, 0, nx, ny - 1, nz),
                      get_1D_index(0, 0, 0, nx, ny, nz)),
              -1.0 / h, 1e-6);
  EXPECT_NEAR(D.coeff(get_1D_index(0, 0, 1, nx, ny - 1, nz),
                      get_1D_index(0, 1, 1, nx, ny, nz)),
              1.0 / h, 1e-6);
  EXPECT_NEAR(D.coeff(get_1D_index(0, 0, 1, nx, ny - 1, nz),
                      get_1D_index(0, 0, 1, nx, ny, nz)),
              -1.0 / h, 1e-6);
}

TEST(PartialDerivativeTest, TestInvalidDir) {
  int nx = 2;
  int ny = 2;
  int nz = 2;
  double h = 1;
  Eigen::SparseMatrix<double> D((nx - 1) * ny * nz, nx * ny * nz);

  EXPECT_DEATH(fd_partial_derivative(nx, ny, nz, h, -1, D), "dir");
  EXPECT_DEATH(fd_partial_derivative(nx, ny, nz, h, 3, D), "dir");
}

TEST(PartialDerivativeTest, TestTwoNonzerosPerRow) {
  int nx = 4, ny = 3, nz = 5;
  Eigen::SparseMatrix<double> D(nx * ny * (nz - 1), nx * ny * nz);
  fd_partial_derivative(nx, ny, nz, 1.0, 2, D);
  for (int i = 0; i < D.rows(); ++i) {
    EXPECT_EQ(count_nonzeros_in_row(D, i), 2) << "Row " << i;
  }
}

TEST(PartialDerivativeTest, TestRowSumsToZero) {
  int nx = 4, ny = 3, nz = 2;
  Eigen::SparseMatrix<double> D((nx - 1) * ny * nz, nx * ny * nz);
  fd_partial_derivative(nx, ny, nz, 53.4, 0, D);
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(D.cols());
  Eigen::VectorXd row_sums = D * ones;
  for (int i = 0; i < row_sums.size(); ++i) {
    EXPECT_NEAR(row_sums(i), 0.0, 1e-6)
        << "Row " << i << " doesn't sum to zero";
  }
}

TEST(PartialDerivativeTest, TestConstantFieldYDir) {
  int nx = 4, ny = 3, nz = 2;
  Eigen::SparseMatrix<double> D(nx * (ny - 1) * nz, nx * ny * nz);
  fd_partial_derivative(nx, ny, nz, 1.0, 1, D);
  Eigen::VectorXd constant = Eigen::VectorXd::Constant(nx * ny * nz, 5.0);
  Eigen::VectorXd result = D * constant;
  EXPECT_TRUE(result.isZero(1e-6))
      << "Derivative of constant field should be zero";
}

TEST(PartialDerivativeTest, TestLinearFieldXDir) {
  int nx = 4, ny = 3, nz = 2;
  double h = 0.5;
  Eigen::SparseMatrix<double> D((nx - 1) * ny * nz, nx * ny * nz);
  fd_partial_derivative(nx, ny, nz, h, 0, D);

  Eigen::VectorXd f(nx * ny * nz);
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
        f(get_1D_index(i, j, k, nx, ny, nz)) = i * h;

  Eigen::VectorXd result = D * f;
  for (int i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result(i), 1.0, 1e-6) << "Expected df/dx=1 at index " << i;
  }
}

int count_nonzeros_in_row(const Eigen::SparseMatrix<double> &M, int row) {
  int count = 0;
  for (int col = 0; col < M.cols(); ++col)
    if (M.coeff(row, col) != 0)
      count++;
  return count;
}
