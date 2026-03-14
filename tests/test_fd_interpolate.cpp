#include "fd_interpolate.h"
#include "index_utils.h"
#include <gtest/gtest.h>

void expect_single_one_in_row(const Eigen::SparseMatrix<double> &W, int row,
                              int oneIndex);

TEST(InterpolateTest, TestValueAtCenter) {
  int nx = 2;
  int ny = 2;
  int nz = 2;
  double h = 1;
  Eigen::RowVector3d corner(0, 0, 0);
  Eigen::MatrixXd P(1, 3);
  P << 0.5, 0.5, 0.5;
  Eigen::SparseMatrix<double> W(1, 8);

  fd_interpolate(nx, ny, nz, h, corner, P, W);

  for (int j = 0; j < W.cols(); ++j) {
    EXPECT_NEAR(W.coeff(0, j), 1.0 / 8.0, 1e-6)
        << "One of the interpolation weights is not as expected";
  }
}

TEST(InterpolateTest, TestValueAtCorners) {
  int nx = 3;
  int ny = 3;
  int nz = 3;
  double h = 1;
  Eigen::RowVector3d corner(0, 0, 0);
  Eigen::MatrixXd P(4, 3);
  P << 0, 0, 0, //
      0, 0, 1,  //
      0, 1, 0,  //
      1, 1, 1;
  Eigen::SparseMatrix<double> W(4, 27);

  fd_interpolate(nx, ny, nz, h, corner, P, W);

  expect_single_one_in_row(W, 0, 0);
  expect_single_one_in_row(W, 1, 9);
  expect_single_one_in_row(W, 2, 3);
  expect_single_one_in_row(W, 3, 1 + 3 + 9);
}

TEST(InterpolateTest, TestPointOutsideGrid) {
  int nx = 3;
  int ny = 3;
  int nz = 3;
  double h = 1;
  Eigen::RowVector3d corner(0, 0, 0);
  Eigen::MatrixXd P(2, 3);
  P << 0, 0, 0, //
      0, 10, 0;
  Eigen::SparseMatrix<double> W(2, nx * ny * nz);

  EXPECT_DEATH(fd_interpolate(nx, ny, nz, h, corner, P, W), "");
}

TEST(InterpolateTest, TestTwoPoints) {
  int nx = 10;
  int ny = 10;
  int nz = 10;
  double h = 0.5;
  Eigen::RowVector3d corner(2, 3, 5);
  Eigen::MatrixXd P(2, 3);
  P << 5, 6.2, 6, //
      4, 4.7, 8.4;
  Eigen::SparseMatrix<double> W(P.rows(), nx * ny * nz);
  fd_interpolate(nx, ny, nz, h, corner, P, W);

  Eigen::MatrixXd X(nx * ny * nz, 3);
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        int index = get_1D_index(i, j, k, nx, ny, nz);
        X(index, 0) = corner(0) + h * i;
        X(index, 1) = corner(1) + h * j;
        X(index, 2) = corner(2) + h * k;
      }
    }
  }

  Eigen::MatrixXd results = W * X;
  for (int i = 0; i < results.rows(); ++i) {
    for (int j = 0; j < results.cols(); ++j) {
      EXPECT_NEAR(P(i, j), results(i, j), 1e-6)
          << "Mismatch at (" << i << "," << j << ")";
    }
  }
}

void expect_single_one_in_row(const Eigen::SparseMatrix<double> &W, int row,
                              int oneIndex) {
  for (int j = 0; j < W.cols(); ++j) {
    if (j == oneIndex) {
      EXPECT_NEAR(W.coeff(row, j), 1.0, 1e-6)
          << "Weight should be 1 at row " << row << " and col " << j;
    } else {
      EXPECT_NEAR(W.coeff(row, j), 0.0, 1e-6)
          << "Weight should be 0 at row " << row << " and col " << j;
    }
  }
}
