#include "fd_grad.h"
#include "index_utils.h"
#include <gtest/gtest.h>

TEST(GradTest, TestInvalidInputSize) {
  int nx = 3, ny = 4, nz = 5;
  int expected_rows =
      (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1);
  double h = 1.0;

  Eigen::SparseMatrix<double> G_small_rows(5, nx * ny * nz);
  ASSERT_DEATH(fd_grad(nx, ny, nz, h, G_small_rows), "");
  Eigen::SparseMatrix<double> G_large_rows(1000, nx * ny * nz);
  ASSERT_DEATH(fd_grad(nx, ny, nz, h, G_large_rows), "");

  Eigen::SparseMatrix<double> G_small_cols(expected_rows, nx * ny * nz - 1);
  ASSERT_DEATH(fd_grad(nx, ny, nz, h, G_small_cols), "");
  Eigen::SparseMatrix<double> G_large_cols(expected_rows, nx * ny * nz + 1);
  ASSERT_DEATH(fd_grad(nx, ny, nz, h, G_large_cols), "");
}

TEST(GradTest, TestLinearInput) {
  int nx = 4, ny = 3, nz = 5;
  int expected_rows =
      (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1);
  double h = 0.7;
  double slope_x = 2.2, slope_y = 5.0, slope_z = 9.12;

  Eigen::SparseMatrix<double> G(expected_rows, nx * ny * nz);
  fd_grad(nx, ny, nz, h, G);

  Eigen::VectorXd f(nx * ny * nz);
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        f(get_1D_index(i, j, k, nx, ny, nz)) =
            i * h * slope_x + j * h * slope_y + k * h * slope_z;
      }
    }
  }

  Eigen::VectorXd result = G * f;

  int offset = 0;
  int block_x = (nx - 1) * ny * nz;
  for (int i = 0; i < block_x; ++i)
    EXPECT_NEAR(result(offset + i), slope_x, 1e-6)
        << "X block mismatch at index " << i;

  offset += block_x;
  int block_y = nx * (ny - 1) * nz;
  for (int i = 0; i < block_y; ++i)
    EXPECT_NEAR(result(offset + i), slope_y, 1e-6)
        << "Y block mismatch at index " << i;

  offset += block_y;
  int block_z = nx * ny * (nz - 1);
  for (int i = 0; i < block_z; ++i)
    EXPECT_NEAR(result(offset + i), slope_z, 1e-6)
        << "Z block mismatch at index " << i;
}
