#include "fd_interpolate.h"
#include "index_utils.h"

void fd_interpolate(const int nx, const int ny, const int nz, const double h,
                    const Eigen::RowVector3d &corner, const Eigen::MatrixXd &P,
                    Eigen::SparseMatrix<double> &W) {
  assert(P.cols() == 3 && "P must have 3 columns (x,y,z)");
  for (int row = 0; row < P.rows(); ++row) {
    int i = (P(row, 0) - corner(0)) / h;
    int j = (P(row, 1) - corner(1)) / h;
    int k = (P(row, 2) - corner(2)) / h;

    int base_index = get_1D_index(i, j, k, nx, ny, nz);
    double w_x = (P(row, 0) - corner(0)) / h - i;
    double w_y = (P(row, 1) - corner(1)) / h - j;
    double w_z = (P(row, 2) - corner(2)) / h - k;

    W.insert(row, base_index) = (1 - w_x) * (1 - w_y) * (1 - w_z);
    W.insert(row, base_index + 1) = (w_x) * (1 - w_y) * (1 - w_z);
    W.insert(row, base_index + nx) = (1 - w_x) * (w_y) * (1 - w_z);
    W.insert(row, base_index + nx + 1) = (w_x) * (w_y) * (1 - w_z);
    W.insert(row, base_index + nx * ny) = (1 - w_x) * (1 - w_y) * (w_z);
    W.insert(row, base_index + nx * ny + 1) = (w_x) * (1 - w_y) * (w_z);
    W.insert(row, base_index + nx * ny + nx) = (1 - w_x) * (w_y) * (w_z);
    W.insert(row, base_index + nx * ny + nx + 1) = (w_x) * (w_y) * (w_z);
  }
}
