#include "fd_interpolate.h"
#include "index_utils.h"

void fd_interpolate(const int nx, const int ny, const int nz, const double h,
                    const Eigen::RowVector3d &corner, const Eigen::MatrixXd &P,
                    Eigen::SparseMatrix<double> &W) {
  assert(P.cols() == 3 && "P must have 3 columns (x,y,z)");

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(P.rows() * 8);

  for (int row = 0; row < P.rows(); ++row) {
    // std::cout << "interpolating row " << row << std::endl;
    double dx = (P(row, 0) - corner(0)) / h;
    double dy = (P(row, 1) - corner(1)) / h;
    double dz = (P(row, 2) - corner(2)) / h;
    int i = (int)dx, j = (int)dy, k = (int)dz;
    double w_x = dx - i, w_y = dy - j, w_z = dz - k;

    int base_index = get_1D_index(i, j, k, nx, ny, nz);

    triplets.push_back({row, base_index, (1 - w_x) * (1 - w_y) * (1 - w_z)});
    triplets.push_back({row, base_index + 1, (w_x) * (1 - w_y) * (1 - w_z)});
    triplets.push_back({row, base_index + nx, (1 - w_x) * (w_y) * (1 - w_z)});
    triplets.push_back({row, base_index + nx + 1, (w_x) * (w_y) * (1 - w_z)});
    triplets.push_back(
        {row, base_index + nx * ny, (1 - w_x) * (1 - w_y) * (w_z)});
    triplets.push_back(
        {row, base_index + nx * ny + 1, (w_x) * (1 - w_y) * (w_z)});
    triplets.push_back(
        {row, base_index + nx * ny + nx, (1 - w_x) * (w_y) * (w_z)});
    triplets.push_back(
        {row, base_index + nx * ny + nx + 1, (w_x) * (w_y) * (w_z)});
  }

  W.setFromTriplets(triplets.begin(), triplets.end());
}
