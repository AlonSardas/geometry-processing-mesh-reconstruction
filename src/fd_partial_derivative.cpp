#include "fd_partial_derivative.h"
#include "index_utils.h"

void fd_partial_derivative(const int nx, const int ny, const int nz,
                           const double h, const int dir,
                           Eigen::SparseMatrix<double> &D) {
  assert(0 <= dir && dir <= 2 && "dir must be 0,1,2");

  int dir_indicator[3]{};
  dir_indicator[dir] = 1;

  int target_nx = nx - dir_indicator[0];
  int target_ny = ny - dir_indicator[1];
  int target_nz = nz - dir_indicator[2];

  for (unsigned int i = 0; i < target_nx; ++i) {
    for (unsigned int j = 0; j < target_ny; ++j) {
      for (unsigned int k = 0; k < target_nz; ++k) {
        unsigned int target_index =
            get_1D_index(i, j, k, target_nx, target_ny, target_nz);
        unsigned int index0 = get_1D_index(i, j, k, nx, ny, nz);
        D.insert(target_index, index0) = -1.0 / h;
        unsigned int index1 =
            get_1D_index(i + dir_indicator[0], j + dir_indicator[1],
                         k + dir_indicator[2], nx, ny, nz);
        D.insert(target_index, index1) = 1.0 / h;
      }
    }
  }
}
