#include <Eigen/Sparse>
#include <vector>

inline void vstackSparseMatrix(
    const std::vector<const Eigen::SparseMatrix<double> *> &matrices,
    Eigen::SparseMatrix<double> &out) {
  int total_rows = 0;
  int non_zeroes = 0;
  int cols = out.cols();
  for (const auto &m : matrices) {
    assert(m->cols() == cols);
    total_rows += m->rows();
    non_zeroes += m->nonZeros();
  }
  assert(out.rows() == total_rows);

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(non_zeroes);

  int row_offset = 0;
  for (const auto &m : matrices) {
    for (int k = 0; k < m->outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(*m, k); it; ++it) {
        triplets.push_back({static_cast<int>(it.row()) + row_offset,
                            static_cast<int>(it.col()), it.value()});
      }
    }
    row_offset += m->rows();
  }
  out.setFromTriplets(triplets.begin(), triplets.end());
}
