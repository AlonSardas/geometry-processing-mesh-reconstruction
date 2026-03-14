#ifndef INDEX_UTILS_H
#define INDEX_UTILS_H

inline unsigned int get_1D_index(unsigned int i, unsigned int j, unsigned int k,
                                 unsigned int nx, unsigned int ny,
                                 unsigned int nz) {
  return i + j * nx + k * nx * ny;
}

#endif