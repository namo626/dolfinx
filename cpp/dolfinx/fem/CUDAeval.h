#ifndef CUDAEVAL_H_
#define CUDAEVAL_H_

#include <cstdio>
#include <cstdint>
#include <array>
#include <span>

namespace dolfinx::fem
{
  void CUDAeval(std::span<const double> x, std::array<std::size_t, 2> xshape,
            std::span<const std::int32_t> cells, std::span<double> u,
            std::array<std::size_t, 2> ushape);

  void basis_expansion(int p, int bs_element, int space_dimension,
                                int value_size, int ushape1, double* u,
                                const double* coefficients,
                                const double* basis_values);
}
#endif // CUDAEVAL_H_
