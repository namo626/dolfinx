#ifndef CUDAEVAL_H_
#define CUDAEVAL_H_

#include <cstdio>
#include <cstdint>
#include <array>
#include <span>

namespace dolfinx::fem
{
  template<typename geometry_type, typename value_type>
  void CUDAeval(std::span<const geometry_type> x, std::array<std::size_t, 2> xshape,
            std::span<const std::int32_t> cells, std::span<value_type> u,
            std::array<std::size_t, 2> ushape);

__global__ void basis_expansion(int p, int bs_element, int space_dimension,
                                int value_size, int ushape1, double* u,
                                const double* coefficients,
                                const double* basis_values);
}
#endif // CUDAEVAL_H_
