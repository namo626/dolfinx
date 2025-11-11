#include <cstdio>

namespace dolfinx::fem
{
__global__ void basis_expansion(int p, int bs_element, int space_dimension,
                                int value_size, int ushape1, double* u,
                                const double* coefficients,
                                const double* basis_values)
{

  // Compute expansion
  for (int k = 0; k < bs_element; ++k)
  {
    for (std::size_t i = 0; i < space_dimension; ++i)
    {
      for (std::size_t j = 0; j < value_size; ++j)
      {
        u[p * ushape1 + (j * bs_element + k)]
            += coefficients[bs_element * i + k]
               * basis_values[i * value_size + j];
      }
    }
  }
}
} // namespace dolfinx::fem
