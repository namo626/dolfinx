#include "CUDAeval.h"
#include <__clang_cuda_builtin_vars.h>

namespace dolfinx::fem
{
    __global__ void _basis_expansion(int p, int bs_element, int space_dimension,
                                    int value_size, int ushape1, double* u,
                                    const double* coefficients,
                                    const double* basis_values) {

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

    void basis_expansion(int p, int bs_element, int space_dimension,
                                    int value_size, int ushape1, double* u,
                                    const double* coefficients,
                                    const double* basis_values) {
      _basis_expansion<<<1, 1>>>(p, bs_element, space_dimension, value_size,
                                 ushape1, u, coefficients, basis_values);
    }

    __global__ void _CUDAexpand(int num_cells, int bs_element,
                                int space_dimension, int value_size, double* u,
                                int ushape1, double* coefficients_p,
                                double* basis_values_p)
    {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p < num_cells)
        {
          for (int k = 0; k < bs_element; ++k)
          {
            for (std::size_t i = 0; i < space_dimension; ++i)
            {
              for (std::size_t j = 0; j < value_size; ++j)
              {
                u[p * ushape[1] + (j * bs_element + k)]
                    += coefficients[bs_element * i + k + p]
                       * basis_values[i * j + p];
              }
            }
          }
        }
    }

    void CUDAexpand(int num_cells, int bs_element, int space_dimension,
                    int value_size, double* u, int ushape1,
                    double* coefficients_p, double* basis_values_p) {

      const int numthreads = 128;
      const int numblocks = num_cells / numthreads + 1;
      _CUDAexpand<<<numblocks, numthreads>>>(
          num_cells, bs_element, space_dimension, value_size, u, ushape1,
          coefficients_p, basis_values_p);
        
    }

} // namespace dolfinx::fem
