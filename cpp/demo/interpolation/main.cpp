
#include <basix/finite-element.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/dolfinx_geometry.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <dolfinx.h>
#include <span>
#include <typeinfo>

using namespace dolfinx;
using T = double;

int main(int argc, char* argv[]) {
  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  
  // auto element = basix::FiniteElement<T>(
  //       basix::element::family::P, basix::cell::type::triangle, 2,
  //       basix::element::lagrange_variant::unset,
  //       basix::element::dpc_variant::unset, false);

  int num_cells = 10;
  T lower = 0.;
  T upper = 10.;
  auto element = basix::create_element<T>(
                                         basix::element::family::P,
                                         basix::cell::type::interval,
                                         2,
                                         basix::element::lagrange_variant::unset,
                                         basix::element::dpc_variant::unset, false
                                         );
  const auto mesh = std::make_shared<mesh::Mesh<T>>(
                                              mesh::create_interval<T>(
                                                                       MPI_COMM_WORLD,
                                                                       num_cells,
                                                                       {lower, upper}));

  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, element, {}));

  auto f = std::make_shared<fem::Function<T>>(V);

  auto dof = f->x()->array();
  std::cout << "Type of dof is: " << typeid(dof).name() << std::endl;

  for (auto x: dof) {
    std::cout << x << ' ';
  }
  std::cout << std::endl;

  f->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> f;
        for (std::size_t p = 0; p < x.extent(1); ++p)
        {
          f.push_back(x(0, p) * x(0, p));
        }

        return {f, {f.size()}};
      });

  dof = f->x()->array();
  for (auto x: dof) {
    std::cout << x << ' ';
  }
  std::cout << std::endl;

  /* Create coordinates */

  int num_evals = 100;
  std::vector<T> v(num_evals);
  for (int i = 0; i < num_evals; i++) {
    v[i] = i*upper/(T)num_evals;
  }

  std::vector<T> v_coords(3*num_evals, 0.);
  {
    int i = 0;
    int j = 0;
    while (i < v_coords.size())
    {
      v_coords[i] = v[j];
      j += 1;
      i += 3;
    }
  }
  std::span<const T> coords(v_coords);

  const mesh::Mesh<T> mesh2 = *f->function_space()->mesh();

  const auto bb_tree = geometry::BoundingBoxTree<T>(mesh2, 1, 1e-10);
  auto potential_cells = geometry::compute_collisions(bb_tree, coords);
  auto colliding_cells = geometry::compute_colliding_cells(*f->function_space()->mesh(), potential_cells, coords);

  return 0;
}
