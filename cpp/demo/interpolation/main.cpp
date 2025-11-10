
#include <basix/finite-element.h>
#include <cstddef>
#include <cstdint>
#include <dolfinx/fem/Function.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/dolfinx_geometry.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <dolfinx.h>
#include <span>
#include <typeinfo>

template<typename T>
void printVec(const std::vector<T>& vec) {
  for (auto v : vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}
template<typename T>
void printSpan(const std::span<T>& sp) {
  for (auto v : sp) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}

using namespace dolfinx;
using T = double;

std::vector<int> get_cells(int elem_size, std::vector<T> coords) {
  std::vector<int> cells;
  for ( auto x : coords ) {
    cells.push_back((int)std::floor(x / (T)elem_size));
  }
  return cells;
}

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

  const int num_evals = 20;
  std::vector<T> v(num_evals);
  for (int i = 0; i < num_evals; i++) {
    v[i] = i*upper/(T)num_evals;
  }
  std::cout << "Coordinates = " << std::endl;
  printVec(v);

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
  std::cout << "3D Coordinates = " << std::endl;
  //printVec(v_coords);
  printSpan(coords);

#if 1
  const int tdim = mesh->topology()->dim();
  auto cell_map = mesh->topology()->index_map(tdim);
  std::vector<std::int32_t> entities(cell_map->size_local(), 0);
  std::cout << "Entities size = " << entities.size() << std::endl;

  std::vector<int> ent = {0,1,2,3,4,5,6,7,8,9};

  const auto bb_tree = geometry::BoundingBoxTree<T>(*mesh, mesh->topology()->dim(), ent ,1e-12);
  std::cout << "Num_bboxes = " << bb_tree.num_bboxes() << std::endl;
  auto potential_cells = geometry::compute_collisions(bb_tree, coords);
  auto colliding_cells = geometry::compute_colliding_cells(*mesh, potential_cells, coords);

  std::vector<std::int32_t> cells;
  for (int i = 0; i < num_evals; i++) {
    //std::cout << "Link size: " << colliding_cells.links(1).size() << std::endl;
    if (colliding_cells.links(i).size() > 0)
      cells.push_back(colliding_cells.links(i)[0]);
  }

  std::cout << "Colliding cells size = " << cells.size() << std::endl;
#endif 

  /* Find cell index that each point resides in */

  printVec(cells);

  //std::span<const std::int32_t> cells_span(cells);

  const std::size_t value_size = f->function_space()->value_size();
  std::cout << "Value_size = " << value_size  << std::endl;
  std::vector<T> u(num_evals * value_size);
  f->eval(coords, {num_evals, 3}, cells, u, {num_evals, value_size});

  printVec(u);

  return 0;
}

