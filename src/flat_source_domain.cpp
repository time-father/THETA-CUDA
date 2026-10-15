#include "flat_source_domain.h"

#include <functional>
#include <unordered_set>
#include <vector>

#include "memory.h"
#include "simulation.h"
#include "xml_interface.h"


namespace theta {

//==============================================================================
// Destructor
//==============================================================================
FlatSourceDomain::~FlatSourceDomain()
{
  // 1. Free geometry objects
  for (auto& g : geometry_) {
    if (g != nullptr) {
      delete g;
      g = nullptr;
    }
  }
  geometry_.clear();


}

//==============================================================================
//==============================================================================
FlatSourceDomain::FlatSourceDomain(
  const int64_t& num_of_groups, pugi::xml_node node, bool is_mesh)
{
  // [FIXED] Initialize pointers to nullptr to prevent segfaults in destructor

  // TODO:
  active_ = false;
  n_realizations_ = 0;

  // Set ng_
  ng_ = num_of_groups;

  //----------------------------------------------------------------------------
  // Initialize geometry information from mode->geometry node
  //----------------------------------------------------------------------------
  // Read x division
  if (check_for_node(node, "x")) {
    // Get x model
    pugi::xml_node x_node = node.child("x");
    // Generate x division
    geometry_.push_back(new Division(x_node));
  } else {
    fatal_error("```x``` node must be provided in input.xml!");
  }

  // Read y division (optional)
  if (check_for_node(node, "y")) {
    // Get y model
    pugi::xml_node y_node = node.child("y");
    // Generate y division
    geometry_.push_back(new Division(y_node));
  }

  // Read z division (optional)
  if (check_for_node(node, "z")) {
    // Get z model
    pugi::xml_node z_node = node.child("z");
    // Generate z division
    geometry_.push_back(new Division(z_node));
  }

  // Get the number of dimensions about the model
  n_dim_ = geometry_.size();

  // Get the number of fsr regions
  n_region_ = 1;
  for (int64_t i = 0; i < n_dim_; i++) {
    n_region_ *= geometry_[i]->n_mesh();
  }

  //----------------------------------------------------------------------------
  // Initialize calculation information from settings namespace
  //----------------------------------------------------------------------------
  n_order_ = 1;
  n_source_region_ = n_region_ * ng_;
  n_source_region_high_order_ = n_source_region_ * n_order_ * n_order_;

  // Resize lower and upper size to n_dim_
  lower_.resize(n_dim_);
  upper_.resize(n_dim_);
  for (int64_t i = 0; i < n_dim_; ++i) {
    lower_[i] = geometry_[i]->mesh_edges()[0];
    upper_[i] = geometry_[i]->mesh_edges()[geometry_[i]->n_mesh()];
  }

  // Quick treatment for a pure mesh for MGMC tally
  if (is_mesh) {
    // Only need:
    // new_scalar_flux_
    // final_scalar_flux_
    // final_scalar_flux_sq_
    // volume_analytic_ (not normalized)

    //================================================================================
    // For MGMC simulation, we don't consider higher-order flux
    //================================================================================
    // Set n_order_ to 1 for MGMC meshes
    n_order_ = 1;
    n_source_region_high_order_ = n_source_region_;

    new_scalar_flux_.assign(n_source_region_, 0.0);
    final_scalar_flux_.assign(n_source_region_, 0.0);
    final_scalar_flux_sq_.assign(n_source_region_, 0.0);
    volume_analytic_.assign(n_region_, 0.0);

    // Set stride_
    stride_ = {1, 1, 1};

    int64_t y_stride;
    int64_t z_stride;
    switch (n_dim_) {
    case 3:
      z_stride = geometry_[0]->n_mesh() * geometry_[1]->n_mesh();
      y_stride = geometry_[0]->n_mesh();
      stride_[1] = y_stride;
      stride_[2] = z_stride;
      break;

    case 2:
      y_stride = geometry_[0]->n_mesh();
      stride_[1] = y_stride;
      break;

    case 1:
      break;

    default:
      fatal_error("FlatSourceDomain: n_dim_ must be 1, 2, or 3");
    }

    // Calculate volume
    for (int64_t i = 0; i < n_region_; i++) {
      double volume = 1.0;
      int64_t remainder = i;

      for (int64_t d = n_dim_ - 1; d >= 0; --d) {
        int64_t idx = remainder / stride_[d];
        remainder = remainder % stride_[d];

        double dd =
          geometry_[d]->mesh_edges(idx + 1) - geometry_[d]->mesh_edges(idx);

        volume *= dd;
      }

      volume_analytic_[i] = volume;
    }



    // Quick return
    return;
  }

  // Allocate memory
  new_scalar_flux_.assign(n_source_region_high_order_, 0.0);
  old_scalar_flux_.assign(n_source_region_high_order_, 1.0);
  final_scalar_flux_.assign(n_source_region_high_order_, 0.0);
  final_scalar_flux_sq_.assign(n_source_region_high_order_, 0.0);

  source_.assign(n_source_region_high_order_, 0.0);
  volume_analytic_.assign(n_region_, 0.0);
  volume_estimated_.assign(n_region_, 0.0);
  volume_naive_.assign(n_region_, 0.0);
  volume_.assign(n_region_, 0.0);
  n_tracks_.assign(n_region_, 0);
  whether_hit_.assign(n_region_, false);
  materials_.assign(n_region_, 0);
  has_external_source_.assign(n_region_, false);



  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    fill_values(old_scalar_flux_, 0.0);
    external_source_.assign(n_source_region_, 0.0);
  }

  //----------------------------------------------------------------------------
  // Initialize materials layout information from model->materials node
  //----------------------------------------------------------------------------
  // Read materials map id
  std::vector<int64_t> materials;
  if (check_for_node(node, "materials")) {
    materials = get_node_array<int64_t>(node, "materials");
  } else {
    fatal_error("No materials in input.xml!");
  }

  // Calculate the number of materials maps
  int64_t n_mat_region_ = 1;
  for (int64_t i = 0; i < n_dim_; i++) {
    n_mat_region_ *= geometry_[i]->n_mat();
  }

  // Check material number
  if (materials.size() != n_mat_region_) {
    fatal_error("Incomplete materials in input.xml!");
  }

  int64_t x_div = geometry_[0]->n_mat();
  int64_t y_div = 1;
  int64_t z_div = 1;
  if (n_dim_ >= 2) {
    y_div = geometry_[1]->n_mat();
  }
  if (n_dim_ >= 3) {
    z_div = geometry_[2]->n_mat();
  }
  // Convert materials map id to 3D vector
  auto materials_in = convert_to_3d(materials, x_div, y_div, z_div);

  // Set stride_
  stride_ = {1, 1, 1};

  // Assign materials in all region and calculate the total volume
  total_volume_ = 0.0;
  MeshIndex mat_index = {0, 0, 0};
  int64_t y_stride;
  int64_t z_stride;

  switch (n_dim_) {
  case 3:
    z_stride = geometry_[0]->n_mesh() * geometry_[1]->n_mesh();
    y_stride = geometry_[0]->n_mesh();
    stride_[1] = y_stride;
    stride_[2] = z_stride;
    for (int64_t i = 0; i < geometry_[0]->n_mesh(); i++) {
      mat_index[0] = geometry_[0]->maps(i);
      for (int64_t j = 0; j < geometry_[1]->n_mesh(); j++) {
        mat_index[1] = geometry_[1]->maps(j);
        for (int64_t k = 0; k < geometry_[2]->n_mesh(); k++) {
          mat_index[2] = geometry_[2]->maps(k);

          // Assign material
          int64_t id = materials_in[mat_index[2]][mat_index[1]][mat_index[0]];
          materials_[k * z_stride + j * y_stride + i] =
            simulation::mat_map.at(id); // Return internal index

          // Calculate volume
          double dx =
            geometry_[0]->mesh_edges(i + 1) - geometry_[0]->mesh_edges(i);
          double dy =
            geometry_[1]->mesh_edges(j + 1) - geometry_[1]->mesh_edges(j);
          double dz =
            geometry_[2]->mesh_edges(k + 1) - geometry_[2]->mesh_edges(k);
          volume_analytic_[k * z_stride + j * y_stride + i] = dx * dy * dz;
          total_volume_ += volume_analytic_[k * z_stride + j * y_stride + i];
        }
      }
    }
    break;

  case 2:
    y_stride = geometry_[0]->n_mesh();
    stride_[1] = y_stride;
    for (int64_t i = 0; i < geometry_[0]->n_mesh(); i++) {
      mat_index[0] = geometry_[0]->maps(i);
      for (int64_t j = 0; j < geometry_[1]->n_mesh(); j++) {
        mat_index[1] = geometry_[1]->maps(j);

        // Assign material
        int64_t id = materials_in[0][mat_index[1]][mat_index[0]];
        materials_[j * y_stride + i] = simulation::mat_map.at(id);

        // Calculate volume
        double dx =
          geometry_[0]->mesh_edges(i + 1) - geometry_[0]->mesh_edges(i);
        double dy =
          geometry_[1]->mesh_edges(j + 1) - geometry_[1]->mesh_edges(j);
        volume_analytic_[j * y_stride + i] = dx * dy;
        total_volume_ += volume_analytic_[j * y_stride + i];
      }
    }
    break;

  case 1:
    for (int64_t i = 0; i < geometry_[0]->n_mesh(); i++) {
      // Assign material
      mat_index[0] = geometry_[0]->maps(i);
      int64_t id = materials_in[0][0][mat_index[0]];
      materials_[i] = simulation::mat_map.at(id);

      double dx = geometry_[0]->mesh_edges(i + 1) - geometry_[0]->mesh_edges(i);
      volume_analytic_[i] = dx;
      total_volume_ += volume_analytic_[i];
    }
    break;

  default:
    fatal_error("FlatSourceDomain: n_dim_ must be 1, 2, or 3");
  }

  // Normalize volume_analytic
  for (int64_t i = 0; i < n_region_; i++) {
    volume_analytic_[i] /= total_volume_;
  }
}

//==============================================================================
// FlatSourceDomain Constructor for meshsource
//==============================================================================
FlatSourceDomain::FlatSourceDomain(
  pugi::xml_node meshsource_node)
{
  ng_ = 1;

  //----------------------------------------------------------------------------
  // Initialize geometry information from source->space node
  //----------------------------------------------------------------------------
  // Read x division
  if (check_for_node(meshsource_node, "x")) {
    // Get x model
    pugi::xml_node x_node = meshsource_node.child("x");

    // Generate x division
    geometry_.push_back(new Division(x_node));
  } else {
    fatal_error("```x``` node must be provided in input.xml!");
  }

  // Read y division (optional)
  if (check_for_node(meshsource_node, "y")) {
    // Get y model
    pugi::xml_node y_node = meshsource_node.child("y");

    // Generate y division
    geometry_.push_back(new Division(y_node));
  }

  // Read z division (optional)
  if (check_for_node(meshsource_node, "z")) {
    // Get z model
    pugi::xml_node z_node = meshsource_node.child("z");

    // Generate z division
    geometry_.push_back(new Division(z_node));
  }

  // Get the number of dimensions about the meshsource
  n_dim_ = geometry_.size();

  // Get the number of meshsource regions
  n_region_ = 1;
  for (int64_t i = 0; i < n_dim_; i++) {
    n_region_ *= geometry_[i]->n_mesh();
  }

  //----------------------------------------------------------------------------
  // Initialize calculation information from settings namespace
  //----------------------------------------------------------------------------
  n_order_ = 1; // Set n_order to 1 for to support meshsource
  n_source_region_ = n_region_ * ng_;
  n_source_region_high_order_ = n_source_region_ * n_order_ * n_order_;

  // Resize lower and upper size to n_dim_
  lower_.resize(n_dim_);
  upper_.resize(n_dim_);
  for (int64_t i = 0; i < n_dim_; i++) {
    lower_[i] = geometry_[i]->mesh_edges()[0];
    upper_[i] = geometry_[i]->mesh_edges()[geometry_[i]->n_mesh()];
  }

  // Set stride
  stride_ = {1, 1, 1};

  switch (n_dim_) {
  case 3:
    stride_[1] = geometry_[0]->n_mesh();
    stride_[2] = geometry_[0]->n_mesh() * geometry_[1]->n_mesh();
    break;

  case 2:
    stride_[1] = geometry_[0]->n_mesh();
    break;

  case 1:
    break;

  default:
    fatal_error("FlatSourceDomain: n_dim_ must be 1, 2, or 3");
  }

  // Read distribution
  std::vector<double> distribution;
  if (check_for_node(meshsource_node, "distribution")) {
    distribution = get_node_array<double>(meshsource_node, "distribution");
  } else {
    fatal_error("No distribution in MeshSource!");
  }

  // Check distribution number
  if (distribution.size() != n_region_) {
    fatal_error("Length of distribution should == number of regions!");
  }

  // Normalization is skipped

  // Allocate memory
  source_.assign(distribution.begin(),
    distribution.end()); // Set source space distribution


}


//==============================================================================
// Member functions for Methods
//==============================================================================
//! Convert vector coordinate to Position format
Position FlatSourceDomain::lower_left() const
{
  Position p {0.0, 0.0, 0.0};
  for (int64_t i = 0; i < n_dim_; i++) {
    p[i] = lower_[i];
  }
  return p;
}

Position FlatSourceDomain::upper_right() const
{
  Position p {0.0, 0.0, 0.0};
  for (int64_t i = 0; i < n_dim_; i++) {
    p[i] = upper_[i];
  }
  return p;
}

int64_t FlatSourceDomain::material(MeshIndex& index) const
{
  int64_t idx = 0;
  for (int64_t i = 0; i < n_dim_; i++) {
    idx += index[i] * stride_[i];
  }
  // TODO: Check
  return materials_[idx];
}

int64_t FlatSourceDomain::material(int64_t i) const
{
  return materials_[i];
}

int64_t FlatSourceDomain::flatten(MeshIndex& index) const
{
  int64_t idx = 0;
  for (int64_t i = 0; i < n_dim_; i++) {
    idx += index[i] * stride_[i];
  }
  return idx;
}

int64_t FlatSourceDomain::flatten(std::vector<int64_t>& index) const
{
  int64_t idx = 0;
  for (int64_t i = 0; i < n_dim_; i++) {
    idx += index[i] * stride_[i];
  }
  return idx;
}

//! Check if mesh index belongs to full domain
bool FlatSourceDomain::if_valid(MeshIndex& index)
{
  bool valid = true;
  for (int64_t i = 0; i < n_dim_; i++) {
    if (index[i] < 0 || index[i] >= get_n_mesh(i)) {
      valid = false;
      break;
    }
  }
  return valid;
}

bool FlatSourceDomain::if_valid(std::vector<int64_t>& index)
{
  bool valid = true;
  for (int64_t i = 0; i < n_dim_; i++) {
    if (index[i] < 0 || index[i] >= get_n_mesh(i)) {
      valid = false;
      break;
    }
  }
  return valid;
}

//! Get 3D mesh index from positon (strict check version)
// Find the mesh index for a given r.
// Throws std::out_of_range if any dim is outside the mesh domain.
MeshIndex FlatSourceDomain::get_index(Position r) const
{
  MeshIndex index = {0, 0, 0};
  for (int64_t i = 0; i < n_dim_; i++) {
    index[i] = geometry_[i]->get_index(r[i]);
  }
  return index;
}

//! Get 3D mesh index from position (not strict check version)
// If x is outside the mesh domain, sets in_mesh to false and returns the
// nearest valid index.
void FlatSourceDomain::get_mesh_index(
  Position r, MeshIndex& index, bool& in_mesh)
{
  in_mesh = true;
  index = {0, 0, 0};

  for (int64_t i = 0; i < n_dim_; i++) {
    index[i] = geometry_[i]->get_mesh_index(r[i], in_mesh);
  }
}

//! Set source for fsr domain
void FlatSourceDomain::set_fixed_source()
{
  if (simulation::sources.size() == 0)
    return;

  for (int64_t i_src = 0; i_src < simulation::sources.size(); i_src++) {
    simulation::sources[i_src]->set_source_to_FSR();
  }

// Divide by total XS
  for (int64_t i = 0; i < n_region_; i++) {
    if (has_external_source_[i]) {
      int64_t i_src = i * ng_;
      for (int64_t g = 0; g < ng_; g++) {
        external_source_[i_src + g] /=
          simulation::materials[material(i)]->sigma_t[g];
      }
    }
  }
}

//! Check if source mesh overlaps with the fsr mesh
// d: dimension to check, x0: lower bound, x1: upper bound
// if not overlap, i0 and i1 will correspond to the closest mesh index (lower
// boundary)
void FlatSourceDomain::overlap(
  int64_t d, double x0, double x1, int64_t& i0, int64_t& i1, bool& overlap)
{
  // Quick return
  if (d > n_dim_) {
    fatal_error("Dimension is too high for overlap check!");
  }

  // Set default value
  overlap = true;

  int64_t n = geometry_[d]->n_mesh();

  // Quick return when out of range
  if (x0 > geometry_[d]->mesh_edges(n) || x1 < geometry_[d]->mesh_edges(0)) {
    overlap = false;
    return;
  }

  // Get index
  i0 = closest(geometry_[d]->mesh_edges(), x0);
  i1 = closest(geometry_[d]->mesh_edges(), x1);

  return;
}

//==============================================================================
// For free memory
//==============================================================================
//! Free geometry memory
void FlatSourceDomain::free_geometry()
{
  // [MODIFIED] Set pointer to null to prevent double free in destructor
  for (auto& g : geometry_) {
    if (g != nullptr) {
      delete g;
      g = nullptr;
    }
  }
  geometry_.clear();


}

} // namespace theta
