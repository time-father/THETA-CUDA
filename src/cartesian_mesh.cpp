#include "cartesian_mesh.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "error.h"
#include "flat_source_domain.h"
#include "material.h"
#include "math_func.h"
#include "memory.h"
#include "position.h"
#include "random_ray_simulation.h"
#include "settings.h"
#include "simulation.h"
#include "source.h"
#include "xml_interface.h"
#include "xs/scatt_data.h"

namespace theta {


//==============================================================================
// Division constructor from node
//==============================================================================
Division::Division(pugi::xml_node node)
{
  // Read map edges from edges node
  std::vector<double> edges;
  if (check_for_node(node, "edges")) {
    edges = get_node_array<double>(node, "edges");
  } else if (check_for_node(node, "width")) {
    // Define by width of each division
    auto width = get_node_array<double>(node, "width");
    edges.push_back(0.0);
    for (int64_t i = 0; i < width.size(); ++i) {
      edges.push_back(edges[i] + width[i]);
    }
  } else {
    fatal_error("Incomplete edges in input.xml!");
  }

  // Read resolutions from resolutions node
  std::vector<int64_t> resolutions;
  if (check_for_node(node, "resolutions")) {
    resolutions = get_node_array<int64_t>(node, "resolutions");
    if (resolutions.size() != edges.size() - 1) {
      fatal_error("Inconsistent resolutions in input.xml!");
    }
  } else {
    // Default to one mesh for each division
    for (int64_t i = 0; i < edges.size() - 1; ++i) {
      resolutions.push_back(1);
    }
  }

  // Calculate the number of mesh, map edges offset and mesh edges and
  // corresponding map id
  n_mesh_ = 0;
  offsets_.push_back(0);
  for (int64_t i = 0; i < resolutions.size(); ++i) {
    // Set interval
    int64_t res = resolutions[i];
    double interval = (edges[i + 1] - edges[i]) / res;
    // Add edges
    for (int64_t j = 0; j < res; ++j) {
      mesh_edges_.push_back(edges[i] + j * interval);
      maps_.push_back(i);
    }
    n_mesh_ += res;
    offsets_.push_back(n_mesh_);
  }
  // Add last edge
  mesh_edges_.push_back(edges[edges.size() - 1]);

  // Calculate mesh centers
  for (int64_t i = 0; i < n_mesh_; ++i) {
    double center = (mesh_edges_[i] + mesh_edges_[i + 1]) / 2.0;
    mesh_center_.push_back(center);
  }

  // Get number of materials map
  n_mat_ = edges.size() - 1;

  // Get boundary conditions
  if (check_for_node(node, "boundary")) {
    auto boundary = get_node_array<int64_t>(node, "boundary");
    lower_boundary_type_ = static_cast<BoundaryType>(boundary[0]);
    upper_boundary_type_ = static_cast<BoundaryType>(boundary[1]);
  } else {
    lower_boundary_type_ = BoundaryType::REFLECTIVE;
    upper_boundary_type_ = BoundaryType::REFLECTIVE;
    warning("Incomplete boundary in input.xml! (Neglect this warning in MGMC "
            "mode)");
  }
}

//==============================================================================
// Division constructor from other Division object
//==============================================================================
Division::Division(const Division* from)
{
  this->n_mesh_ = from->n_mesh_;
  this->n_mat_ = from->n_mat_;
  this->mesh_edges_ = from->mesh_edges_;
  this->mesh_center_ = from->mesh_center_;
  this->maps_ = from->maps_;
  this->offsets_ = from->offsets_;
  this->lower_boundary_type_ = from->lower_boundary_type_;
  this->upper_boundary_type_ = from->upper_boundary_type_;
}

//==============================================================================
// Division Member functions for Methods
//==============================================================================
//! Find the mesh index for a given position x.
//! Throws std::out_of_range if x is outside the mesh domain.
int64_t Division::get_index(double x) const
{
  int64_t i = lower_bound_index(mesh_edges_.begin(), mesh_edges_.end(), x);
  if (i < 0 || i >= n_mesh_) {
    fatal_error("Ray is outside of the geometry.");
  }
  return i;
}

//! Find the mesh index for a given position x.
// If x is outside the mesh domain, sets in_mesh to false and returns the
// nearest valid index.
int64_t Division::get_mesh_index(double x, bool& in_mesh) const
{
  int64_t i = lower_bound_index(mesh_edges_.begin(), mesh_edges_.end(), x);
  if (i < 0 || i >= n_mesh_) {
    in_mesh = false;
    // Set to a valid index
    i = std::max(i, static_cast<int64_t>(0));
    i = std::min(i, n_mesh_ - 1);
  }
  return i;
}

void read_model(pugi::xml_node node)
{

  // Allocate geometry
  simulation::model =
    std::make_unique<FlatSourceDomain>(settings::num_of_groups, node, false);

  // Set lower_left and upper_right
  for (int64_t d = 0; d < simulation::model->n_dim(); ++d) {
    simulation::lower_left_full[d] = simulation::model->lower_left()[d];
    simulation::upper_right_full[d] = simulation::model->upper_right()[d];
  }
}

void free_memory_model()
{
  simulation::model->free_geometry();
}

} // namespace theta
