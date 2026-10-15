#ifndef THETA_FLAT_SOURCE_DOMAIN_H
#define THETA_FLAT_SOURCE_DOMAIN_H

#include <cstdint>
#include <vector>

#include "cartesian_mesh.h"
#include "memory.h"

#include "pugixml.hpp"

namespace theta {

/*
 * The FlatSourceDomain class encompasses data and methods for storing
 * scalar flux and source region for all flat source regions in a
 * random ray simulation domain.
 */

class FlatSourceDomain {
public:
  //==============================================================================
  // Constructors and destructor
  //==============================================================================
  // Default constructor
  FlatSourceDomain() = default;
  FlatSourceDomain(
    const int64_t& num_of_groups, pugi::xml_node node, bool is_mesh = false);
  FlatSourceDomain(pugi::xml_node meshsource_node);
  virtual ~FlatSourceDomain();



  //==============================================================================
  // Member functions for Methods
  //==============================================================================


  //! Convert vector coordinate to Position format
  Position lower_left() const;
  Position upper_right() const;

  //! Get material inner index from mesh index.
  int64_t material(MeshIndex& index) const;

  //! Get material inner index from a one-dimensional index.
  int64_t material(int64_t i) const;

  //! Get all materials inner index
  std::vector<int64_t> materials() const { return materials_; }

  //! Flatten 3D mesh index to 1D index
  int64_t flatten(std::vector<int64_t>& index) const;
  int64_t flatten(MeshIndex& index) const;

  //! Check if mesh index belongs to full domain
  bool if_valid(MeshIndex& index);
  bool if_valid(std::vector<int64_t>& index);

  //! Get 3D mesh index from positon (strict check version)
  // Find the mesh index for a given r.
  // Throws std::out_of_range if any dim is outside the mesh domain.
  MeshIndex get_index(Position r) const;

  //! Get 3D mesh index from position (not strict check version)
  // If x is outside the mesh domain, sets in_mesh to false and returns the
  // nearest valid index.
  void get_mesh_index(Position r, MeshIndex& index, bool& in_mesh);

  //! Set source for fsr domain
  void set_fixed_source();

  //! Check if source mesh overlaps with the frs mesh
  // d: dimension to check, x0: lower bound, x1: upper bound
  // if not overlap, i0 and i1 will correspond to the closest mesh index (lower
  // boundary)
  void overlap(
    int64_t d, double x0, double x1, int64_t& i0, int64_t& i1, bool& overlap);

  //==============================================================================
  // For free memory
  //==============================================================================
  //! Free geometry memory
  void free_geometry();

  //==============================================================================
  // Virtual functions for derived class (empty realization)
  //==============================================================================
  virtual const std::vector<Position>& mesh_centers() const
  {
    static std::vector<Position> empty;
    return empty;
  };
  //! reset some variables for each batch
  virtual void batch_reset() {}

  //==============================================================================
  // Member functions for Accessors and Setters
  //==============================================================================
  int64_t ng() const { return ng_; }
  int64_t n_dim() const { return n_dim_; }
  int64_t n_order() const { return n_order_; }
  int64_t n_region() const { return n_region_; }

  int64_t get_n_mesh(int64_t i) const { return geometry_[i]->n_mesh(); }
  int64_t find_mesh_index(int64_t i, double x) const
  {
    return geometry_[i]->get_index(x);
  }
  double get_mesh_center(int64_t i, int64_t j) const
  {
    return geometry_[i]->mesh_center(j);
  }

  int64_t& n_realizations() { return n_realizations_; }
  const int64_t& n_realizations() const { return n_realizations_; }
  double get_edge(int64_t i, int64_t j) const
  {
    return geometry_[i]->mesh_edges(j);
  }
  const Division* geometry(int64_t i) const { return geometry_[i]; }
  int64_t stride(int64_t d) const { return stride_[d]; }
  int64_t n_source_region() const { return n_source_region_; }
  int64_t n_source_region_high_order() const
  {
    return n_source_region_high_order_;
  }

  std::vector<double>& volume_analytic() { return volume_analytic_; }
  const std::vector<double>& volume_analytic() const
  {
    return volume_analytic_;
  }

  double& total_volume() { return total_volume_; }
  const double& total_volume() const { return total_volume_; }

  bool active() const { return active_; }
  void set_active(bool active) { active_ = active; }

  //==============================================================================
  // Member variables for public
  //==============================================================================
public:
  // scalar flux for the current iteration
  // dim: [num_region, l, m , num_groups]
  // length = num_region * (l+1)* (l+1) * num_gourps
  std::vector<double> new_scalar_flux_;      // new scalar flux
  std::vector<double> old_scalar_flux_;      // old scalar flux
  std::vector<double> final_scalar_flux_;    // final scalar flux
  std::vector<double> final_scalar_flux_sq_; // square of final flux

  std::vector<double>
    source_; // angular moments of the source (including: scatter source fission
             // source, and external source)
  std::vector<double>
    external_source_; // external source (nregion_, ng_) in fixed source mode
  std::vector<double> volume_analytic_;  // analytic volume of each region
  std::vector<double> volume_estimated_; // estimated volume of each region
  std::vector<double> volume_naive_;     // naive volume of each region
  std::vector<double> volume_;

  std::vector<bool>
    has_external_source_;     // Flag indicating if the region has
                              // external source, length = (nregion_, )
  std::vector<int> n_tracks_; // The number of hitting ray in each region within
                              // the active length
  std::vector<bool> whether_hit_; // Flag indicating if the region has been hit


  //==============================================================================
  // Member variables for protected
  //==============================================================================
protected:
  // Energy and Space information
  int64_t ng_;                         // number of energy groups
  int64_t n_dim_;                      // number of dimensions
  int64_t n_order_;                    // number of orders
  int64_t n_region_;                   // number of mesh regions
  int64_t n_source_region_;            // n_region_ x ng_
  int64_t n_source_region_high_order_; // n_region_ x ng_ x n_order_ x n_order_
  std::vector<Division*>
    geometry_; // each dimension Division object information, size=n_dim_
  std::vector<double> lower_; // The lower corner of current domain
  std::vector<double> upper_; // The upper corner of current domain

  // All fsr mesh materials layout information
  std::vector<int64_t> materials_; // inner material id for each region

  // Array process information
  std::array<int64_t, 3> stride_; // stride for accessing data in 3D mesh
  bool active_ {false};           // Flag indicating if the domain is active
  int64_t n_realizations_ {0};    // number of realizations for this domain
  double total_volume_ {0.0};     // total volume of the domain

}; // class FlatSourceDomain

} // namespace theta

#endif // THETA_FLAT_SOURCE_DOMAIN_H
