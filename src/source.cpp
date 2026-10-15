#include "source.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "error.h"
#include "flat_source_domain.h"
#include "settings.h"
#include "simulation.h"
#include "xml_interface.h"

namespace theta {

//==============================================================================
// Class implementation: Base class for sources in space
//==============================================================================
std::unique_ptr<SourceInSpace> SourceInSpace::create(pugi::xml_node node)
{
  if (check_for_node(node, "type") &&
      get_node_value(node, "type", true, true) == "mesh") {
    return std::make_unique<MeshSource>(node);
  }
  fatal_error("Only mesh spatial sources are supported.");
}

//==============================================================================
// Class implementation: Mesh source in space
//==============================================================================
MeshSource::MeshSource(pugi::xml_node node)
  : mesh_(std::make_unique<FlatSourceDomain>(node))
{
  if (mesh_->n_dim() != simulation::model->n_dim()) {
    fatal_error("Dimension of MeshSource should == dimension of model!");
  }
}

void MeshSource::set_source_to_FSR(double strength)
{
  int64_t i0;
  int64_t i1;
  bool overlap;

  // Allocate vectors to store boundary data
  std::vector<std::vector<double>> bdy;
  bdy.resize(mesh_->n_dim());
  for (auto& bd : bdy) {
    bd.resize(2);
  }
  std::vector<std::vector<int64_t>> fsr;
  fsr.resize(3);
  for (auto& f : fsr) {
    f.resize(2);
  }

  // Flag to determine whether to skip a FSRs
  std::vector<bool> filled;
  filled.assign(simulation::model->n_region(), false);

  bool skip = false;
  for (int64_t i = 0; i < mesh_->n_region(); i++) {
    // Set all FSRs included by a single bin in the mesh

    // Get bdy of this mesh bin
    int64_t remainder = i;
    for (int64_t d = mesh_->n_dim() - 1; d >= 0; --d) {
      int64_t idx = remainder / mesh_->stride(d);
      remainder = remainder % mesh_->stride(d);

      bdy[d][0] = mesh_->get_edge(d, idx);
      bdy[d][1] = mesh_->get_edge(d, idx + 1);
    }

    // Determine the FSRs included
    skip = false;
    for (int64_t d = 0; d < 3; d++) {
      fsr[d][0] = 0;
      fsr[d][1] = 0;
    }

    // Judge if current meshsource bin overlaps with total fsr geometry
    for (int64_t d = 0; d < mesh_->n_dim(); d++) {
      // i0: fsr mesh left boundary corresponding to bdy[d][0]
      // i1: fsr mesh right boundary corresponding to bdy[d][1]
      simulation::model->overlap(d, bdy[d][0], bdy[d][1], i0, i1, overlap);

      if (overlap) {
        fsr[d][0] = i0;
        fsr[d][1] = std::max(i0, i1 - 1);
      } else {
        skip = true;
        break;
      }
    }

    // No FSR overlapping with this bin
    if (skip) {
      continue;
    }

    // Set FSRs overlapping with this bin
    int64_t zz = 0;
    int64_t yy = 0;
    for (int64_t iz = fsr[2][0]; iz <= fsr[2][1]; iz++) {
      zz = iz * simulation::model->stride(2);
      for (int64_t iy = fsr[1][0]; iy <= fsr[1][1]; iy++) {
        yy = iy * simulation::model->stride(1);
        for (int64_t ix = fsr[0][0]; ix <= fsr[0][1]; ix++) {
          // Set idx
          int64_t idx = zz + yy + ix;

          // Skip if already set
          if (filled[idx])
            continue;

          // Get strength
          double src = mesh_->source_[i];

          // Set flag
          simulation::model->has_external_source_[idx] = true;
          filled[idx] = true;

          // Set source
          int64_t i_src = idx * simulation::model->ng();
          for (int64_t g = 0; g < simulation::model->ng(); g++) {
            simulation::model->external_source_[i_src + g] += src * strength;
          }
        }
      }
    }
  }
}

//==============================================================================
// Class implementation: Energy source
//==============================================================================
std::unique_ptr<EnergySource> EnergySource::create(pugi::xml_node node)
{
  if (check_for_node(node, "type")) {
    if (get_node_value(node, "type", true, true) == "group") {
      return std::make_unique<EnergyGroupSource>(node);
    } else {
      fatal_error("Only group energy distributions are supported.");
    }
  } else {
    fatal_error("Energy source type is required.");
  }
}

//==============================================================================
// Class implementation: Energy group source
//==============================================================================
EnergyGroupSource::EnergyGroupSource(pugi::xml_node node)
{
  // Read groups
  if (!check_for_node(node, "groups")) {
    fatal_error("Energy group indices are required for mesh sources.");
  }
  const std::vector<int64_t> groups_in =
    get_node_array<int64_t>(node, "groups");

  // Read distribution
  std::vector<double> distribution_in;
  if (check_for_node(node, "distribution")) {
    distribution_in = get_node_array<double>(node, "distribution");
  } else {
    fatal_error("```distribution``` missed in energy of source!");
  }

  // Compare length
  if (groups_in.size() != distribution_in.size()) {
    fatal_error(
      "Length of groups and distribution in energy of space is unequal!");
  }

  distribution_.resize(settings::num_of_groups, 0.0);
  for (int64_t g = 0; g < groups_in.size(); g++) {
    distribution_[groups_in[g]] = distribution_in[g];
  }
}

//! Multiply the space source by energy distribution
void EnergyGroupSource::multiply_source_to_FSR()
{
  // Multiply by energy distribution
  for (int64_t i = 0; i < simulation::model->n_region(); i++) {
    if (simulation::model->has_external_source_[i]) {
      int64_t i_src = i * simulation::model->ng();
      for (int64_t g = 0; g < simulation::model->ng(); g++) {
        simulation::model->external_source_[i_src + g] *= distribution_[g];
      }
    }
  }
}

//==============================================================================
// Class implementation: Source for simulation
//==============================================================================
Source::Source(pugi::xml_node node)
{
  // Set source strength
  if (check_for_node(node, "strength")) {
    strength_ = std::stod(get_node_value(node, "strength"));
    if (strength_ < 0.0) {
      fatal_error("Source strength is negative.");
    }
  } else {
    fatal_error("Source strength should be provided.");
  }

  // Readin space
  if (check_for_node(node, "space")) {
    space_ = SourceInSpace::create(node.child("space"));
  } else {
    fatal_error("```space``` node does not exist in source!");
  }

  if (!check_for_node(node, "energy")) {
    fatal_error("Energy distribution is required for mesh sources.");
  }
  energy_ = EnergySource::create(node.child("energy"));
}

void Source::set_source_to_FSR()
{
  // Set space source
  this->space_->set_source_to_FSR(this->strength_);

  // Multiply by energy distribution
  this->energy_->multiply_source_to_FSR();
}

//==============================================================================
// Non-member functions
//==============================================================================
//! Read sources from XML node
void read_sources(pugi::xml_node node)
{
  for (pugi::xml_node src_node : node.children("source")) {
    simulation::sources.push_back(std::make_unique<Source>(src_node));
  }
}

//! Free memory allocated for sources
void free_memory_source()
{
  simulation::sources.clear();
}
} // namespace theta
