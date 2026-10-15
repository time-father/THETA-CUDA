#include "material.h"

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "error.h"
#include "settings.h"
#include "simulation.h"
#include "xml_interface.h"

namespace theta {

Material::Material(pugi::xml_node mat_node)
{
  if (check_for_node(mat_node, "id")) {
    mat_id = std::stoi(get_node_value(mat_node, "id"));
  } else {
    fatal_error("`id` node does not exist in input.xml!");
  }

  if (!check_for_node(mat_node, "sigma_t")) {
    fatal_error("`sigma_t` is required for material " + std::to_string(mat_id));
  }
  sigma_t = get_node_array<double>(mat_node, "sigma_t");

  if (!check_for_node(mat_node, "sigma_s")) {
    fatal_error("`sigma_s` is required for material " + std::to_string(mat_id));
  }
  const std::vector<double> scatter_raw =
    get_node_array<double>(mat_node, "sigma_s");
  const int64_t groups = settings::num_of_groups;
  const int64_t values_per_order = groups * groups;
  if (groups <= 0 || scatter_raw.size() < static_cast<size_t>(values_per_order)) {
    fatal_error("`sigma_s` must contain a zero-order group matrix.");
  }

  std::vector<std::vector<std::vector<double>>> scatter(
    groups, std::vector<std::vector<double>>(groups, std::vector<double>(1, 0.0)));
  for (int64_t incoming = 0; incoming < groups; ++incoming) {
    for (int64_t outgoing = 0; outgoing < groups; ++outgoing) {
      const size_t offset = static_cast<size_t>(incoming * groups + outgoing);
      scatter[incoming][outgoing][0] = scatter_raw[offset];
    }
  }
  sigma_s_high_order.from_3d_vector(scatter, groups, 1);

  const bool has_nu_fission = check_for_node(mat_node, "nu_sigma_f");
  const bool has_chi = check_for_node(mat_node, "chi");
  if (has_nu_fission != has_chi) {
    fatal_error("`nu_sigma_f` and `chi` must be provided together.");
  }
  if (has_nu_fission) {
    nu_sigma_f = get_node_array<double>(mat_node, "nu_sigma_f");
    chi = get_node_array<double>(mat_node, "chi");
    if (nu_sigma_f.size() != static_cast<size_t>(groups) ||
        chi.size() != static_cast<size_t>(groups)) {
      fatal_error("Fission cross sections must match the number of groups.");
    }
    const double chi_sum = std::accumulate(chi.begin(), chi.end(), 0.0);
    if (chi_sum <= 0.0) {
      fatal_error("The `chi` spectrum must have a positive sum.");
    }
    for (double& value : chi) {
      value /= chi_sum;
    }
    fissionable = true;
  } else {
    nu_sigma_f.assign(groups, 0.0);
    chi.assign(groups, 0.0);
    fissionable = false;
  }
}

void read_materials(pugi::xml_node node)
{
  if (check_for_node(node, "bugle_path")) {
    fatal_error(
      "BUGLE library input was removed; provide macro cross sections in each material.");
  }

  const auto material_nodes = node.children("material");
  simulation::n_materials =
    std::distance(material_nodes.begin(), material_nodes.end());
  if (simulation::n_materials == 0) {
    fatal_error("At least one `material` node is required.");
  }

  if (check_for_node(node, "ng")) {
    settings::num_of_groups = std::stoll(get_node_value(node, "ng"));
  } else {
    fatal_error("`ng` node does not exist in input.xml!");
  }

  simulation::materials.reserve(simulation::n_materials);
  for (pugi::xml_node mat_node : material_nodes) {
    if (mat_node.child("nuclide")) {
      fatal_error(
        "Nuclide density input was removed; provide macro cross sections directly.");
    }
    simulation::materials.push_back(std::make_unique<Material>(mat_node));
  }

  for (int64_t index = 0; index < simulation::n_materials; ++index) {
    const int64_t id = simulation::materials[index]->mat_id;
    if (!simulation::mat_map.emplace(id, index).second) {
      fatal_error("Duplicate material " + std::to_string(id) + " in input.xml!");
    }
  }
}

void free_memory_material()
{
  simulation::materials.clear();
  simulation::mat_map.clear();
}

} // namespace theta
