#ifndef THETA_MATERIAL_H
#define THETA_MATERIAL_H

#include <cstdint>
#include <vector>

#include "pugixml.hpp"
#include "xs/scatt_data.h"

namespace theta {

class Material {
public:
  int64_t mat_id {0};
  std::vector<double> sigma_t;
  std::vector<double> nu_sigma_f;
  ScatterXs sigma_s_high_order;
  std::vector<double> chi;
  bool fissionable {false};

  explicit Material(pugi::xml_node mat_node);
};

void free_memory_material();

void read_materials(pugi::xml_node node);

} // namespace theta

#endif // THETA_MATERIAL_H
