#ifndef THETA_SOURCE_H
#define THETA_SOURCE_H

#include <memory>
#include <vector>

#include "flat_source_domain.h"
#include "pugixml.hpp"

namespace theta {

class SourceInSpace {
public:
  virtual ~SourceInSpace() = default;
  static std::unique_ptr<SourceInSpace> create(pugi::xml_node node);

  virtual void set_source_to_FSR(double strength) = 0;
};

class MeshSource : public SourceInSpace {
public:
  explicit MeshSource(pugi::xml_node node);
  void set_source_to_FSR(double strength) override;

private:
  std::unique_ptr<FlatSourceDomain> mesh_;
};

class EnergySource {
public:
  virtual ~EnergySource() = default;
  static std::unique_ptr<EnergySource> create(pugi::xml_node node);

  virtual void multiply_source_to_FSR() = 0;
};

class EnergyGroupSource : public EnergySource {
public:
  explicit EnergyGroupSource(pugi::xml_node node);
  void multiply_source_to_FSR() override;

private:
  std::vector<double> distribution_;
};

class Source {
public:
  explicit Source(pugi::xml_node node);
  void set_source_to_FSR();

private:
  double strength_;
  std::unique_ptr<SourceInSpace> space_;
  std::unique_ptr<EnergySource> energy_;
};

void read_sources(pugi::xml_node node);
void free_memory_source();

} // namespace theta

#endif // THETA_SOURCE_H
