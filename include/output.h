#ifndef THETA_OUTPUT_H
#define THETA_OUTPUT_H

#include <string>

#include "pugixml.hpp"

#include "xml_interface.h"

namespace theta {

class Output {
public:
  std::string flux = "flux.csv";
  std::string flux_std = "flux_std.csv";
  std::string run_info = "run_info.log";
  std::string batch_info = "batch_info.csv";
  std::string exce_time = "time.csv";
  std::string geom = "";
  std::string vtk = "";
  std::string mat = "";
  std::string binary = "true";
  std::string bin_precision = "double";

  Output();
  ~Output() = default;
  void read_output(pugi::xml_node node);
};

// Display the main title banner as well as information about the
// program developers, version, and date/time which the problem was run.
void theta_title();

// Retrieve a time stamp.
//
// return current time stamp (format: "yyyy-mm-dd hh:mm:ss")
std::string time_stamp();

std::string append_flux_suffix(const std::string& name, const std::string& suffix);

} // namespace theta

#endif // THETA_OUTPUT_H
