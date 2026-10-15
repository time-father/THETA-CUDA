#include "input.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include "pugixml.hpp"

#include "error.h"
#include "material.h"
#include "memory.h"
#include "output.h"
#include "settings.h"
#include "simulation.h"
#include "source.h"
#include "xml_interface.h"

namespace theta {

void parse_command_line(int argc, char* argv[])
{
  int i = 1;
  while (i < argc) {
    std::string arg {argv[i]};
    if (arg[0] == '-') {

      if (arg == "-g" || arg == "--gpu") {
        if (++i >= argc) {
          fatal_error("GPU id not specified");
        }
        settings::gpu_id = std::stoi(argv[i]);
        if (settings::gpu_id < 0) {
          fatal_error("GPU id must be >= 0");
        }
      } else if (arg == "-p" || arg == "--plot") {
        settings::run_mode = RunMode::PLOT;
      } else {
        fatal_error("Unknown command-line option: " + arg);
      }

    } else {
      // Get filename
      settings::input_name = argv[i];

      // Check if this file exists
      if (!file_exists(settings::input_name)) {
        fatal_error("Input file " + settings::input_name + " does not exist!");
      }
    }
    ++i;
  }
}

void read_input(std::string input_name)
{
  if (!file_exists(input_name)) {
    fatal_error("Input file " + input_name + " does not exist!");
  }

  // Parse settings.xml file
  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_file(input_name.c_str());
  if (!result) {
    fatal_error(
      "Error processing input" + input_name + " file: " + result.description());
  }

  // Get root element
  pugi::xml_node root = doc.document_element();

  // Read output
  if (check_for_node(root, "output")) {
    // Get output node
    pugi::xml_node output_node = root.child("output");
    simulation::out_file.read_output(output_node);

  } else {
    // fatal_error("```output``` node does not exist in input.xml!");
  }

  // Read settings
  if (check_for_node(root, "settings")) {
    // Get settings node
    pugi::xml_node settings_node = root.child("settings");

    // Read settings
    read_settings(settings_node);
  } else {
    fatal_error("```settings``` node does not exist in input.xml!");
  }

  // Read materials
  if (check_for_node(root, "materials")) {
    // Get material node
    pugi::xml_node materials_node = root.child("materials");

    // Allocate material
    read_materials(materials_node);
  } else {
    fatal_error("```materials``` node does not exist in input.xml!");
  }

  // Read model
  if (check_for_node(root, "model")) {
    // Get model node
    pugi::xml_node model_node = root.child("model");

    // Allocate geometry
    // Note: This constructs the mesh and initializes
    // simulation::model->n_region()
    read_model(model_node);
  } else {
    fatal_error("```model``` node does not exist in input.xml!");
  }

  // Read sources
  if (check_for_node(root, "sources")) {
    // Get sources node
    pugi::xml_node sources_node = root.child("sources");

    // Allocate sources
    read_sources(sources_node);
  }


}

bool dir_exists(const std::string& path)
{
  struct stat s;
  if (stat(path.c_str(), &s) != 0) {
    return false;
  }
  return s.st_mode & S_IFDIR;
}

bool file_exists(const std::string& filename)
{
  // rule out file being a path to a directory
  if (dir_exists(filename))
    return false;

  std::ifstream s {filename};
  return s.good();
}

} // namespace theta
