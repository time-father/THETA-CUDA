#include "xml_interface.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "error.h"
#include "utils/string_utils.h"

namespace theta {

std::string get_node_value(
  pugi::xml_node node, const char* name, bool lowercase, bool strip)
{
  // Search for either an attribute or child tag and get the data as a char*.
  const pugi::char_t* value_char = nullptr;
  if (node.attribute(name)) {
    value_char = node.attribute(name).value();
  } else if (node.child(name)) {
    value_char = node.child_value(name);
  } else {
    std::ostringstream oss;
    oss << "Node \"" << name << "\" is not a member of the \"" << node.name()
        << "\" XML node";
    fatal_error(oss.str());
  }
  std::string value {value_char};

  // Convert to lower-case if needed
  if (lowercase)
    to_lower(value);

  // Strip leading/trailing whitespace if needed
  if (strip) {
    value.erase(0, value.find_first_not_of(" \t\r\n\f\v"));
    value.erase(value.find_last_not_of(" \t\r\n\f\v") + 1);
  }

  return value;
}

bool get_node_value_bool(pugi::xml_node node, const char* name)
{
  if (node.attribute(name)) {
    return node.attribute(name).as_bool();
  } else if (node.child(name)) {
    return node.child(name).text().as_bool();
  } else {
    std::ostringstream oss;
    oss << "Node \"" << name << "\" is not a member of the \"" << node.name()
        << "\" XML node";
    fatal_error(oss.str());
  }
}

Position get_node_position(
  pugi::xml_node node, const char* name, bool lowercase)
{
  std::vector<double> arr = get_node_array<double>(node, name, lowercase);
  return Position(arr);
}

} // namespace theta
