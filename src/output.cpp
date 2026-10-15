#include "output.h"

#include <ctime>   // for time, localtime
#include <iomanip> // for setw, setprecision, put_time
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif

#include "error.h"
#include "version.h"


namespace theta {

inline bool is_terminal_utf8()
{
#ifdef _WIN32
  return (GetConsoleOutputCP() == 65001);
#else
  const char* lang = std::getenv("LANG");
  if (lang != nullptr && std::string(lang).find("UTF-8") != std::string::npos) {
    return true;
  }
  const char* lc_all = std::getenv("LC_ALL");
  if (lc_all != nullptr &&
      std::string(lc_all).find("UTF-8") != std::string::npos) {
    return true;
  }
  const char* lc_ctype = std::getenv("LC_CTYPE");
  return (lc_ctype != nullptr &&
          std::string(lc_ctype).find("UTF-8") != std::string::npos);
#endif
}

// AUTHOR LJC
// Store the output file name
void Output::read_output(pugi::xml_node node)
{
  if (check_for_node(node, "mat")) {
    mat = get_node_value(node, "mat", false, true);
  }
  if (check_for_node(node, "binary")) {
    binary = get_node_value(node, "binary", true, true);
    pugi::xml_node binary_node = node.child("binary");
    if (binary_node && binary_node.attribute("precision")) {
      bin_precision = get_node_value(binary_node, "precision", true, true);
    }
    if (bin_precision != "float" && bin_precision != "double") {
      fatal_error("The binary precision attribute must be float or double.");
    }
  }
  if (check_for_node(node, "flux")) {
    flux = get_node_value(node, "flux", false, true);
  }
  if (check_for_node(node, "flux_std")) {
    flux_std = get_node_value(node, "flux_std", false, true);
  }
  if (check_for_node(node, "run_info")) {
    run_info = get_node_value(node, "run_info", false, true);
  }
  if (check_for_node(node, "batch_info")) {
    batch_info = get_node_value(node, "batch_info", false, true);
  }
  if (check_for_node(node, "time")) {
    exce_time = get_node_value(node, "time", false, true);
  }
  if (check_for_node(node, "vtk")) {
    vtk = get_node_value(node, "vtk", false, true);
  }
  if (check_for_node(node, "geom")) {
    geom = get_node_value(node, "geom", false, true);
  }
}

Output::Output()
{
  flux = "flux.csv";
  flux_std = "flux_std.csv";
  run_info = "run_info.log";
  batch_info = "batch_info.csv";
  binary = "false";
  bin_precision = "double";
  exce_time = "time.csv";
  vtk = "";
  mat = "";
}

void theta_title()
{
  if (is_terminal_utf8()) {
    std::cout

      << "████████╗██╗  ██╗███████╗████████╗ █████╗ \n"
      << "╚══██╔══╝██║  ██║██╔════╝╚══██╔══╝██╔══██╗\n"
      << "   ██║   ███████║█████╗     ██║   ███████║\n"
      << "   ██║   ██╔══██║██╔══╝     ██║   ██╔══██║\n"
      << "   ██║   ██║  ██║███████╗   ██║   ██║  ██║\n"
      << "   ╚═╝   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝\n"
      << "                 | The Theta Simulation Platform\n"
      << "       Copyright | 2024-2025 Zhejiang University, Hangzhou, China\n"
      << "         License | Non\n"
      << "         Version | " << THETA_VERSION_MAJOR << "."
      << THETA_VERSION_MINOR << "." << THETA_VERSION_RELEASE
      << (THETA_VERSION_DEV ? "-beta" : "") << "\n";
  } else {
    std::cout
      << " _________  ____  ____  ________  _________     _      \n"
      << "|  _   _  ||_   ||   _||_   __  ||  _   _  |   / \\     \n"
      << "|_/ | | \\_|  | |__| |    | |_ \\_||_/ | | \\_|  / _ \\    \n"
      << "    | |      |  __  |    |  _| _     | |     / ___ \\   \n"
      << "   _| |_    _| |  | |_  _| |__/ |   _| |_  _/ /   \\ \\_ \n"
      << "  |_____|  |____||____||________|  |_____||____| |____|\n"
      << "                 | The Theta Simulation Platform\n"
      << "       Copyright | 2024-2025 Zhejiang University, Hangzhou, China\n"
      << "         License | Non\n"
      << "         Version | " << THETA_VERSION_MAJOR << "."
      << THETA_VERSION_MINOR << "." << THETA_VERSION_RELEASE
      << (THETA_VERSION_DEV ? "-beta" : "") << "\n";
  }

  std::cout << "       Date/Time | " << time_stamp() << "\n";
  std::cout << "       Commit ID | " << THETA_GIT_COMMIT_ID << "\n";
  std::cout << " Build Precision | " << THETA_BUILD_PRECISION << "\n";


  std::cout << "\n";
  std::cout.flush();
}

std::string time_stamp()
{
  std::stringstream ts;
  std::time_t t = std::time(nullptr); // get time now
  ts << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
  return ts.str();
}

std::string append_flux_suffix(
  const std::string& name, const std::string& suffix)
{
  if (name.empty()) {
    return name;
  }
  const std::string csv_ext = ".csv";
  const std::string bin_ext = ".bin";

  if (name.size() >= csv_ext.size() &&
      name.substr(name.size() - csv_ext.size()) == csv_ext) {
    return name.substr(0, name.size() - csv_ext.size()) + suffix + csv_ext;
  }
  if (name.size() >= bin_ext.size() &&
      name.substr(name.size() - bin_ext.size()) == bin_ext) {
    return name.substr(0, name.size() - bin_ext.size()) + suffix + bin_ext;
  }
  return name + suffix;
}

} // namespace theta
