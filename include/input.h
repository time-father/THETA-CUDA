#ifndef THETA_INPUT_H
#define THETA_INPUT_H

#include <fstream>
#include <string>
#include <sys/stat.h>

namespace theta {

void parse_command_line(int argc, char* argv[]);

void read_input(std::string input_name);

bool dir_exists(const std::string& path);
bool file_exists(const std::string& filename);

} // namespace theta

#endif // THETA_INPUT_H
