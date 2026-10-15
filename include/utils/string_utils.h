#ifndef THETA_UTILS_STRING_UTILS_H
#define THETA_UTILS_STRING_UTILS_H

#include <string>
#include <vector>

using std::vector;

namespace theta {

std::string& strtrim(std::string& s);

char* strtrim(char* c_str);

std::string to_element(const std::string& name);

void to_lower(std::string& str);

int word_count(const std::string& str);

std::vector<std::string> split(const std::string& in);

bool ends_with(const std::string& value, const std::string& ending);

bool starts_with(const std::string& value, const std::string& beginning);

} // namespace theta

#endif // THETA_UTILS_STRING_UTILS_H
