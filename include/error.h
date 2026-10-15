#ifndef THETA_ERROR_H
#define THETA_ERROR_H

#include <cstring>
#include <sstream>
#include <string>

#if defined(__GNUC__) || defined(__clang__)
#define UNREACHABLE() __builtin_unreachable()
#else
#define UNREACHABLE() (void)0
#endif

namespace theta {

// namespace simulation {

// extern char theta_err_msg[256];

// }

// inline void set_errmsg(const char* message)
// {
//   std::strcpy(simulation::theta_err_msg, message);
// }

// inline void set_errmsg(const std::string& message)
// {
//   std::strcpy(simulation::theta_err_msg, message.c_str());
// }

// inline void set_errmsg(const std::stringstream& message)
// {
//   std::strcpy(simulation::theta_err_msg, message.str().c_str());
// }

[[noreturn]] void fatal_error(const std::string& message, int err = -1);

[[noreturn]] inline void fatal_error(const std::stringstream& message)
{
  fatal_error(message.str());
}

[[noreturn]] inline void fatal_error(const char* message)
{
  fatal_error(std::string {message, std::strlen(message)});
}

void warning(const std::string& message);

inline void warning(const std::stringstream& message)
{
  warning(message.str());
}

} // namespace theta

#endif // THETA_ERROR_H
