#ifndef THETA_VERSION_H
#define THETA_VERSION_H

#include <array>

// The build system overrides these macros; fallbacks keep standalone
// compilation of this header well-defined.
#ifndef THETA_GIT_COMMIT_ID
#define THETA_GIT_COMMIT_ID "unknown"
#endif

#ifndef THETA_BUILD_PRECISION
#define THETA_BUILD_PRECISION "CPU"
#endif

namespace theta {

// theta major, minor, and release numbers
constexpr int THETA_VERSION_MAJOR {1};
constexpr int THETA_VERSION_MINOR {0};
constexpr int THETA_VERSION_RELEASE {0};
constexpr bool THETA_VERSION_DEV {true};
constexpr std::array<int, 3> THETA_VERSION {
  THETA_VERSION_MAJOR, THETA_VERSION_MINOR, THETA_VERSION_RELEASE};

} // namespace theta

#endif // THETA_VERSION_H
