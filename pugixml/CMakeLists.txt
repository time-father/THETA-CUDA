add_library(pugixml STATIC src/pugixml.cpp)
target_include_directories(pugixml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(pugixml PUBLIC cxx_std_11)
