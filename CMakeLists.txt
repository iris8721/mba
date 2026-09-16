cmake_minimum_required(VERSION 3.16)
project(mba LANGUAGES CXX)

# C++20 required: abbreviated function templates (auto params) and if constexpr.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Core library: template implementations with explicit instantiations
# for uint8_t/uint16_t/uint32_t/uint64_t.
add_library(mba_core STATIC
    src/linear_mba.cpp
    src/lattice.cpp
    src/perm_poly.cpp
    src/poly.cpp
    src/solver.cpp
    src/nonlinear.cpp
)
target_include_directories(mba_core PUBLIC include)

add_executable(mba main.cpp)
target_link_libraries(mba PRIVATE mba_core)
