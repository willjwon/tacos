# Check CMake version
cmake_minimum_required(VERSION 3.25)

# Create Project
project(TACOS)

# Setup compilation options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type
set(CMAKE_BUILD_TYPE Release)

# Debug option
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# Project include option
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tacos)

# Target (common) compilation files
file(GLOB srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(TACOS ${srcs})
target_sources(TACOS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
set_target_properties(TACOS PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
