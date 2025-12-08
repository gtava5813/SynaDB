# CMakeLists.txt for syna CMake Integration Example
#
# This example demonstrates how to integrate syna into a CMake project.
# It supports Windows, Linux, and macOS.
#
# Prerequisites:
#   - Build the syna library first: cargo build --release
#   - CMake 3.14 or higher
#
# Usage:
#   mkdir build && cd build
#   cmake ..
#   cmake --build .
#   ./syna_cmake_demo (or syna_cmake_demo.exe on Windows)
#
# _Requirements: 3.4_

cmake_minimum_required(VERSION 3.14)

project(synaCMakeDemo
    VERSION 1.0.0
    DESCRIPTION "CMake integration example for syna database"
    LANGUAGES C CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# ============================================================================
# Find syna Library
# ============================================================================

# Set the path to the syna project root (adjust as needed)
set(syna_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "Path to syna project root")

# Include directory
set(syna_INCLUDE_DIR "${syna_ROOT}/include")

# Library directory (release build)
set(syna_LIB_DIR "${syna_ROOT}/target/release")

# Determine library name based on platform
if(WIN32)
    set(syna_LIB_NAME "synadb")
    set(syna_LIB_FILE "${syna_LIB_DIR}/${syna_LIB_NAME}.dll")
    set(syna_IMPORT_LIB "${syna_LIB_DIR}/${syna_LIB_NAME}.dll.lib")
elseif(APPLE)
    set(syna_LIB_NAME "synadb")
    set(syna_LIB_FILE "${syna_LIB_DIR}/lib${syna_LIB_NAME}.dylib")
else()
    set(syna_LIB_NAME "synadb")
    set(syna_LIB_FILE "${syna_LIB_DIR}/lib${syna_LIB_NAME}.so")
endif()

# Check if library exists
if(NOT EXISTS "${syna_LIB_FILE}")
    message(FATAL_ERROR 
        "syna library not found at: ${syna_LIB_FILE}\n"
        "Please build the library first:\n"
        "  cd ${syna_ROOT} && cargo build --release"
    )
endif()

# Check if header exists
if(NOT EXISTS "${syna_INCLUDE_DIR}/synadb.h")
    message(FATAL_ERROR 
        "syna header not found at: ${syna_INCLUDE_DIR}/synadb.h"
    )
endif()

message(STATUS "Found syna library: ${syna_LIB_FILE}")
message(STATUS "Found syna header: ${syna_INCLUDE_DIR}/synadb.h")

# Create imported library target
add_library(syna SHARED IMPORTED)

set_target_properties(syna PROPERTIES
    IMPORTED_LOCATION "${syna_LIB_FILE}"
    INTERFACE_INCLUDE_DIRECTORIES "${syna_INCLUDE_DIR}"
)

# On Windows, we need to specify the import library
if(WIN32 AND EXISTS "${syna_IMPORT_LIB}")
    set_target_properties(syna PROPERTIES
        IMPORTED_IMPLIB "${syna_IMPORT_LIB}"
    )
endif()

# ============================================================================
# Build Demo Executable
# ============================================================================

add_executable(syna_cmake_demo main.cpp)

target_link_libraries(syna_cmake_demo PRIVATE syna)

# Platform-specific settings
if(UNIX AND NOT APPLE)
    # Linux: Add pthread and dl
    target_link_libraries(syna_cmake_demo PRIVATE pthread dl)
    
    # Set RPATH to find library at runtime
    set_target_properties(syna_cmake_demo PROPERTIES
        BUILD_RPATH "${syna_LIB_DIR}"
        INSTALL_RPATH "${syna_LIB_DIR}"
    )
endif()

if(APPLE)
    # macOS: Set RPATH
    set_target_properties(syna_cmake_demo PROPERTIES
        BUILD_RPATH "${syna_LIB_DIR}"
        INSTALL_RPATH "@executable_path/../lib"
    )
endif()

if(WIN32)
    # Windows: Copy DLL to output directory
    add_custom_command(TARGET syna_cmake_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${syna_LIB_FILE}"
            "$<TARGET_FILE_DIR:syna_cmake_demo>"
        COMMENT "Copying syna DLL to output directory"
    )
endif()

# ============================================================================
# Optional: Install Target
# ============================================================================

include(GNUInstallDirs)

install(TARGETS syna_cmake_demo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# ============================================================================
# Print Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== syna CMake Demo Configuration ===")
message(STATUS "  Platform:        ${CMAKE_SYSTEM_NAME}")
message(STATUS "  C Compiler:      ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  C++ Compiler:    ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  syna Root:   ${syna_ROOT}")
message(STATUS "  Library:         ${syna_LIB_FILE}")
message(STATUS "  Include Dir:     ${syna_INCLUDE_DIR}")
message(STATUS "")

