/**
 * @file main.cpp
 * @brief CMake integration example for SynaDB database
 * 
 * This example demonstrates how to use SynaDB in a CMake-based project.
 * It shows basic database operations using the C API.
 * 
 * Build instructions:
 *   mkdir build && cd build
 *   cmake ..
 *   cmake --build .
 *   ./syna_cmake_demo
 * 
 * _Requirements: 3.4_
 */

#include <iostream>
#include <cstdlib>
#include <cstring>

// Include SynaDB C header
extern "C" {
#include "synadb.h"
}

// Helper function to check errors
bool check_error(int32_t result, const char* operation) {
    if (result < SYNA_SUCCESS) {
        std::cerr << "Error in " << operation << ": " << result << std::endl;
        return false;
    }
    return true;
}

int main() {
    std::cout << "SynaDB CMake Integration Demo" << std::endl;
    std::cout << "================================" << std::endl;
    
    const char* db_path = "cmake_demo.db";
    int32_t result;
    
    // Open database
    std::cout << "\n1. Opening database: " << db_path << std::endl;
    result = SYNA_open(db_path);
    if (!check_error(result, "SYNA_open")) {
        return 1;
    }
    std::cout << "   -> Success!" << std::endl;
    
    // Write values
    std::cout << "\n2. Writing values:" << std::endl;
    
    int64_t offset = SYNA_put_float(db_path, "pi", 3.14159);
    if (offset >= 0) {
        std::cout << "   -> pi = 3.14159 (offset: " << offset << ")" << std::endl;
    }
    
    offset = SYNA_put_int(db_path, "answer", 42);
    if (offset >= 0) {
        std::cout << "   -> answer = 42 (offset: " << offset << ")" << std::endl;
    }
    
    offset = SYNA_put_text(db_path, "greeting", "Hello from CMake!");
    if (offset >= 0) {
        std::cout << "   -> greeting = \"Hello from CMake!\" (offset: " << offset << ")" << std::endl;
    }
    
    // Read values
    std::cout << "\n3. Reading values:" << std::endl;
    
    double pi_value;
    result = SYNA_get_float(db_path, "pi", &pi_value);
    if (result == SYNA_SUCCESS) {
        std::cout << "   -> pi = " << pi_value << std::endl;
    }
    
    int64_t answer_value;
    result = SYNA_get_int(db_path, "answer", &answer_value);
    if (result == SYNA_SUCCESS) {
        std::cout << "   -> answer = " << answer_value << std::endl;
    }
    
    size_t text_len;
    char* greeting = SYNA_get_text(db_path, "greeting", &text_len);
    if (greeting != nullptr) {
        std::cout << "   -> greeting = \"" << greeting << "\" (length: " << text_len << ")" << std::endl;
        SYNA_free_text(greeting, text_len);
    }
    
    // List keys
    std::cout << "\n4. Listing keys:" << std::endl;
    size_t num_keys;
    char** keys = SYNA_keys(db_path, &num_keys);
    if (keys != nullptr) {
        for (size_t i = 0; i < num_keys; ++i) {
            std::cout << "   -> " << keys[i] << std::endl;
        }
        SYNA_free_keys(keys, num_keys);
    }
    
    // Time-series demo
    std::cout << "\n5. Time-series demo:" << std::endl;
    std::cout << "   Writing 5 temperature readings..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        double temp = 20.0 + i * 0.5;
        SYNA_put_float(db_path, "temp", temp);
    }
    
    size_t tensor_len;
    double* tensor = SYNA_get_history_tensor(db_path, "temp", &tensor_len);
    if (tensor != nullptr) {
        std::cout << "   -> History tensor: [";
        for (size_t i = 0; i < tensor_len; ++i) {
            std::cout << tensor[i];
            if (i < tensor_len - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        SYNA_free_tensor(tensor, tensor_len);
    }
    
    // Close database
    std::cout << "\n6. Closing database" << std::endl;
    result = SYNA_close(db_path);
    if (!check_error(result, "SYNA_close")) {
        return 1;
    }
    std::cout << "   -> Success!" << std::endl;
    
    // Cleanup
    std::remove(db_path);
    
    std::cout << "\n================================" << std::endl;
    std::cout << "CMake integration demo completed!" << std::endl;
    std::cout << "\nThis demonstrates that SynaDB can be easily" << std::endl;
    std::cout << "integrated into CMake-based C++ projects." << std::endl;
    
    return 0;
}
