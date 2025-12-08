/**
 * @file raii_wrapper.cpp
 * @brief C++ RAII wrapper for Syna database with exception-safe resource management
 * 
 * This demo shows how to:
 * - Create a C++ wrapper class with RAII semantics
 * - Use smart pointers for automatic memory management
 * - Handle exceptions safely
 * - Provide a modern C++ API
 * 
 * Build: See Makefile in this directory
 * 
 * _Requirements: 3.2_
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <optional>
#include <cstring>

extern "C" {
#include "../../include/synadb.h"
}

namespace Syna {

/**
 * @brief Exception class for Syna errors
 */
class SynaException : public std::runtime_error {
public:
    explicit SynaException(int32_t code, const std::string& message)
        : std::runtime_error(message), error_code_(code) {}
    
    int32_t error_code() const noexcept { return error_code_; }
    
    static std::string code_to_string(int32_t code) {
        switch (code) {
            case syna_SUCCESS:          return "Success";
            case syna_ERR_GENERIC:      return "Generic error";
            case syna_ERR_DB_NOT_FOUND: return "Database not found";
            case syna_ERR_INVALID_PATH: return "Invalid path";
            case syna_ERR_IO:           return "I/O error";
            case syna_ERR_SERIALIZATION: return "Serialization error";
            case syna_ERR_KEY_NOT_FOUND: return "Key not found";
            case syna_ERR_TYPE_MISMATCH: return "Type mismatch";
            case syna_ERR_EMPTY_KEY:    return "Empty key";
            case syna_ERR_KEY_TOO_LONG: return "Key too long";
            case syna_ERR_INTERNAL_PANIC: return "Internal panic";
            default:                        return "Unknown error";
        }
    }

private:
    int32_t error_code_;
};

/**
 * @brief Smart pointer deleter for tensor data
 */
struct TensorDeleter {
    size_t length;
    
    explicit TensorDeleter(size_t len = 0) : length(len) {}
    
    void operator()(double* ptr) const {
        if (ptr && length > 0) {
            syna_free_tensor(ptr, length);
        }
    }
};

/**
 * @brief Smart pointer deleter for text data
 */
struct TextDeleter {
    size_t length;
    
    explicit TextDeleter(size_t len = 0) : length(len) {}
    
    void operator()(char* ptr) const {
        if (ptr) {
            syna_free_text(ptr, length);
        }
    }
};

/**
 * @brief Smart pointer deleter for bytes data
 */
struct BytesDeleter {
    size_t length;
    
    explicit BytesDeleter(size_t len = 0) : length(len) {}
    
    void operator()(uint8_t* ptr) const {
        if (ptr && length > 0) {
            syna_free_bytes(ptr, length);
        }
    }
};

/**
 * @brief Smart pointer deleter for keys array
 */
struct KeysDeleter {
    size_t length;
    
    explicit KeysDeleter(size_t len = 0) : length(len) {}
    
    void operator()(char** ptr) const {
        if (ptr && length > 0) {
            syna_free_keys(ptr, length);
        }
    }
};

/**
 * @brief Type aliases for smart pointers
 */
using TensorPtr = std::unique_ptr<double[], TensorDeleter>;
using TextPtr = std::unique_ptr<char, TextDeleter>;
using BytesPtr = std::unique_ptr<uint8_t[], BytesDeleter>;
using KeysPtr = std::unique_ptr<char*[], KeysDeleter>;

/**
 * @brief Tensor wrapper with automatic memory management
 */
class Tensor {
public:
    Tensor() : data_(nullptr, TensorDeleter(0)), length_(0) {}
    
    Tensor(double* ptr, size_t len)
        : data_(ptr, TensorDeleter(len)), length_(len) {}
    
    // Move-only semantics
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    double* data() const { return data_.get(); }
    size_t size() const { return length_; }
    bool empty() const { return length_ == 0; }
    
    double operator[](size_t idx) const {
        if (idx >= length_) {
            throw std::out_of_range("Tensor index out of range");
        }
        return data_[idx];
    }
    
    // Iterator support
    double* begin() const { return data_.get(); }
    double* end() const { return data_.get() + length_; }
    
    // Convert to vector (copies data)
    std::vector<double> to_vector() const {
        return std::vector<double>(begin(), end());
    }

private:
    TensorPtr data_;
    size_t length_;
};

/**
 * @brief RAII wrapper for Syna database
 * 
 * Automatically opens the database on construction and closes on destruction.
 * All memory returned by FFI functions is managed via smart pointers.
 */
class SynaDB {
public:
    /**
     * @brief Opens or creates a database at the given path
     * @param path Path to the database file
     * @throws SynaException if database cannot be opened
     */
    explicit SynaDB(const std::string& path) : path_(path), is_open_(false) {
        int32_t result = syna_open(path_.c_str());
        if (result != syna_SUCCESS) {
            throw SynaException(result, 
                "Failed to open database: " + SynaException::code_to_string(result));
        }
        is_open_ = true;
    }
    
    /**
     * @brief Closes the database
     */
    ~SynaDB() noexcept {
        if (is_open_) {
            syna_close(path_.c_str());
        }
    }
    
    // Non-copyable, movable
    SynaDB(const SynaDB&) = delete;
    SynaDB& operator=(const SynaDB&) = delete;
    
    SynaDB(SynaDB&& other) noexcept
        : path_(std::move(other.path_)), is_open_(other.is_open_) {
        other.is_open_ = false;
    }
    
    SynaDB& operator=(SynaDB&& other) noexcept {
        if (this != &other) {
            if (is_open_) {
                syna_close(path_.c_str());
            }
            path_ = std::move(other.path_);
            is_open_ = other.is_open_;
            other.is_open_ = false;
        }
        return *this;
    }
    
    // ========== Write Operations ==========
    
    /**
     * @brief Writes a float value
     * @return Byte offset where entry was written
     */
    int64_t put(const std::string& key, double value) {
        int64_t result = syna_put_float(path_.c_str(), key.c_str(), value);
        if (result < 0) {
            throw SynaException(static_cast<int32_t>(result),
                "Failed to put float: " + SynaException::code_to_string(static_cast<int32_t>(result)));
        }
        return result;
    }
    
    /**
     * @brief Writes an integer value
     * @return Byte offset where entry was written
     */
    int64_t put(const std::string& key, int64_t value) {
        int64_t result = syna_put_int(path_.c_str(), key.c_str(), value);
        if (result < 0) {
            throw SynaException(static_cast<int32_t>(result),
                "Failed to put int: " + SynaException::code_to_string(static_cast<int32_t>(result)));
        }
        return result;
    }
    
    /**
     * @brief Writes a text value
     * @return Byte offset where entry was written
     */
    int64_t put(const std::string& key, const std::string& value) {
        int64_t result = syna_put_text(path_.c_str(), key.c_str(), value.c_str());
        if (result < 0) {
            throw SynaException(static_cast<int32_t>(result),
                "Failed to put text: " + SynaException::code_to_string(static_cast<int32_t>(result)));
        }
        return result;
    }
    
    /**
     * @brief Writes a bytes value
     * @return Byte offset where entry was written
     */
    int64_t put(const std::string& key, const std::vector<uint8_t>& value) {
        int64_t result = syna_put_bytes(path_.c_str(), key.c_str(), 
                                            value.data(), value.size());
        if (result < 0) {
            throw SynaException(static_cast<int32_t>(result),
                "Failed to put bytes: " + SynaException::code_to_string(static_cast<int32_t>(result)));
        }
        return result;
    }
    
    // ========== Read Operations ==========
    
    /**
     * @brief Reads a float value
     * @return The value, or std::nullopt if key not found
     * @throws SynaException on type mismatch or other errors
     */
    std::optional<double> get_float(const std::string& key) {
        double value;
        int32_t result = syna_get_float(path_.c_str(), key.c_str(), &value);
        
        if (result == syna_SUCCESS) {
            return value;
        } else if (result == syna_ERR_KEY_NOT_FOUND) {
            return std::nullopt;
        } else {
            throw SynaException(result,
                "Failed to get float: " + SynaException::code_to_string(result));
        }
    }
    
    /**
     * @brief Reads an integer value
     * @return The value, or std::nullopt if key not found
     * @throws SynaException on type mismatch or other errors
     */
    std::optional<int64_t> get_int(const std::string& key) {
        int64_t value;
        int32_t result = syna_get_int(path_.c_str(), key.c_str(), &value);
        
        if (result == syna_SUCCESS) {
            return value;
        } else if (result == syna_ERR_KEY_NOT_FOUND) {
            return std::nullopt;
        } else {
            throw SynaException(result,
                "Failed to get int: " + SynaException::code_to_string(result));
        }
    }
    
    /**
     * @brief Reads a text value
     * @return The value, or std::nullopt if key not found
     * @throws SynaException on type mismatch or other errors
     */
    std::optional<std::string> get_text(const std::string& key) {
        size_t len;
        char* ptr = syna_get_text(path_.c_str(), key.c_str(), &len);
        
        if (ptr != nullptr) {
            // Use smart pointer for automatic cleanup
            TextPtr text(ptr, TextDeleter(len));
            return std::string(text.get(), len);
        }
        return std::nullopt;
    }
    
    /**
     * @brief Reads a bytes value
     * @return The value, or std::nullopt if key not found
     * @throws SynaException on type mismatch or other errors
     */
    std::optional<std::vector<uint8_t>> get_bytes(const std::string& key) {
        size_t len;
        uint8_t* ptr = syna_get_bytes(path_.c_str(), key.c_str(), &len);
        
        if (ptr != nullptr && len > 0) {
            // Use smart pointer for automatic cleanup
            BytesPtr bytes(ptr, BytesDeleter(len));
            return std::vector<uint8_t>(bytes.get(), bytes.get() + len);
        }
        return std::nullopt;
    }
    
    /**
     * @brief Gets the history of float values as a tensor
     * @return Tensor with automatic memory management
     */
    Tensor get_history_tensor(const std::string& key) {
        size_t len;
        double* ptr = syna_get_history_tensor(path_.c_str(), key.c_str(), &len);
        
        if (ptr != nullptr && len > 0) {
            return Tensor(ptr, len);
        }
        return Tensor();
    }
    
    // ========== Key Operations ==========
    
    /**
     * @brief Deletes a key
     */
    void remove(const std::string& key) {
        int32_t result = syna_delete(path_.c_str(), key.c_str());
        if (result != syna_SUCCESS) {
            throw SynaException(result,
                "Failed to delete: " + SynaException::code_to_string(result));
        }
    }
    
    /**
     * @brief Checks if a key exists
     */
    bool exists(const std::string& key) {
        int32_t result = syna_exists(path_.c_str(), key.c_str());
        if (result < 0) {
            throw SynaException(result,
                "Failed to check exists: " + SynaException::code_to_string(result));
        }
        return result == 1;
    }
    
    /**
     * @brief Lists all keys
     */
    std::vector<std::string> keys() {
        size_t len;
        char** ptr = syna_keys(path_.c_str(), &len);
        
        std::vector<std::string> result;
        if (ptr != nullptr && len > 0) {
            // Use smart pointer for automatic cleanup
            KeysPtr keys_ptr(ptr, KeysDeleter(len));
            result.reserve(len);
            for (size_t i = 0; i < len; ++i) {
                result.emplace_back(keys_ptr[i]);
            }
        }
        return result;
    }
    
    // ========== Maintenance ==========
    
    /**
     * @brief Compacts the database
     */
    void compact() {
        int32_t result = syna_compact(path_.c_str());
        if (result != syna_SUCCESS) {
            throw SynaException(result,
                "Failed to compact: " + SynaException::code_to_string(result));
        }
    }
    
    /**
     * @brief Gets the database path
     */
    const std::string& path() const { return path_; }
    
    /**
     * @brief Checks if database is open
     */
    bool is_open() const { return is_open_; }

private:
    std::string path_;
    bool is_open_;
};

} // namespace Syna

// ============================================================================
// Demo Functions
// ============================================================================

void demo_basic_operations() {
    std::cout << "\n=== Demo: Basic Operations with RAII ===" << std::endl;
    
    // Database is automatically opened on construction
    Syna::SynaDB db("demo_raii.db");
    std::cout << "Database opened: " << db.path() << std::endl;
    
    // Write values using overloaded put()
    std::cout << "\nWriting values:" << std::endl;
    db.put("temperature", 23.5);
    std::cout << "  -> temperature = 23.5" << std::endl;
    
    db.put("count", int64_t(42));
    std::cout << "  -> count = 42" << std::endl;
    
    db.put("message", std::string("Hello, C++!"));
    std::cout << "  -> message = \"Hello, C++!\"" << std::endl;
    
    db.put("data", std::vector<uint8_t>{0xDE, 0xAD, 0xBE, 0xEF});
    std::cout << "  -> data = [0xDE, 0xAD, 0xBE, 0xEF]" << std::endl;
    
    // Read values using std::optional
    std::cout << "\nReading values:" << std::endl;
    
    if (auto temp = db.get_float("temperature")) {
        std::cout << "  -> temperature = " << *temp << std::endl;
    }
    
    if (auto count = db.get_int("count")) {
        std::cout << "  -> count = " << *count << std::endl;
    }
    
    if (auto msg = db.get_text("message")) {
        std::cout << "  -> message = \"" << *msg << "\"" << std::endl;
    }
    
    if (auto data = db.get_bytes("data")) {
        std::cout << "  -> data = [";
        for (size_t i = 0; i < data->size(); ++i) {
            std::cout << "0x" << std::hex << static_cast<int>((*data)[i]);
            if (i < data->size() - 1) std::cout << ", ";
        }
        std::cout << std::dec << "]" << std::endl;
    }
    
    // Handle missing keys gracefully
    std::cout << "\nHandling missing keys:" << std::endl;
    if (auto missing = db.get_float("nonexistent")) {
        std::cout << "  -> Found: " << *missing << std::endl;
    } else {
        std::cout << "  -> Key 'nonexistent' not found (as expected)" << std::endl;
    }
    
    // List keys
    std::cout << "\nListing keys:" << std::endl;
    for (const auto& key : db.keys()) {
        std::cout << "  -> " << key << std::endl;
    }
    
    // Database is automatically closed when db goes out of scope
    std::cout << "\nDatabase will be closed automatically..." << std::endl;
}

void demo_tensor_operations() {
    std::cout << "\n=== Demo: Tensor Operations ===" << std::endl;
    
    Syna::SynaDB db("demo_tensor.db");
    
    // Write time-series data
    std::cout << "\nWriting sensor readings:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        double value = 20.0 + i * 0.5;
        db.put("sensor/temp", value);
        std::cout << "  -> Reading " << (i + 1) << ": " << value << std::endl;
    }
    
    // Get history as tensor
    std::cout << "\nExtracting tensor:" << std::endl;
    auto tensor = db.get_history_tensor("sensor/temp");
    
    std::cout << "  -> Size: " << tensor.size() << std::endl;
    std::cout << "  -> Values: [";
    for (size_t i = 0; i < tensor.size(); ++i) {
        std::cout << tensor[i];
        if (i < tensor.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Use range-based for loop
    double sum = 0.0;
    for (double val : tensor) {
        sum += val;
    }
    std::cout << "  -> Mean: " << (sum / tensor.size()) << std::endl;
    
    // Convert to vector if needed
    auto vec = tensor.to_vector();
    std::cout << "  -> Converted to std::vector with " << vec.size() << " elements" << std::endl;
    
    // Tensor memory is automatically freed when it goes out of scope
}

void demo_exception_safety() {
    std::cout << "\n=== Demo: Exception Safety ===" << std::endl;
    
    // Test exception handling
    try {
        Syna::SynaDB db("demo_exceptions.db");
        
        // Write an int
        db.put("int_value", int64_t(100));
        
        // Try to read as float (type mismatch)
        std::cout << "\nTrying to read int as float:" << std::endl;
        auto value = db.get_float("int_value");
        if (value) {
            std::cout << "  -> Got value: " << *value << std::endl;
        } else {
            std::cout << "  -> No value returned (type mismatch)" << std::endl;
        }
        
    } catch (const Syna::SynaException& e) {
        std::cout << "  -> Caught SynaException: " << e.what() << std::endl;
        std::cout << "  -> Error code: " << e.error_code() << std::endl;
    }
    
    // Test RAII cleanup on exception
    std::cout << "\nTesting RAII cleanup on exception:" << std::endl;
    try {
        Syna::SynaDB db("demo_cleanup.db");
        db.put("test", 1.0);
        
        // Simulate an exception
        throw std::runtime_error("Simulated error");
        
    } catch (const std::exception& e) {
        std::cout << "  -> Exception caught: " << e.what() << std::endl;
        std::cout << "  -> Database was automatically closed (RAII)" << std::endl;
    }
}

void demo_move_semantics() {
    std::cout << "\n=== Demo: Move Semantics ===" << std::endl;
    
    // Create database
    Syna::SynaDB db1("demo_move.db");
    db1.put("value", 42.0);
    std::cout << "Created db1 at: " << db1.path() << std::endl;
    
    // Move to new variable
    Syna::SynaDB db2 = std::move(db1);
    std::cout << "Moved to db2 at: " << db2.path() << std::endl;
    std::cout << "db1 is_open: " << (db1.is_open() ? "true" : "false") << std::endl;
    std::cout << "db2 is_open: " << (db2.is_open() ? "true" : "false") << std::endl;
    
    // Read from moved database
    if (auto val = db2.get_float("value")) {
        std::cout << "Read from db2: " << *val << std::endl;
    }
}

int main() {
    std::cout << "Syna C++ RAII Wrapper Demo" << std::endl;
    std::cout << "==============================" << std::endl;
    
    try {
        demo_basic_operations();
        demo_tensor_operations();
        demo_exception_safety();
        demo_move_semantics();
        
        std::cout << "\n==============================" << std::endl;
        std::cout << "All demos completed successfully!" << std::endl;
        
        // Cleanup demo databases
        std::remove("demo_raii.db");
        std::remove("demo_tensor.db");
        std::remove("demo_exceptions.db");
        std::remove("demo_cleanup.db");
        std::remove("demo_move.db");
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

