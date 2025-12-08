/**
 * @file basic_usage.c
 * @brief Demonstrates basic Entangle database operations in C
 * 
 * This demo shows how to:
 * - Open and close a database
 * - Write values of all types (float, int, text, bytes)
 * - Read values back
 * - Delete keys
 * - List all keys
 * - Handle errors properly
 * 
 * Build: See Makefile in this directory
 * 
 * _Requirements: 3.1_
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/entangle.h"

/* Helper macro for error checking */
#define CHECK_ERROR(result, operation) \
    do { \
        if ((result) < ENTANGLE_SUCCESS) { \
            fprintf(stderr, "Error in %s: %d\n", operation, result); \
            return 1; \
        } \
    } while(0)

/* Helper function to print error code meaning */
static const char* error_to_string(int32_t code) {
    switch (code) {
        case ENTANGLE_SUCCESS:         return "Success";
        case ENTANGLE_ERR_GENERIC:     return "Generic error";
        case ENTANGLE_ERR_DB_NOT_FOUND: return "Database not found";
        case ENTANGLE_ERR_INVALID_PATH: return "Invalid path";
        case ENTANGLE_ERR_IO:          return "I/O error";
        case ENTANGLE_ERR_SERIALIZATION: return "Serialization error";
        case ENTANGLE_ERR_KEY_NOT_FOUND: return "Key not found";
        case ENTANGLE_ERR_TYPE_MISMATCH: return "Type mismatch";
        case ENTANGLE_ERR_EMPTY_KEY:   return "Empty key";
        case ENTANGLE_ERR_KEY_TOO_LONG: return "Key too long";
        case ENTANGLE_ERR_INTERNAL_PANIC: return "Internal panic";
        default:                       return "Unknown error";
    }
}

/* Demo: Basic CRUD operations */
static int demo_crud(const char* db_path) {
    int32_t result;
    int64_t offset;
    
    printf("\n=== Demo: Basic CRUD Operations ===\n");
    
    /* Open database */
    printf("Opening database at: %s\n", db_path);
    result = entangle_open(db_path);
    CHECK_ERROR(result, "entangle_open");
    printf("  -> Database opened successfully\n");
    
    /* Write a float value */
    printf("\nWriting float value: temperature = 23.5\n");
    offset = entangle_put_float(db_path, "temperature", 23.5);
    if (offset < 0) {
        fprintf(stderr, "  -> Error: %s\n", error_to_string((int32_t)offset));
        return 1;
    }
    printf("  -> Written at offset: %lld\n", (long long)offset);
    
    /* Write an integer value */
    printf("\nWriting int value: count = 42\n");
    offset = entangle_put_int(db_path, "count", 42);
    if (offset < 0) {
        fprintf(stderr, "  -> Error: %s\n", error_to_string((int32_t)offset));
        return 1;
    }
    printf("  -> Written at offset: %lld\n", (long long)offset);
    
    /* Write a text value */
    printf("\nWriting text value: message = \"Hello, Entangle!\"\n");
    offset = entangle_put_text(db_path, "message", "Hello, Entangle!");
    if (offset < 0) {
        fprintf(stderr, "  -> Error: %s\n", error_to_string((int32_t)offset));
        return 1;
    }
    printf("  -> Written at offset: %lld\n", (long long)offset);
    
    /* Write a bytes value */
    uint8_t binary_data[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE};
    printf("\nWriting bytes value: binary_data = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE]\n");
    offset = entangle_put_bytes(db_path, "binary_data", binary_data, sizeof(binary_data));
    if (offset < 0) {
        fprintf(stderr, "  -> Error: %s\n", error_to_string((int32_t)offset));
        return 1;
    }
    printf("  -> Written at offset: %lld\n", (long long)offset);
    
    /* Read float value back */
    double temp_value;
    printf("\nReading float value: temperature\n");
    result = entangle_get_float(db_path, "temperature", &temp_value);
    if (result == ENTANGLE_SUCCESS) {
        printf("  -> Value: %f\n", temp_value);
    } else {
        fprintf(stderr, "  -> Error: %s\n", error_to_string(result));
    }
    
    /* Read integer value back */
    int64_t count_value;
    printf("\nReading int value: count\n");
    result = entangle_get_int(db_path, "count", &count_value);
    if (result == ENTANGLE_SUCCESS) {
        printf("  -> Value: %lld\n", (long long)count_value);
    } else {
        fprintf(stderr, "  -> Error: %s\n", error_to_string(result));
    }
    
    /* Read text value back */
    size_t text_len;
    printf("\nReading text value: message\n");
    char* text_value = entangle_get_text(db_path, "message", &text_len);
    if (text_value != NULL) {
        printf("  -> Value: \"%s\" (length: %zu)\n", text_value, text_len);
        entangle_free_text(text_value, text_len);
    } else {
        fprintf(stderr, "  -> Error: Key not found or type mismatch\n");
    }
    
    /* Read bytes value back */
    size_t bytes_len;
    printf("\nReading bytes value: binary_data\n");
    uint8_t* bytes_value = entangle_get_bytes(db_path, "binary_data", &bytes_len);
    if (bytes_value != NULL) {
        printf("  -> Value: [");
        for (size_t i = 0; i < bytes_len; i++) {
            printf("0x%02X%s", bytes_value[i], i < bytes_len - 1 ? ", " : "");
        }
        printf("] (length: %zu)\n", bytes_len);
        entangle_free_bytes(bytes_value, bytes_len);
    } else {
        fprintf(stderr, "  -> Error: Key not found or type mismatch\n");
    }
    
    /* Check if key exists */
    printf("\nChecking if 'temperature' exists\n");
    result = entangle_exists(db_path, "temperature");
    printf("  -> Exists: %s\n", result == 1 ? "yes" : "no");
    
    printf("\nChecking if 'nonexistent' exists\n");
    result = entangle_exists(db_path, "nonexistent");
    printf("  -> Exists: %s\n", result == 1 ? "yes" : "no");
    
    /* List all keys */
    size_t num_keys;
    printf("\nListing all keys:\n");
    char** keys = entangle_keys(db_path, &num_keys);
    if (keys != NULL) {
        for (size_t i = 0; i < num_keys; i++) {
            printf("  -> Key %zu: %s\n", i + 1, keys[i]);
        }
        entangle_free_keys(keys, num_keys);
    } else if (num_keys == 0) {
        printf("  -> No keys found\n");
    } else {
        fprintf(stderr, "  -> Error listing keys\n");
    }
    
    /* Delete a key */
    printf("\nDeleting key: count\n");
    result = entangle_delete(db_path, "count");
    CHECK_ERROR(result, "entangle_delete");
    printf("  -> Key deleted successfully\n");
    
    /* Verify deletion */
    printf("\nVerifying deletion - reading 'count':\n");
    result = entangle_get_int(db_path, "count", &count_value);
    if (result == ENTANGLE_ERR_KEY_NOT_FOUND) {
        printf("  -> Key not found (as expected after deletion)\n");
    } else if (result == ENTANGLE_SUCCESS) {
        printf("  -> Unexpected: Key still exists with value %lld\n", (long long)count_value);
    } else {
        fprintf(stderr, "  -> Error: %s\n", error_to_string(result));
    }
    
    /* List keys after deletion */
    printf("\nListing keys after deletion:\n");
    keys = entangle_keys(db_path, &num_keys);
    if (keys != NULL) {
        for (size_t i = 0; i < num_keys; i++) {
            printf("  -> Key %zu: %s\n", i + 1, keys[i]);
        }
        entangle_free_keys(keys, num_keys);
    } else if (num_keys == 0) {
        printf("  -> No keys found\n");
    }
    
    /* Close database */
    printf("\nClosing database\n");
    result = entangle_close(db_path);
    CHECK_ERROR(result, "entangle_close");
    printf("  -> Database closed successfully\n");
    
    return 0;
}

/* Demo: Time-series data and tensor extraction */
static int demo_timeseries(const char* db_path) {
    int32_t result;
    
    printf("\n=== Demo: Time-Series and Tensor Extraction ===\n");
    
    /* Open database */
    result = entangle_open(db_path);
    CHECK_ERROR(result, "entangle_open");
    
    /* Write multiple float values to simulate time-series */
    printf("\nWriting 10 sensor readings:\n");
    for (int i = 0; i < 10; i++) {
        double value = 20.0 + (double)i * 0.5;  /* Simulated temperature */
        int64_t offset = entangle_put_float(db_path, "sensor/temp", value);
        if (offset < 0) {
            fprintf(stderr, "  -> Error writing value %d\n", i);
            return 1;
        }
        printf("  -> Reading %d: %.1f\n", i + 1, value);
    }
    
    /* Extract history as tensor */
    size_t tensor_len;
    printf("\nExtracting history tensor for 'sensor/temp':\n");
    double* tensor = entangle_get_history_tensor(db_path, "sensor/temp", &tensor_len);
    if (tensor != NULL) {
        printf("  -> Tensor length: %zu\n", tensor_len);
        printf("  -> Values: [");
        for (size_t i = 0; i < tensor_len; i++) {
            printf("%.1f%s", tensor[i], i < tensor_len - 1 ? ", " : "");
        }
        printf("]\n");
        
        /* Calculate simple statistics */
        double sum = 0.0, min = tensor[0], max = tensor[0];
        for (size_t i = 0; i < tensor_len; i++) {
            sum += tensor[i];
            if (tensor[i] < min) min = tensor[i];
            if (tensor[i] > max) max = tensor[i];
        }
        printf("  -> Mean: %.2f, Min: %.1f, Max: %.1f\n", 
               sum / tensor_len, min, max);
        
        /* Free tensor memory */
        entangle_free_tensor(tensor, tensor_len);
    } else {
        fprintf(stderr, "  -> Error extracting tensor\n");
    }
    
    /* Close database */
    result = entangle_close(db_path);
    CHECK_ERROR(result, "entangle_close");
    
    return 0;
}

/* Demo: Error handling */
static int demo_error_handling(const char* db_path) {
    int32_t result;
    
    printf("\n=== Demo: Error Handling ===\n");
    
    /* Try to read from unopened database */
    printf("\nTrying to read from unopened database:\n");
    double value;
    result = entangle_get_float(db_path, "test", &value);
    printf("  -> Result: %d (%s)\n", result, error_to_string(result));
    
    /* Open database */
    result = entangle_open(db_path);
    CHECK_ERROR(result, "entangle_open");
    
    /* Try to read non-existent key */
    printf("\nTrying to read non-existent key:\n");
    result = entangle_get_float(db_path, "nonexistent_key", &value);
    printf("  -> Result: %d (%s)\n", result, error_to_string(result));
    
    /* Write an int, try to read as float (type mismatch) */
    printf("\nWriting int, then trying to read as float:\n");
    entangle_put_int(db_path, "int_value", 100);
    result = entangle_get_float(db_path, "int_value", &value);
    printf("  -> Result: %d (%s)\n", result, error_to_string(result));
    
    /* Try with NULL pointers */
    printf("\nTrying operations with NULL pointers:\n");
    result = entangle_get_float(db_path, NULL, &value);
    printf("  -> get_float with NULL key: %d (%s)\n", result, error_to_string(result));
    
    result = entangle_get_float(db_path, "test", NULL);
    printf("  -> get_float with NULL out: %d (%s)\n", result, error_to_string(result));
    
    /* Close database */
    result = entangle_close(db_path);
    CHECK_ERROR(result, "entangle_close");
    
    return 0;
}

/* Demo: Compaction */
static int demo_compaction(const char* db_path) {
    int32_t result;
    
    printf("\n=== Demo: Database Compaction ===\n");
    
    /* Open database */
    result = entangle_open(db_path);
    CHECK_ERROR(result, "entangle_open");
    
    /* Write multiple values to same key */
    printf("\nWriting 5 values to same key 'counter':\n");
    for (int i = 1; i <= 5; i++) {
        entangle_put_int(db_path, "counter", i * 10);
        printf("  -> Write %d: %d\n", i, i * 10);
    }
    
    /* Check history before compaction */
    size_t tensor_len;
    printf("\nHistory before compaction:\n");
    double* tensor = entangle_get_history_tensor(db_path, "counter", &tensor_len);
    if (tensor != NULL) {
        printf("  -> Length: %zu values\n", tensor_len);
        entangle_free_tensor(tensor, tensor_len);
    }
    
    /* Compact database */
    printf("\nCompacting database...\n");
    result = entangle_compact(db_path);
    CHECK_ERROR(result, "entangle_compact");
    printf("  -> Compaction complete\n");
    
    /* Check history after compaction */
    printf("\nHistory after compaction:\n");
    tensor = entangle_get_history_tensor(db_path, "counter", &tensor_len);
    if (tensor != NULL) {
        printf("  -> Length: %zu value(s)\n", tensor_len);
        if (tensor_len > 0) {
            printf("  -> Latest value: %.0f\n", tensor[0]);
        }
        entangle_free_tensor(tensor, tensor_len);
    }
    
    /* Close database */
    result = entangle_close(db_path);
    CHECK_ERROR(result, "entangle_close");
    
    return 0;
}

int main(int argc, char* argv[]) {
    const char* db_path = "demo_basic.db";
    
    /* Allow custom database path */
    if (argc > 1) {
        db_path = argv[1];
    }
    
    printf("Entangle C Basic Usage Demo\n");
    printf("============================\n");
    printf("Database path: %s\n", db_path);
    
    /* Run demos */
    int result = 0;
    
    result = demo_crud(db_path);
    if (result != 0) {
        fprintf(stderr, "CRUD demo failed\n");
        return result;
    }
    
    result = demo_timeseries(db_path);
    if (result != 0) {
        fprintf(stderr, "Time-series demo failed\n");
        return result;
    }
    
    result = demo_error_handling(db_path);
    if (result != 0) {
        fprintf(stderr, "Error handling demo failed\n");
        return result;
    }
    
    result = demo_compaction(db_path);
    if (result != 0) {
        fprintf(stderr, "Compaction demo failed\n");
        return result;
    }
    
    printf("\n============================\n");
    printf("All demos completed successfully!\n");
    
    /* Cleanup: remove demo database */
    remove(db_path);
    
    return 0;
}
