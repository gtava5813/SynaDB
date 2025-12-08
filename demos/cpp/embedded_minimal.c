/**
 * @file embedded_minimal.c
 * @brief Minimal memory footprint demo for embedded/constrained environments
 * 
 * This demo shows how to use Entangle with minimal memory overhead:
 * - No dynamic allocation in the hot path (after initial setup)
 * - Stack-based buffers for reading data
 * - Careful memory management
 * - Suitable for embedded systems with limited RAM
 * 
 * Memory Requirements:
 * - Stack: ~256 bytes for local variables
 * - Heap: Managed by Entangle library (configurable)
 * - The library itself requires ~1-2 MB for the shared object
 * 
 * Build: See Makefile in this directory
 * 
 * _Requirements: 3.3_
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../include/entangle.h"

/* ============================================================================
 * Configuration for Embedded Systems
 * ============================================================================ */

/* Maximum key length we'll use (keep small for embedded) */
#define MAX_KEY_LEN 32

/* Maximum text value length we'll handle */
#define MAX_TEXT_LEN 128

/* Number of sensor readings to buffer before writing */
#define SENSOR_BUFFER_SIZE 10

/* ============================================================================
 * Memory-Efficient Data Structures
 * ============================================================================ */

/**
 * @brief Fixed-size sensor reading structure
 * 
 * Using fixed-size structures avoids dynamic allocation.
 * Total size: 16 bytes per reading
 */
typedef struct {
    uint32_t timestamp;  /* Unix timestamp (4 bytes) */
    float value;         /* Sensor value (4 bytes) */
    uint16_t sensor_id;  /* Sensor identifier (2 bytes) */
    uint8_t flags;       /* Status flags (1 byte) */
    uint8_t reserved;    /* Padding for alignment (1 byte) */
} SensorReading;

/**
 * @brief Ring buffer for sensor readings
 * 
 * Fixed-size buffer that overwrites oldest data when full.
 * No dynamic allocation needed.
 */
typedef struct {
    SensorReading readings[SENSOR_BUFFER_SIZE];
    uint8_t head;
    uint8_t count;
} SensorBuffer;

/* ============================================================================
 * Helper Functions (No Dynamic Allocation)
 * ============================================================================ */

/**
 * @brief Initialize sensor buffer
 */
static void buffer_init(SensorBuffer* buf) {
    buf->head = 0;
    buf->count = 0;
    /* Zero-initialize readings */
    memset(buf->readings, 0, sizeof(buf->readings));
}

/**
 * @brief Add reading to buffer (overwrites oldest if full)
 */
static void buffer_push(SensorBuffer* buf, const SensorReading* reading) {
    uint8_t idx = (buf->head + buf->count) % SENSOR_BUFFER_SIZE;
    
    if (buf->count < SENSOR_BUFFER_SIZE) {
        buf->count++;
    } else {
        /* Buffer full, advance head (overwrite oldest) */
        buf->head = (buf->head + 1) % SENSOR_BUFFER_SIZE;
    }
    
    buf->readings[idx] = *reading;
}

/**
 * @brief Build key string without dynamic allocation
 * 
 * Uses a static buffer - NOT thread-safe, but suitable for
 * single-threaded embedded systems.
 */
static const char* build_key(const char* prefix, uint16_t id) {
    static char key_buffer[MAX_KEY_LEN];
    snprintf(key_buffer, MAX_KEY_LEN, "%s/%u", prefix, id);
    return key_buffer;
}

/**
 * @brief Print memory usage estimate
 */
static void print_memory_usage(void) {
    printf("\n=== Memory Usage Estimate ===\n");
    printf("  SensorReading struct:  %zu bytes\n", sizeof(SensorReading));
    printf("  SensorBuffer struct:   %zu bytes\n", sizeof(SensorBuffer));
    printf("  Key buffer (static):   %d bytes\n", MAX_KEY_LEN);
    printf("  Total stack usage:     ~%zu bytes\n", 
           sizeof(SensorBuffer) + MAX_KEY_LEN + 64);
    printf("\n  Note: Entangle library manages its own heap allocation\n");
    printf("  for index structures and file I/O buffers.\n");
}

/* ============================================================================
 * Demo: Minimal Memory Sensor Logger
 * ============================================================================ */

/**
 * @brief Demonstrates minimal-memory sensor data logging
 * 
 * This pattern is suitable for embedded systems:
 * 1. Buffer readings in fixed-size ring buffer
 * 2. Periodically flush to database
 * 3. No dynamic allocation in the logging loop
 */
static int demo_sensor_logger(const char* db_path) {
    int32_t result;
    SensorBuffer buffer;
    SensorReading reading;
    
    printf("\n=== Demo: Minimal Memory Sensor Logger ===\n");
    
    /* Initialize buffer (stack allocation) */
    buffer_init(&buffer);
    
    /* Open database */
    result = entangle_open(db_path);
    if (result != ENTANGLE_SUCCESS) {
        fprintf(stderr, "Failed to open database: %d\n", result);
        return 1;
    }
    
    /* Simulate sensor readings */
    printf("\nSimulating sensor readings...\n");
    for (uint32_t i = 0; i < 15; i++) {
        /* Create reading (no heap allocation) */
        reading.timestamp = 1700000000 + i * 60;  /* 1 minute intervals */
        reading.value = 20.0f + (float)(i % 5) * 0.5f;
        reading.sensor_id = (uint16_t)(i % 3);  /* 3 sensors */
        reading.flags = 0x01;  /* Valid reading */
        
        /* Add to buffer */
        buffer_push(&buffer, &reading);
        
        printf("  -> Buffered: sensor=%u, value=%.1f, ts=%u\n",
               reading.sensor_id, reading.value, reading.timestamp);
        
        /* Flush to database every SENSOR_BUFFER_SIZE readings */
        if (buffer.count == SENSOR_BUFFER_SIZE) {
            printf("  -> Flushing buffer to database...\n");
            
            for (uint8_t j = 0; j < buffer.count; j++) {
                uint8_t idx = (buffer.head + j) % SENSOR_BUFFER_SIZE;
                SensorReading* r = &buffer.readings[idx];
                
                /* Build key without allocation */
                const char* key = build_key("sensor", r->sensor_id);
                
                /* Write to database */
                int64_t offset = entangle_put_float(db_path, key, (double)r->value);
                if (offset < 0) {
                    fprintf(stderr, "Write failed: %lld\n", (long long)offset);
                }
            }
            
            /* Reset buffer */
            buffer_init(&buffer);
            printf("  -> Buffer flushed and reset\n");
        }
    }
    
    /* Flush remaining readings */
    if (buffer.count > 0) {
        printf("\nFlushing remaining %u readings...\n", buffer.count);
        for (uint8_t j = 0; j < buffer.count; j++) {
            uint8_t idx = (buffer.head + j) % SENSOR_BUFFER_SIZE;
            SensorReading* r = &buffer.readings[idx];
            const char* key = build_key("sensor", r->sensor_id);
            entangle_put_float(db_path, key, (double)r->value);
        }
    }
    
    /* Read back data for verification */
    printf("\nVerifying stored data:\n");
    for (uint16_t sensor_id = 0; sensor_id < 3; sensor_id++) {
        const char* key = build_key("sensor", sensor_id);
        size_t len;
        double* tensor = entangle_get_history_tensor(db_path, key, &len);
        
        if (tensor != NULL && len > 0) {
            printf("  -> %s: %zu readings, latest=%.1f\n", 
                   key, len, tensor[len - 1]);
            entangle_free_tensor(tensor, len);
        }
    }
    
    /* Close database */
    result = entangle_close(db_path);
    if (result != ENTANGLE_SUCCESS) {
        fprintf(stderr, "Failed to close database: %d\n", result);
        return 1;
    }
    
    return 0;
}

/* ============================================================================
 * Demo: Stack-Only Operations
 * ============================================================================ */

/**
 * @brief Demonstrates operations using only stack memory
 * 
 * All local variables are on the stack. The only heap usage
 * is internal to the Entangle library.
 */
static int demo_stack_only(const char* db_path) {
    int32_t result;
    double float_value;
    int64_t int_value;
    
    printf("\n=== Demo: Stack-Only Operations ===\n");
    
    /* Open database */
    result = entangle_open(db_path);
    if (result != ENTANGLE_SUCCESS) {
        return 1;
    }
    
    /* Write operations - no heap allocation needed */
    printf("\nWriting values (stack-only):\n");
    
    entangle_put_float(db_path, "temp", 25.5);
    printf("  -> temp = 25.5\n");
    
    entangle_put_int(db_path, "count", 100);
    printf("  -> count = 100\n");
    
    /* Read operations - output to stack variables */
    printf("\nReading values (stack-only):\n");
    
    result = entangle_get_float(db_path, "temp", &float_value);
    if (result == ENTANGLE_SUCCESS) {
        printf("  -> temp = %.1f\n", float_value);
    }
    
    result = entangle_get_int(db_path, "count", &int_value);
    if (result == ENTANGLE_SUCCESS) {
        printf("  -> count = %lld\n", (long long)int_value);
    }
    
    /* Check existence - no allocation */
    printf("\nChecking key existence:\n");
    result = entangle_exists(db_path, "temp");
    printf("  -> 'temp' exists: %s\n", result == 1 ? "yes" : "no");
    
    result = entangle_exists(db_path, "missing");
    printf("  -> 'missing' exists: %s\n", result == 1 ? "yes" : "no");
    
    /* Delete - no allocation */
    printf("\nDeleting 'count':\n");
    result = entangle_delete(db_path, "count");
    printf("  -> Delete result: %s\n", 
           result == ENTANGLE_SUCCESS ? "success" : "failed");
    
    /* Close database */
    entangle_close(db_path);
    
    return 0;
}

/* ============================================================================
 * Demo: Careful Memory Management
 * ============================================================================ */

/**
 * @brief Demonstrates careful memory management for constrained systems
 * 
 * Shows how to:
 * - Immediately free returned memory
 * - Avoid holding multiple allocations
 * - Process data in place
 */
static int demo_careful_memory(const char* db_path) {
    int32_t result;
    
    printf("\n=== Demo: Careful Memory Management ===\n");
    
    /* Open database */
    result = entangle_open(db_path);
    if (result != ENTANGLE_SUCCESS) {
        return 1;
    }
    
    /* Write some test data */
    printf("\nWriting test data...\n");
    for (int i = 0; i < 5; i++) {
        char key[MAX_KEY_LEN];
        snprintf(key, MAX_KEY_LEN, "item/%d", i);
        entangle_put_int(db_path, key, i * 10);
    }
    
    /* Process keys one at a time to minimize memory usage */
    printf("\nProcessing keys (one at a time):\n");
    
    size_t num_keys;
    char** keys = entangle_keys(db_path, &num_keys);
    
    if (keys != NULL && num_keys > 0) {
        /* Process each key immediately */
        for (size_t i = 0; i < num_keys; i++) {
            printf("  -> Processing: %s\n", keys[i]);
            /* In a real embedded system, you might:
             * - Read the value
             * - Process it
             * - Send it over a network
             * - All without storing multiple values in memory
             */
        }
        
        /* Free keys immediately after use */
        entangle_free_keys(keys, num_keys);
        keys = NULL;  /* Prevent accidental reuse */
        num_keys = 0;
        
        printf("  -> Keys freed immediately after processing\n");
    }
    
    /* Process tensor data in chunks if needed */
    printf("\nProcessing tensor data:\n");
    
    /* Write time-series data */
    for (int i = 0; i < 20; i++) {
        entangle_put_float(db_path, "series", (double)i);
    }
    
    size_t tensor_len;
    double* tensor = entangle_get_history_tensor(db_path, "series", &tensor_len);
    
    if (tensor != NULL && tensor_len > 0) {
        /* Process in place - calculate statistics without extra allocation */
        double sum = 0.0;
        double min = tensor[0];
        double max = tensor[0];
        
        for (size_t i = 0; i < tensor_len; i++) {
            sum += tensor[i];
            if (tensor[i] < min) min = tensor[i];
            if (tensor[i] > max) max = tensor[i];
        }
        
        printf("  -> Tensor length: %zu\n", tensor_len);
        printf("  -> Statistics (computed in-place):\n");
        printf("     Mean: %.2f, Min: %.1f, Max: %.1f\n", 
               sum / tensor_len, min, max);
        
        /* Free immediately */
        entangle_free_tensor(tensor, tensor_len);
        tensor = NULL;
        tensor_len = 0;
        
        printf("  -> Tensor freed immediately after processing\n");
    }
    
    /* Close database */
    entangle_close(db_path);
    
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char* argv[]) {
    const char* db_path = "demo_embedded.db";
    int result = 0;
    
    /* Allow custom database path */
    if (argc > 1) {
        db_path = argv[1];
    }
    
    printf("Entangle Embedded Minimal Demo\n");
    printf("==============================\n");
    printf("Database path: %s\n", db_path);
    
    /* Print memory usage information */
    print_memory_usage();
    
    /* Run demos */
    result = demo_stack_only(db_path);
    if (result != 0) {
        fprintf(stderr, "Stack-only demo failed\n");
        return result;
    }
    
    result = demo_sensor_logger(db_path);
    if (result != 0) {
        fprintf(stderr, "Sensor logger demo failed\n");
        return result;
    }
    
    result = demo_careful_memory(db_path);
    if (result != 0) {
        fprintf(stderr, "Careful memory demo failed\n");
        return result;
    }
    
    printf("\n==============================\n");
    printf("All embedded demos completed!\n");
    printf("\nKey takeaways for embedded systems:\n");
    printf("  1. Use fixed-size buffers (no malloc in hot path)\n");
    printf("  2. Free returned memory immediately after use\n");
    printf("  3. Process data in-place when possible\n");
    printf("  4. Use stack variables for simple read operations\n");
    printf("  5. Buffer writes and flush periodically\n");
    
    /* Cleanup */
    remove(db_path);
    
    return 0;
}
