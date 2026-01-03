/**
 * @file Syna.h
 * @brief C-ABI interface for Syna embedded database
 * 
 * Syna is an embedded, log-structured, columnar-mapped database engine
 * optimized for time-series data and AI/ML tensor extraction.
 * 
 * @example
 * ```c
 * #include "Syna.h"
 * 
 * int main() {
 *     // Open database
 *     if (SYNA_open("my.db") != SYNA_SUCCESS) {
 *         return 1;
 *     }
 *     
 *     // Write values
 *     SYNA_put_float("my.db", "temperature", 23.5);
 *     
 *     // Read values
 *     double temp;
 *     if (SYNA_get_float("my.db", "temperature", &temp) == SYNA_SUCCESS) {
 *         printf("Temperature: %f\n", temp);
 *     }
 *     
 *     // Close database
 *     SYNA_close("my.db");
 *     return 0;
 * }
 * ```
 */

#ifndef SYNA_H
#define SYNA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Error Codes
 * ============================================================================ */

/** Operation completed successfully */
#define SYNA_SUCCESS         1

/** Generic/unspecified error */
#define SYNA_ERR_GENERIC     0

/** Database not found in registry (call SYNA_open first) */
#define SYNA_ERR_DB_NOT_FOUND    -1

/** Invalid path (null pointer or invalid UTF-8) */
#define SYNA_ERR_INVALID_PATH    -2

/** I/O error during file operations */
#define SYNA_ERR_IO              -3

/** Serialization/deserialization error */
#define SYNA_ERR_SERIALIZATION   -4

/** Key not found in database */
#define SYNA_ERR_KEY_NOT_FOUND   -5

/** Type mismatch (e.g., reading float from int key) */
#define SYNA_ERR_TYPE_MISMATCH   -6

/** Empty key is not allowed */
#define SYNA_ERR_EMPTY_KEY       -7

/** Key exceeds maximum length (65535 bytes) */
#define SYNA_ERR_KEY_TOO_LONG    -8

/** Internal panic (should not occur in normal operation) */
#define SYNA_ERR_INTERNAL_PANIC  -100

/* ============================================================================
 * Database Lifecycle Functions
 * ============================================================================ */

/**
 * Opens a database at the given path and registers it in the global registry.
 * 
 * If the database file exists, the index is rebuilt by scanning all entries.
 * If the file doesn't exist, a new empty database is created.
 * If the database is already open, this function returns success.
 * 
 * @param path  Null-terminated path to the database file
 * @return      SYNA_SUCCESS on success, error code on failure
 * 
 * @note The database must be closed with SYNA_close() when done.
 */
int32_t SYNA_open(const char* path);

/**
 * Closes a database and removes it from the global registry.
 * 
 * Flushes any pending writes to disk before closing.
 * 
 * @param path  Null-terminated path to the database file
 * @return      SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_close(const char* path);

/* ============================================================================
 * Write Functions
 * ============================================================================ */

/**
 * Writes a float (f64) value to the database.
 * 
 * @param path   Null-terminated path to the database file
 * @param key    Null-terminated key string (max 65535 bytes)
 * @param value  The float value to store
 * @return       Byte offset where entry was written (>= 0), or error code (< 0)
 */
int64_t SYNA_put_float(const char* path, const char* key, double value);

/**
 * Writes an integer (i64) value to the database.
 * 
 * @param path   Null-terminated path to the database file
 * @param key    Null-terminated key string (max 65535 bytes)
 * @param value  The integer value to store
 * @return       Byte offset where entry was written (>= 0), or error code (< 0)
 */
int64_t SYNA_put_int(const char* path, const char* key, int64_t value);

/**
 * Writes a text (string) value to the database.
 * 
 * @param path   Null-terminated path to the database file
 * @param key    Null-terminated key string (max 65535 bytes)
 * @param value  Null-terminated text value to store
 * @return       Byte offset where entry was written (>= 0), or error code (< 0)
 */
int64_t SYNA_put_text(const char* path, const char* key, const char* value);

/**
 * Writes a byte array to the database.
 * 
 * @param path   Null-terminated path to the database file
 * @param key    Null-terminated key string (max 65535 bytes)
 * @param data   Pointer to byte array to store
 * @param len    Length of the byte array
 * @return       Byte offset where entry was written (>= 0), or error code (< 0)
 */
int64_t SYNA_put_bytes(const char* path, const char* key, const uint8_t* data, size_t len);

/* ============================================================================
 * Read Functions
 * ============================================================================ */

/**
 * Reads the latest float value for a key.
 * 
 * @param path  Null-terminated path to the database file
 * @param key   Null-terminated key string
 * @param out   Pointer to store the retrieved value
 * @return      SYNA_SUCCESS on success, error code on failure
 * 
 * @note Returns SYNA_ERR_KEY_NOT_FOUND if key doesn't exist
 * @note Returns SYNA_ERR_TYPE_MISMATCH if value is not a float
 */
int32_t SYNA_get_float(const char* path, const char* key, double* out);

/**
 * Reads the latest integer value for a key.
 * 
 * @param path  Null-terminated path to the database file
 * @param key   Null-terminated key string
 * @param out   Pointer to store the retrieved value
 * @return      SYNA_SUCCESS on success, error code on failure
 * 
 * @note Returns SYNA_ERR_KEY_NOT_FOUND if key doesn't exist
 * @note Returns SYNA_ERR_TYPE_MISMATCH if value is not an integer
 */
int32_t SYNA_get_int(const char* path, const char* key, int64_t* out);

/**
 * Reads the latest text value for a key.
 * 
 * @param path     Null-terminated path to the database file
 * @param key      Null-terminated key string
 * @param out_len  Pointer to store the string length (excluding null terminator)
 * @return         Pointer to null-terminated string, or NULL on error/not found
 * 
 * @warning The returned pointer MUST be freed with SYNA_free_text()
 * 
 * @example
 * ```c
 * size_t len;
 * char* text = SYNA_get_text("my.db", "message", &len);
 * if (text) {
 *     printf("Message: %s (len=%zu)\n", text, len);
 *     SYNA_free_text(text, len);
 * }
 * ```
 */
char* SYNA_get_text(const char* path, const char* key, size_t* out_len);

/**
 * Frees memory allocated by SYNA_get_text().
 * 
 * @param ptr  Pointer returned by SYNA_get_text()
 * @param len  Length returned by SYNA_get_text() (excluding null terminator)
 * 
 * @note Safe to call with NULL pointer (no-op)
 */
void SYNA_free_text(char* ptr, size_t len);

/**
 * Reads the latest byte array value for a key.
 * 
 * @param path     Null-terminated path to the database file
 * @param key      Null-terminated key string
 * @param out_len  Pointer to store the array length
 * @return         Pointer to byte array, or NULL on error/not found
 * 
 * @warning The returned pointer MUST be freed with SYNA_free_bytes()
 * 
 * @example
 * ```c
 * size_t len;
 * uint8_t* data = SYNA_get_bytes("my.db", "binary_data", &len);
 * if (data) {
 *     // Use data...
 *     SYNA_free_bytes(data, len);
 * }
 * ```
 */
uint8_t* SYNA_get_bytes(const char* path, const char* key, size_t* out_len);

/**
 * Frees memory allocated by SYNA_get_bytes().
 * 
 * @param ptr  Pointer returned by SYNA_get_bytes()
 * @param len  Length returned by SYNA_get_bytes()
 * 
 * @note Safe to call with NULL pointer or zero length (no-op)
 */
void SYNA_free_bytes(uint8_t* ptr, size_t len);

/* ============================================================================
 * Tensor Functions (AI/ML)
 * ============================================================================ */

/**
 * Retrieves the complete history of float values for a key as a contiguous array.
 * 
 * This function is optimized for AI/ML workloads where you need to feed
 * time-series data directly to frameworks like PyTorch or TensorFlow.
 * 
 * @param path     Null-terminated path to the database file
 * @param key      Null-terminated key string
 * @param out_len  Pointer to store the array length
 * @return         Pointer to contiguous f64 array, or NULL on error
 * 
 * @warning The returned pointer MUST be freed with SYNA_free_tensor()
 * 
 * @example
 * ```c
 * size_t len;
 * double* tensor = SYNA_get_history_tensor("my.db", "sensor", &len);
 * if (tensor) {
 *     // Use tensor data...
 *     SYNA_free_tensor(tensor, len);
 * }
 * ```
 */
double* SYNA_get_history_tensor(const char* path, const char* key, size_t* out_len);

/**
 * Frees memory allocated by SYNA_get_history_tensor().
 * 
 * @param ptr  Pointer returned by SYNA_get_history_tensor()
 * @param len  Length returned by SYNA_get_history_tensor()
 * 
 * @note Safe to call with NULL pointer or zero length (no-op)
 * @note Must only be called once per pointer
 */
void SYNA_free_tensor(double* ptr, size_t len);

/* ============================================================================
 * Vector Functions (AI/ML Embeddings)
 * ============================================================================ */

/**
 * Stores a vector (embedding) in the database.
 * 
 * Vectors are stored with their dimensionality for validation during
 * similarity search operations. This is optimized for AI/ML embedding
 * storage.
 * 
 * @param path        Null-terminated path to the database file
 * @param key         Null-terminated key string (max 65535 bytes)
 * @param data        Pointer to f32 array containing the vector data
 * @param dimensions  Number of dimensions (elements) in the vector
 * @return            Byte offset where entry was written (>= 0), or error code (< 0)
 * 
 * @example
 * ```c
 * float embedding[768] = { ... };  // e.g., BERT embedding
 * int64_t offset = SYNA_put_vector("vectors.db", "doc/1", embedding, 768);
 * if (offset < 0) {
 *     // Handle error
 * }
 * ```
 */
int64_t SYNA_put_vector(const char* path, const char* key, 
                        const float* data, uint16_t dimensions);

/**
 * Retrieves a vector (embedding) from the database.
 * 
 * @param path            Null-terminated path to the database file
 * @param key             Null-terminated key string
 * @param out_data        Pointer to store the allocated f32 array pointer
 * @param out_dimensions  Pointer to store the number of dimensions
 * @return                SYNA_SUCCESS on success, error code on failure
 * 
 * @warning The returned data pointer MUST be freed with SYNA_free_vector()
 * 
 * @note Returns SYNA_ERR_KEY_NOT_FOUND if key doesn't exist
 * @note Returns SYNA_ERR_TYPE_MISMATCH if value is not a vector
 * 
 * @example
 * ```c
 * float* data;
 * uint16_t dims;
 * if (SYNA_get_vector("vectors.db", "doc/1", &data, &dims) == SYNA_SUCCESS) {
 *     printf("Vector has %u dimensions\n", dims);
 *     // Use data...
 *     SYNA_free_vector(data, dims);
 * }
 * ```
 */
int32_t SYNA_get_vector(const char* path, const char* key,
                        float** out_data, uint16_t* out_dimensions);

/**
 * Frees memory allocated by SYNA_get_vector().
 * 
 * @param data        Pointer returned by SYNA_get_vector() in out_data
 * @param dimensions  Dimensions returned by SYNA_get_vector() in out_dimensions
 * 
 * @note Safe to call with NULL pointer or zero dimensions (no-op)
 * @note Must only be called once per pointer
 */
void SYNA_free_vector(float* data, uint16_t dimensions);

/* ============================================================================
 * VectorStore Functions (High-Level Vector Search)
 * ============================================================================ */

/** Distance metric: Cosine similarity (1 - cos_sim) */
#define SYNA_METRIC_COSINE       0

/** Distance metric: Euclidean (L2) distance */
#define SYNA_METRIC_EUCLIDEAN    1

/** Distance metric: Negative dot product */
#define SYNA_METRIC_DOT_PRODUCT  2

/**
 * Creates a new vector store at the given path.
 * 
 * A vector store provides high-level operations for embedding storage
 * and similarity search. It wraps the underlying database with:
 * - Dimension validation
 * - Brute-force k-nearest neighbor search
 * - Key prefixing for namespace isolation
 * 
 * @param path        Null-terminated path to the database file
 * @param dimensions  Number of dimensions for vectors (64-4096)
 * @param metric      Distance metric (SYNA_METRIC_COSINE, SYNA_METRIC_EUCLIDEAN, 
 *                    or SYNA_METRIC_DOT_PRODUCT)
 * @return            SYNA_SUCCESS on success, error code on failure
 * 
 * @example
 * ```c
 * // Create a vector store for 768-dim embeddings with cosine similarity
 * if (SYNA_vector_store_new("vectors.db", 768, SYNA_METRIC_COSINE) != SYNA_SUCCESS) {
 *     // Handle error
 * }
 * ```
 */
int32_t SYNA_vector_store_new(const char* path, uint16_t dimensions, int32_t metric);

/**
 * Inserts a vector into the vector store.
 * 
 * @param path        Null-terminated path to the database file
 * @param key         Null-terminated key string (without prefix)
 * @param data        Pointer to f32 array containing the vector data
 * @param dimensions  Number of dimensions (must match store configuration)
 * @return            SYNA_SUCCESS on success, error code on failure
 * 
 * @note Returns SYNA_ERR_DB_NOT_FOUND if vector store was not created with
 *       SYNA_vector_store_new()
 * 
 * @example
 * ```c
 * float embedding[768] = { ... };
 * if (SYNA_vector_store_insert("vectors.db", "doc1", embedding, 768) != SYNA_SUCCESS) {
 *     // Handle error
 * }
 * ```
 */
int32_t SYNA_vector_store_insert(const char* path, const char* key,
                                  const float* data, uint16_t dimensions);

/**
 * Searches for k nearest neighbors in the vector store.
 * 
 * Returns a JSON array of results with the following structure:
 * ```json
 * [
 *   {"key": "doc1", "score": 0.123, "vector": [0.1, 0.2, ...]},
 *   {"key": "doc2", "score": 0.456, "vector": [0.3, 0.4, ...]}
 * ]
 * ```
 * 
 * @param path        Null-terminated path to the database file
 * @param query       Pointer to f32 array containing the query vector
 * @param dimensions  Number of dimensions in the query vector
 * @param k           Maximum number of results to return
 * @param out_json    Pointer to store the JSON result string
 * @return            Number of results found (>= 0), or error code (< 0)
 * 
 * @warning The returned JSON string MUST be freed with SYNA_free_json()
 * 
 * @note Returns SYNA_ERR_DB_NOT_FOUND if vector store was not created with
 *       SYNA_vector_store_new()
 * 
 * @example
 * ```c
 * float query[768] = { ... };
 * char* json;
 * int32_t count = SYNA_vector_store_search("vectors.db", query, 768, 5, &json);
 * if (count >= 0) {
 *     printf("Found %d results: %s\n", count, json);
 *     SYNA_free_json(json);
 * }
 * ```
 */
int32_t SYNA_vector_store_search(const char* path, const float* query,
                                  uint16_t dimensions, size_t k, char** out_json);

/**
 * Closes a vector store and saves any pending changes.
 * 
 * This removes the store from the global registry and saves the HNSW
 * index to disk if it has unsaved changes.
 * 
 * @param path  Null-terminated path to the database file
 * @return      SYNA_SUCCESS on success, error code on failure
 * 
 * @note After closing, the store cannot be used until reopened with
 *       SYNA_vector_store_new().
 */
int32_t SYNA_vector_store_close(const char* path);

/**
 * Flushes any pending changes to disk without closing the store.
 * 
 * This saves the HNSW index if it has unsaved changes.
 * 
 * @param path  Null-terminated path to the database file
 * @return      SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_vector_store_flush(const char* path);

/**
 * Frees a JSON string allocated by SYNA_vector_store_search().
 * 
 * @param json  Pointer returned by SYNA_vector_store_search() in out_json
 * 
 * @note Safe to call with NULL pointer (no-op)
 * @note Must only be called once per pointer
 */
void SYNA_free_json(char* json);

/* ============================================================================
 * TensorEngine Functions (Batch ML Operations)
 * ============================================================================ */

/** Data type: 32-bit floating point */
#define SYNA_DTYPE_FLOAT32  0

/** Data type: 64-bit floating point */
#define SYNA_DTYPE_FLOAT64  1

/** Data type: 32-bit signed integer */
#define SYNA_DTYPE_INT32    2

/** Data type: 64-bit signed integer */
#define SYNA_DTYPE_INT64    3

/**
 * Loads all values matching a pattern as a contiguous tensor.
 * 
 * Keys are matched using glob-style patterns:
 * - "prefix/*" matches all keys starting with "prefix/"
 * - "prefix*" matches all keys starting with "prefix"
 * - "exact_key" matches only that exact key
 * 
 * @param path       Null-terminated path to the database file
 * @param pattern    Glob-style pattern to match keys
 * @param dtype      Data type (SYNA_DTYPE_FLOAT32, SYNA_DTYPE_FLOAT64, etc.)
 * @param out_data   Pointer to store the allocated data buffer
 * @param out_len    Pointer to store the number of elements
 * @return           SYNA_SUCCESS on success, error code on failure
 * 
 * @warning The returned data pointer MUST be freed with SYNA_free_tensor_data()
 * 
 * @example
 * ```c
 * double* data;
 * size_t len;
 * if (SYNA_tensor_get("data.db", "sensor/*", SYNA_DTYPE_FLOAT64, 
 *                     (void**)&data, &len) == SYNA_SUCCESS) {
 *     printf("Loaded %zu values\n", len);
 *     // Use data...
 *     SYNA_free_tensor_data(data, len, SYNA_DTYPE_FLOAT64);
 * }
 * ```
 */
int32_t SYNA_tensor_get(const char* path, const char* pattern, int32_t dtype,
                        void** out_data, size_t* out_len);

/**
 * Stores tensor data with auto-generated keys.
 * 
 * Each element in the tensor is stored with a key of the form
 * "{key_prefix}{index}" where index is zero-padded to 8 digits.
 * 
 * @param path        Null-terminated path to the database file
 * @param key_prefix  Prefix for generated keys (e.g., "train/")
 * @param data        Pointer to the tensor data
 * @param len         Number of elements in the tensor
 * @param dtype       Data type of the tensor elements
 * @return            Number of elements stored (>= 0), or error code (< 0)
 * 
 * @example
 * ```c
 * double values[] = {1.0, 2.0, 3.0, 4.0};
 * int32_t count = SYNA_tensor_put("data.db", "values/", values, 4, SYNA_DTYPE_FLOAT64);
 * if (count >= 0) {
 *     printf("Stored %d values\n", count);
 * }
 * ```
 */
int32_t SYNA_tensor_put(const char* path, const char* key_prefix,
                        const void* data, size_t len, int32_t dtype);

/**
 * Frees memory allocated by SYNA_tensor_get().
 * 
 * @param data   Pointer returned by SYNA_tensor_get() in out_data
 * @param len    Length returned by SYNA_tensor_get() in out_len
 * @param dtype  Data type used in SYNA_tensor_get()
 * 
 * @note Safe to call with NULL pointer or zero length (no-op)
 * @note Must only be called once per pointer
 */
void SYNA_free_tensor_data(void* data, size_t len, int32_t dtype);

/* ============================================================================
 * HNSW Index Functions (Approximate Nearest Neighbor)
 * ============================================================================ */

/**
 * Creates a new HNSW index for approximate nearest neighbor search.
 * 
 * HNSW (Hierarchical Navigable Small World) provides O(log N) search time
 * compared to O(N) for brute force, enabling million-scale vector search
 * with <10ms latency.
 * 
 * @param path        Null-terminated path to the index file (.hnsw)
 * @param dimensions  Number of dimensions for vectors (64-4096)
 * @param metric      Distance metric (SYNA_METRIC_COSINE, SYNA_METRIC_EUCLIDEAN,
 *                    or SYNA_METRIC_DOT_PRODUCT)
 * @param m           Max connections per node (default: 16, range: 8-64)
 * @param ef_construction  Build quality parameter (default: 200, range: 100-500)
 * @return            SYNA_SUCCESS on success, error code on failure
 * 
 * @example
 * ```c
 * // Create HNSW index with default parameters
 * if (SYNA_hnsw_create("vectors.hnsw", 768, SYNA_METRIC_COSINE, 16, 200) != SYNA_SUCCESS) {
 *     // Handle error
 * }
 * ```
 */
int32_t SYNA_hnsw_create(const char* path, uint16_t dimensions, int32_t metric,
                         size_t m, size_t ef_construction);

/**
 * Loads an existing HNSW index from a file.
 * 
 * @param path  Null-terminated path to the index file (.hnsw)
 * @return      SYNA_SUCCESS on success, error code on failure
 * 
 * @note The index must have been created with SYNA_hnsw_create() or saved
 *       with SYNA_hnsw_save().
 */
int32_t SYNA_hnsw_load(const char* path);

/**
 * Saves the HNSW index to a file.
 * 
 * @param path  Null-terminated path to save the index file (.hnsw)
 * @return      SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_hnsw_save(const char* path);

/**
 * Inserts a vector into the HNSW index.
 * 
 * @param path        Null-terminated path to the index file
 * @param key         Null-terminated key string for the vector
 * @param data        Pointer to f32 array containing the vector data
 * @param dimensions  Number of dimensions (must match index configuration)
 * @return            SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_hnsw_insert(const char* path, const char* key,
                         const float* data, uint16_t dimensions);

/**
 * Searches for k approximate nearest neighbors in the HNSW index.
 * 
 * Returns a JSON array of results with the following structure:
 * ```json
 * [
 *   {"key": "doc1", "distance": 0.123},
 *   {"key": "doc2", "distance": 0.456}
 * ]
 * ```
 * 
 * @param path        Null-terminated path to the index file
 * @param query       Pointer to f32 array containing the query vector
 * @param dimensions  Number of dimensions in the query vector
 * @param k           Maximum number of results to return
 * @param ef_search   Search quality parameter (higher = better recall, slower)
 * @param out_json    Pointer to store the JSON result string
 * @return            Number of results found (>= 0), or error code (< 0)
 * 
 * @warning The returned JSON string MUST be freed with SYNA_free_json()
 * 
 * @example
 * ```c
 * float query[768] = { ... };
 * char* json;
 * int32_t count = SYNA_hnsw_search("vectors.hnsw", query, 768, 10, 100, &json);
 * if (count >= 0) {
 *     printf("Found %d results: %s\n", count, json);
 *     SYNA_free_json(json);
 * }
 * ```
 */
int32_t SYNA_hnsw_search(const char* path, const float* query,
                         uint16_t dimensions, size_t k, size_t ef_search,
                         char** out_json);

/**
 * Gets statistics about the HNSW index.
 * 
 * Returns a JSON object with the following structure:
 * ```json
 * {
 *   "num_nodes": 100000,
 *   "max_level": 4,
 *   "total_edges": 1600000,
 *   "avg_edges_per_node": 16.0
 * }
 * ```
 * 
 * @param path      Null-terminated path to the index file
 * @param out_json  Pointer to store the JSON result string
 * @return          SYNA_SUCCESS on success, error code on failure
 * 
 * @warning The returned JSON string MUST be freed with SYNA_free_json()
 */
int32_t SYNA_hnsw_stats(const char* path, char** out_json);

/**
 * Closes and frees an HNSW index.
 * 
 * @param path  Null-terminated path to the index file
 * @return      SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_hnsw_close(const char* path);

/* ============================================================================
 * Delete Functions
 * ============================================================================ */

/**
 * Deletes a key from the database by appending a tombstone entry.
 * 
 * The key will no longer appear in SYNA_keys() and SYNA_get_*()
 * will return SYNA_ERR_KEY_NOT_FOUND.
 * 
 * @param path  Null-terminated path to the database file
 * @param key   Null-terminated key string to delete
 * @return      SYNA_SUCCESS on success, error code on failure
 * 
 * @note Writing to a deleted key "resurrects" it with the new value
 */
int32_t SYNA_delete(const char* path, const char* key);

/**
 * Checks if a key exists in the database and is not deleted.
 * 
 * @param path  Null-terminated path to the database file
 * @param key   Null-terminated key string to check
 * @return      1 if key exists, 0 if not, negative error code on failure
 */
int32_t SYNA_exists(const char* path, const char* key);

/* ============================================================================
 * Maintenance Functions
 * ============================================================================ */

/**
 * Compacts the database by rewriting only the latest non-deleted entries.
 * 
 * This operation reclaims disk space by removing:
 * - Deleted entries (tombstones)
 * - Old versions of keys (only latest value is kept)
 * 
 * @param path  Null-terminated path to the database file
 * @return      SYNA_SUCCESS on success, error code on failure
 * 
 * @warning After compaction, get_history() will only return the latest value
 */
int32_t SYNA_compact(const char* path);

/**
 * Returns a list of all non-deleted keys in the database.
 * 
 * @param path     Null-terminated path to the database file
 * @param out_len  Pointer to store the number of keys
 * @return         Array of null-terminated key strings, or NULL on error
 * 
 * @warning The returned array MUST be freed with SYNA_free_keys()
 * 
 * @example
 * ```c
 * size_t len;
 * char** keys = SYNA_keys("my.db", &len);
 * if (keys) {
 *     for (size_t i = 0; i < len; i++) {
 *         printf("Key: %s\n", keys[i]);
 *     }
 *     SYNA_free_keys(keys, len);
 * }
 * ```
 */
char** SYNA_keys(const char* path, size_t* out_len);

/**
 * Frees memory allocated by SYNA_keys().
 * 
 * @param keys  Array returned by SYNA_keys()
 * @param len   Length returned by SYNA_keys()
 * 
 * @note Safe to call with NULL pointer or zero length (no-op)
 * @note Must only be called once per pointer
 */
void SYNA_free_keys(char** keys, size_t len);

/* ============================================================================
 * Model Registry Functions
 * ============================================================================ */

/** Model stage: Development (initial stage) */
#define SYNA_STAGE_DEVELOPMENT  0

/** Model stage: Staging (testing before production) */
#define SYNA_STAGE_STAGING      1

/** Model stage: Production (actively serving) */
#define SYNA_STAGE_PRODUCTION   2

/** Model stage: Archived (retired) */
#define SYNA_STAGE_ARCHIVED     3

/**
 * Opens or creates a model registry at the given path.
 * 
 * @param path  Null-terminated path to the registry database file
 * @return      SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_model_registry_open(const char* path);

/**
 * Saves a model to the registry with automatic versioning.
 * 
 * @param path            Null-terminated path to the registry
 * @param name            Null-terminated model name
 * @param data            Pointer to the model data bytes
 * @param data_len        Length of the model data
 * @param metadata_json   Null-terminated JSON string with metadata (can be NULL)
 * @param out_version     Pointer to store the assigned version number
 * @param out_checksum    Pointer to store the checksum string (must free with SYNA_free_text)
 * @param out_checksum_len Pointer to store the checksum string length
 * @return                SYNA_SUCCESS on success, error code on failure
 * 
 * @warning The returned checksum string MUST be freed with SYNA_free_text()
 */
int64_t SYNA_model_save(const char* path, const char* name,
                        const uint8_t* data, size_t data_len,
                        const char* metadata_json,
                        uint32_t* out_version,
                        char** out_checksum, size_t* out_checksum_len);

/**
 * Loads a model from the registry with checksum verification.
 * 
 * @param path          Null-terminated path to the registry
 * @param name          Null-terminated model name
 * @param version       Version number to load (0 for latest)
 * @param out_data      Pointer to store the model data pointer
 * @param out_data_len  Pointer to store the model data length
 * @param out_meta_json Pointer to store the metadata JSON string
 * @param out_meta_len  Pointer to store the metadata JSON length
 * @return              SYNA_SUCCESS on success, error code on failure
 * 
 * @warning The returned data MUST be freed with SYNA_free_bytes()
 * @warning The returned metadata JSON MUST be freed with SYNA_free_text()
 */
int32_t SYNA_model_load(const char* path, const char* name, uint32_t version,
                        uint8_t** out_data, size_t* out_data_len,
                        char** out_meta_json, size_t* out_meta_len);

/**
 * Lists all versions of a model.
 * 
 * Returns a JSON array of version metadata.
 * 
 * @param path      Null-terminated path to the registry
 * @param name      Null-terminated model name
 * @param out_json  Pointer to store the JSON result string
 * @param out_len   Pointer to store the JSON string length
 * @return          Number of versions found (>= 0), or error code (< 0)
 * 
 * @warning The returned JSON string MUST be freed with SYNA_free_json()
 */
int32_t SYNA_model_list(const char* path, const char* name,
                        char** out_json, size_t* out_len);

/**
 * Sets the deployment stage for a model version.
 * 
 * @param path     Null-terminated path to the registry
 * @param name     Null-terminated model name
 * @param version  Version number to update
 * @param stage    Stage (SYNA_STAGE_DEVELOPMENT, SYNA_STAGE_STAGING, etc.)
 * @return         SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_model_set_stage(const char* path, const char* name,
                             uint32_t version, int32_t stage);

/* ============================================================================
 * Experiment Tracking Functions
 * ============================================================================ */

/** Run status: Running (in progress) */
#define SYNA_RUN_RUNNING    0

/** Run status: Completed (finished successfully) */
#define SYNA_RUN_COMPLETED  1

/** Run status: Failed (encountered error) */
#define SYNA_RUN_FAILED     2

/** Run status: Killed (manually terminated) */
#define SYNA_RUN_KILLED     3

/**
 * Opens or creates an experiment tracker at the given path.
 * 
 * @param path  Null-terminated path to the tracker database file
 * @return      SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_exp_tracker_open(const char* path);

/**
 * Starts a new experiment run.
 * 
 * @param path           Null-terminated path to the tracker
 * @param experiment     Null-terminated experiment name
 * @param tags_json      Null-terminated JSON array of tags (can be NULL)
 * @param out_run_id     Pointer to store the run ID string
 * @param out_run_id_len Pointer to store the run ID length
 * @return               SYNA_SUCCESS on success, error code on failure
 * 
 * @warning The returned run ID MUST be freed with SYNA_free_text()
 */
int32_t SYNA_exp_start_run(const char* path, const char* experiment,
                           const char* tags_json,
                           char** out_run_id, size_t* out_run_id_len);

/**
 * Logs a parameter (hyperparameter) for a run.
 * 
 * @param path    Null-terminated path to the tracker
 * @param run_id  Null-terminated run ID
 * @param key     Null-terminated parameter name
 * @param value   Null-terminated parameter value
 * @return        SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_exp_log_param(const char* path, const char* run_id,
                           const char* key, const char* value);

/**
 * Logs a metric value for a run.
 * 
 * @param path    Null-terminated path to the tracker
 * @param run_id  Null-terminated run ID
 * @param key     Null-terminated metric name
 * @param value   The metric value (f64)
 * @param step    Step number (-1 for auto-generated timestamp)
 * @return        SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_exp_log_metric(const char* path, const char* run_id,
                            const char* key, double value, int64_t step);

/**
 * Logs an artifact (file, plot, model) for a run.
 * 
 * @param path      Null-terminated path to the tracker
 * @param run_id    Null-terminated run ID
 * @param name      Null-terminated artifact name
 * @param data      Pointer to the artifact data
 * @param data_len  Length of the artifact data
 * @return          SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_exp_log_artifact(const char* path, const char* run_id,
                              const char* name,
                              const uint8_t* data, size_t data_len);

/**
 * Ends a run with the given status.
 * 
 * @param path    Null-terminated path to the tracker
 * @param run_id  Null-terminated run ID
 * @param status  Status (SYNA_RUN_COMPLETED, SYNA_RUN_FAILED, etc.)
 * @return        SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_exp_end_run(const char* path, const char* run_id, int32_t status);

/* ============================================================================
 * Gravity Well Index (GWI) Functions
 * ============================================================================ */

/**
 * Creates a new Gravity Well Index at the given path.
 * 
 * GWI is a novel append-only vector index with O(N) build time,
 * 168x faster than HNSW for large datasets.
 * 
 * @param path        Null-terminated path to the index file (.gwi)
 * @param dimensions  Number of dimensions for vectors (64-7168)
 * @return            SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_gwi_new(const char* path, uint16_t dimensions);

/**
 * Opens an existing Gravity Well Index from a file.
 * 
 * @param path  Null-terminated path to the index file (.gwi)
 * @return      SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_gwi_open(const char* path);

/**
 * Initializes the GWI with sample vectors to create attractors.
 * 
 * Must be called before inserting vectors. The sample vectors are used
 * to create "gravity wells" that organize the index structure.
 * 
 * @param path        Null-terminated path to the index file
 * @param vectors     Pointer to f32 array of sample vectors (contiguous)
 * @param count       Number of sample vectors
 * @param dimensions  Number of dimensions per vector
 * @return            SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_gwi_initialize(const char* path, const float* vectors,
                            size_t count, uint16_t dimensions);

/**
 * Inserts a single vector into the GWI.
 * 
 * @param path        Null-terminated path to the index file
 * @param key         Null-terminated key string for the vector
 * @param vector      Pointer to f32 array containing the vector data
 * @param dimensions  Number of dimensions (must match index configuration)
 * @return            SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_gwi_insert(const char* path, const char* key,
                        const float* vector, uint16_t dimensions);

/**
 * Inserts multiple vectors into the GWI (batch mode).
 * 
 * @param path        Null-terminated path to the index file
 * @param keys        Array of null-terminated key strings
 * @param vectors     Pointer to f32 array of vectors (contiguous)
 * @param count       Number of vectors to insert
 * @param dimensions  Number of dimensions per vector
 * @return            Number of vectors inserted (>= 0), or error code (< 0)
 */
int32_t SYNA_gwi_insert_batch(const char* path, const char* const* keys,
                              const float* vectors, size_t count,
                              uint16_t dimensions);

/**
 * Searches for k nearest neighbors in the GWI.
 * 
 * @param path        Null-terminated path to the index file
 * @param query       Pointer to f32 array containing the query vector
 * @param dimensions  Number of dimensions in the query vector
 * @param k           Maximum number of results to return
 * @param out_json    Pointer to store the JSON result string
 * @return            Number of results found (>= 0), or error code (< 0)
 * 
 * @warning The returned JSON string MUST be freed with SYNA_free_json()
 */
int32_t SYNA_gwi_search(const char* path, const float* query,
                        uint16_t dimensions, size_t k, char** out_json);

/**
 * Searches with custom nprobe parameter for recall tuning.
 * 
 * Higher nprobe values increase recall at the cost of search time.
 * - nprobe=50: ~98% recall
 * - nprobe=100: ~100% recall
 * 
 * @param path        Null-terminated path to the index file
 * @param query       Pointer to f32 array containing the query vector
 * @param dimensions  Number of dimensions in the query vector
 * @param k           Maximum number of results to return
 * @param nprobe      Number of attractors to probe (higher = better recall)
 * @param out_json    Pointer to store the JSON result string
 * @return            Number of results found (>= 0), or error code (< 0)
 * 
 * @warning The returned JSON string MUST be freed with SYNA_free_json()
 */
int32_t SYNA_gwi_search_nprobe(const char* path, const float* query,
                               uint16_t dimensions, size_t k, size_t nprobe,
                               char** out_json);

/**
 * Flushes any pending changes to disk.
 * 
 * @param path  Null-terminated path to the index file
 * @return      SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_gwi_flush(const char* path);

/**
 * Closes the GWI and removes it from the registry.
 * 
 * @param path  Null-terminated path to the index file
 * @return      SYNA_SUCCESS on success, error code on failure
 */
int32_t SYNA_gwi_close(const char* path);

/**
 * Returns the number of vectors in the GWI.
 * 
 * @param path  Null-terminated path to the index file
 * @return      Number of vectors (>= 0), or error code (< 0)
 */
int64_t SYNA_gwi_len(const char* path);

#ifdef __cplusplus
}
#endif

#endif /* SYNA_H */

