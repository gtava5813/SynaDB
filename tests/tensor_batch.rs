//! Property-based tests for batch tensor operations.
//!
//! **Feature: Syna-ai-native, Property 20: Batch Tensor Shape Consistency**
//! **Validates: Requirements 2.1, 2.2**
//!
//! This test verifies that:
//! - Storing a tensor and loading it back preserves the shape
//! - The number of elements in the returned tensor matches the stored count
//! - Data round-trips correctly through put_tensor/get_tensor

use proptest::prelude::*;
use synadb::tensor::{DType, TensorEngine};
use synadb::SynaDB;
use tempfile::tempdir;

/// Generator for valid tensor element counts (10-1000 elements).
fn arb_n_elements() -> impl Strategy<Value = usize> {
    10usize..1000
}

/// Generator for valid data types.
fn arb_dtype() -> impl Strategy<Value = DType> {
    prop_oneof![
        Just(DType::Float32),
        Just(DType::Float64),
        Just(DType::Int32),
        Just(DType::Int64),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-ai-native, Property 20: Batch Tensor Shape Consistency**
    ///
    /// For any batch tensor operation with specified shape S, the returned tensor
    /// SHALL have shape S, and the total elements SHALL equal the product of dimensions.
    ///
    /// This property tests that:
    /// 1. put_tensor stores the correct number of elements
    /// 2. get_tensor returns a tensor with shape matching the stored element count
    /// 3. The round-trip preserves the data
    ///
    /// **Validates: Requirements 2.1, 2.2**
    #[test]
    fn prop_tensor_roundtrip_preserves_shape(
        n_elements in arb_n_elements(),
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let db = SynaDB::new(&db_path).expect("failed to create db");
        let mut engine = TensorEngine::new(db);

        // Create tensor data with n_elements Float64 values
        let original_values: Vec<f64> = (0..n_elements)
            .map(|i| i as f64 * 0.1)
            .collect();
        let data: Vec<u8> = original_values
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // Store tensor with shape [n_elements]
        let shape = vec![n_elements];
        let stored_count = engine
            .put_tensor("tensor/", &data, &shape, DType::Float64)
            .expect("put_tensor should succeed");

        // Verify stored count matches expected
        prop_assert_eq!(
            stored_count,
            n_elements,
            "stored count should equal n_elements"
        );

        // Load tensor back
        let (loaded_data, loaded_shape) = engine
            .get_tensor("tensor/*", DType::Float64)
            .expect("get_tensor should succeed");

        // Verify shape matches
        prop_assert_eq!(
            loaded_shape.len(),
            1,
            "loaded shape should be 1D"
        );
        prop_assert_eq!(
            loaded_shape[0],
            n_elements,
            "loaded shape[0] should equal n_elements"
        );

        // Verify data size matches
        let expected_bytes = n_elements * DType::Float64.size();
        prop_assert_eq!(
            loaded_data.len(),
            expected_bytes,
            "loaded data size should match expected bytes"
        );

        // Verify values round-trip correctly
        let loaded_values: Vec<f64> = loaded_data
            .chunks(8)
            .map(|chunk| {
                let arr: [u8; 8] = chunk.try_into().expect("chunk should be 8 bytes");
                f64::from_le_bytes(arr)
            })
            .collect();

        prop_assert_eq!(
            loaded_values.len(),
            original_values.len(),
            "loaded values count should match original"
        );

        for (i, (&expected, &actual)) in original_values.iter().zip(loaded_values.iter()).enumerate() {
            prop_assert!(
                (expected - actual).abs() < 1e-10,
                "value at index {} should match: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    /// **Feature: Syna-ai-native, Property 20: Batch Tensor Shape Consistency (Multi-dtype)**
    ///
    /// For any data type, storing and loading a tensor should preserve the element count.
    ///
    /// **Validates: Requirements 2.1, 2.2**
    #[test]
    fn prop_tensor_shape_consistency_all_dtypes(
        n_elements in 10usize..500,
        dtype in arb_dtype(),
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let db = SynaDB::new(&db_path).expect("failed to create db");
        let mut engine = TensorEngine::new(db);

        // Create tensor data based on dtype
        let element_size = dtype.size();
        let data: Vec<u8> = match dtype {
            DType::Float32 => {
                (0..n_elements)
                    .flat_map(|i| (i as f32 * 0.1).to_le_bytes())
                    .collect()
            }
            DType::Float64 => {
                (0..n_elements)
                    .flat_map(|i| (i as f64 * 0.1).to_le_bytes())
                    .collect()
            }
            DType::Int32 => {
                (0..n_elements)
                    .flat_map(|i| (i as i32).to_le_bytes())
                    .collect()
            }
            DType::Int64 => {
                (0..n_elements)
                    .flat_map(|i| (i as i64).to_le_bytes())
                    .collect()
            }
        };

        // Store tensor
        let shape = vec![n_elements];
        let stored_count = engine
            .put_tensor("data/", &data, &shape, dtype)
            .expect("put_tensor should succeed");

        prop_assert_eq!(
            stored_count,
            n_elements,
            "stored count should equal n_elements for {:?}",
            dtype
        );

        // Load tensor back
        let (loaded_data, loaded_shape) = engine
            .get_tensor("data/*", dtype)
            .expect("get_tensor should succeed");

        // Verify shape
        prop_assert_eq!(
            loaded_shape[0],
            n_elements,
            "loaded shape should equal n_elements for {:?}",
            dtype
        );

        // Verify data size
        let expected_bytes = n_elements * element_size;
        prop_assert_eq!(
            loaded_data.len(),
            expected_bytes,
            "loaded data size should match for {:?}",
            dtype
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Basic unit test to verify the property test setup works.
    #[test]
    fn test_tensor_roundtrip_basic() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let db = SynaDB::new(&db_path).expect("failed to create db");
        let mut engine = TensorEngine::new(db);

        // Store 10 float64 values
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let data: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();

        let count = engine
            .put_tensor("test/", &data, &[10], DType::Float64)
            .expect("put_tensor should succeed");
        assert_eq!(count, 10);

        let (loaded_data, shape) = engine
            .get_tensor("test/*", DType::Float64)
            .expect("get_tensor should succeed");

        assert_eq!(shape, vec![10]);
        assert_eq!(loaded_data.len(), 80); // 10 * 8 bytes
    }
}
