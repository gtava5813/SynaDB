//! Property-based tests for Model Checksum Integrity.
//!
//! **Feature: Syna-ai-native, Property 21: Model Checksum Integrity**
//! **Validates: Requirements 4.3, 4.4**

use proptest::prelude::*;
use std::collections::HashMap;
use synadb::model_registry::ModelRegistry;
use tempfile::tempdir;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// **Feature: Syna-ai-native, Property 21: Model Checksum Integrity**
    ///
    /// For any model data, saving it to the registry and loading it back
    /// SHALL produce identical data and matching checksums.
    /// The checksum verification on load SHALL ensure data integrity.
    ///
    /// **Validates: Requirements 4.3, 4.4**
    #[test]
    fn prop_model_checksum_verified_on_load(
        data in prop::collection::vec(any::<u8>(), 100..10000),
    ) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("models.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        // Save model
        let version = registry.save_model(
            "test_model",
            &data,
            HashMap::new(),
        ).unwrap();

        // Load and verify checksum matches
        let (loaded_data, loaded_version) = registry.load_model(
            "test_model",
            Some(version.version),
        ).unwrap();

        prop_assert_eq!(data, loaded_data, "Loaded data should match original");
        prop_assert_eq!(version.checksum, loaded_version.checksum, "Checksums should match");
    }

    /// Test that different data produces different checksums.
    ///
    /// **Validates: Requirements 4.3**
    #[test]
    fn prop_different_data_different_checksum(
        data1 in prop::collection::vec(any::<u8>(), 100..1000),
        data2 in prop::collection::vec(any::<u8>(), 100..1000),
    ) {
        // Skip if data happens to be identical
        prop_assume!(data1 != data2);

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("models.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let v1 = registry.save_model("model", &data1, HashMap::new()).unwrap();
        let v2 = registry.save_model("model", &data2, HashMap::new()).unwrap();

        prop_assert_ne!(v1.checksum, v2.checksum,
            "Different data should produce different checksums");
    }

    /// Test that identical data produces identical checksums.
    ///
    /// **Validates: Requirements 4.3**
    #[test]
    fn prop_identical_data_identical_checksum(
        data in prop::collection::vec(any::<u8>(), 100..5000),
    ) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("models.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let v1 = registry.save_model("model", &data, HashMap::new()).unwrap();
        let v2 = registry.save_model("model", &data, HashMap::new()).unwrap();

        prop_assert_eq!(v1.checksum, v2.checksum,
            "Identical data should produce identical checksums");
    }

    /// Test checksum integrity across database reopen.
    ///
    /// **Validates: Requirements 4.3, 4.4**
    #[test]
    fn prop_checksum_persists_across_reopen(
        data in prop::collection::vec(any::<u8>(), 100..5000),
    ) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("models.db");

        let original_checksum: String;
        let original_version: u32;

        // Save model and close
        {
            let mut registry = ModelRegistry::new(&db_path).unwrap();
            let version = registry.save_model("test_model", &data, HashMap::new()).unwrap();
            original_checksum = version.checksum.clone();
            original_version = version.version;
        }

        // Reopen and verify
        {
            let mut registry = ModelRegistry::new(&db_path).unwrap();
            let (loaded_data, loaded_version) = registry.load_model(
                "test_model",
                Some(original_version),
            ).unwrap();

            prop_assert_eq!(data, loaded_data, "Data should match after reopen");
            prop_assert_eq!(original_checksum, loaded_version.checksum,
                "Checksum should match after reopen");
        }
    }
}
