// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

//! Experiment tracking for ML workflows.
//!
//! This module provides:
//! - [`ExperimentTracker`] - The main tracker struct for logging ML experiments
//! - [`Run`] - A single experiment run with parameters, metrics, and artifacts
//! - [`RunStatus`] - Status enum for tracking run state
//!
//! # Usage
//!
//! ```rust,no_run
//! use synadb::experiment::{ExperimentTracker, RunStatus};
//! use std::collections::HashMap;
//!
//! let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
//!
//! // Start a new run
//! let run_id = tracker.start_run("mnist_classifier", vec!["baseline".to_string()]).unwrap();
//!
//! // Log parameters
//! tracker.log_param(&run_id, "learning_rate", "0.001").unwrap();
//! tracker.log_param(&run_id, "batch_size", "32").unwrap();
//!
//! // Log metrics
//! tracker.log_metric(&run_id, "loss", 0.5, Some(1)).unwrap();
//! tracker.log_metric(&run_id, "accuracy", 0.85, Some(1)).unwrap();
//!
//! // End the run
//! tracker.end_run(&run_id, RunStatus::Completed).unwrap();
//! ```
//!
//! _Requirements: 5.1_

use crate::engine::SynaDB;
use crate::error::{Result, SynaError};
use crate::types::Atom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;

// =============================================================================
// Run Status Enum
// =============================================================================

/// Status of an experiment run.
///
/// Runs progress through states as they execute:
/// - `Running` - Run is currently in progress
/// - `Completed` - Run finished successfully
/// - `Failed` - Run encountered an error
/// - `Killed` - Run was manually terminated
///
/// # Examples
///
/// ```rust
/// use synadb::experiment::RunStatus;
///
/// let status = RunStatus::Running;
/// assert_eq!(status, RunStatus::Running);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RunStatus {
    /// Run is currently in progress.
    #[default]
    Running,
    /// Run finished successfully.
    Completed,
    /// Run encountered an error.
    Failed,
    /// Run was manually terminated.
    Killed,
}

impl std::fmt::Display for RunStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunStatus::Running => write!(f, "Running"),
            RunStatus::Completed => write!(f, "Completed"),
            RunStatus::Failed => write!(f, "Failed"),
            RunStatus::Killed => write!(f, "Killed"),
        }
    }
}

// =============================================================================
// Run Struct
// =============================================================================

/// A single experiment run.
///
/// Contains all information about a specific run of an experiment,
/// including parameters, metrics, and artifacts.
///
/// # Fields
///
/// - `id` - Unique identifier for the run (UUID v4)
/// - `experiment_name` - Name of the experiment this run belongs to
/// - `started_at` - Unix timestamp when the run started
/// - `ended_at` - Unix timestamp when the run ended (None if still running)
/// - `status` - Current status of the run
/// - `params` - Hyperparameters and configuration values
/// - `tags` - User-defined tags for categorization
///
/// _Requirements: 5.1, 5.2_
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// Unique identifier for the run (UUID v4).
    pub id: String,
    /// Name of the experiment this run belongs to.
    pub experiment_name: String,
    /// Unix timestamp (seconds since epoch) when the run started.
    pub started_at: u64,
    /// Unix timestamp (seconds since epoch) when the run ended.
    /// None if the run is still in progress.
    pub ended_at: Option<u64>,
    /// Current status of the run.
    pub status: RunStatus,
    /// Hyperparameters and configuration values.
    pub params: HashMap<String, String>,
    /// User-defined tags for categorization.
    pub tags: Vec<String>,
}

impl Run {
    /// Creates a new Run with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the run
    /// * `experiment_name` - Name of the experiment
    /// * `started_at` - Unix timestamp when the run started
    /// * `tags` - User-defined tags for categorization
    pub fn new(id: String, experiment_name: String, started_at: u64, tags: Vec<String>) -> Self {
        Self {
            id,
            experiment_name,
            started_at,
            ended_at: None,
            status: RunStatus::Running,
            params: HashMap::new(),
            tags,
        }
    }
}

// =============================================================================
// Run Filter
// =============================================================================

/// Filter for querying runs.
///
/// All filter fields are optional. When multiple fields are set,
/// they are combined with AND logic (all conditions must match).
///
/// # Examples
///
/// ```rust
/// use synadb::experiment::{RunFilter, RunStatus};
///
/// // Filter by experiment name
/// let filter = RunFilter {
///     experiment: Some("mnist".to_string()),
///     ..Default::default()
/// };
///
/// // Filter by status
/// let filter = RunFilter {
///     status: Some(RunStatus::Completed),
///     ..Default::default()
/// };
///
/// // Filter by tags
/// let filter = RunFilter {
///     tags: Some(vec!["baseline".to_string()]),
///     ..Default::default()
/// };
///
/// // Filter by parameter value
/// let filter = RunFilter {
///     param_filter: Some(("learning_rate".to_string(), "0.001".to_string())),
///     ..Default::default()
/// };
/// ```
///
/// _Requirements: 5.5, 5.6_
#[derive(Debug, Clone, Default)]
pub struct RunFilter {
    /// Filter by experiment name.
    pub experiment: Option<String>,
    /// Filter by run status.
    pub status: Option<RunStatus>,
    /// Filter by tags (all specified tags must be present).
    pub tags: Option<Vec<String>>,
    /// Filter by parameter value (key, value).
    pub param_filter: Option<(String, String)>,
}

// =============================================================================
// Experiment Tracker
// =============================================================================

/// Experiment tracker for logging ML experiments.
///
/// The tracker provides:
/// - Unique run ID generation (UUID v4)
/// - Parameter logging (hyperparameters, config)
/// - Metric logging with optional step numbers
/// - Artifact storage
/// - Run status management
///
/// # Storage Format
///
/// Experiments are stored in the underlying SynaDB with the following key structure:
/// - `exp/{experiment}/run/{run_id}/meta` - JSON metadata (Run struct)
/// - `exp/{experiment}/run/{run_id}/param/{key}` - Parameter values
/// - `exp/{experiment}/run/{run_id}/metric/{key}/{step}` - Metric values
/// - `exp/{experiment}/run/{run_id}/artifact/{name}` - Artifact data
///
/// # Examples
///
/// ```rust,no_run
/// use synadb::experiment::{ExperimentTracker, RunStatus};
///
/// // Create a new tracker
/// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
///
/// // Start a run with tags
/// let run_id = tracker.start_run("mnist", vec!["baseline".to_string()]).unwrap();
///
/// // Log hyperparameters
/// tracker.log_param(&run_id, "lr", "0.001").unwrap();
/// tracker.log_param(&run_id, "batch_size", "32").unwrap();
///
/// // Log metrics during training
/// for epoch in 0..10 {
///     tracker.log_metric(&run_id, "loss", 1.0 / (epoch + 1) as f64, Some(epoch as u64)).unwrap();
/// }
///
/// // End the run
/// tracker.end_run(&run_id, RunStatus::Completed).unwrap();
/// ```
///
/// _Requirements: 5.1_
pub struct ExperimentTracker {
    /// The underlying database for storage.
    db: SynaDB,
}

impl ExperimentTracker {
    /// Creates or opens an experiment tracker at the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database file
    ///
    /// # Returns
    ///
    /// * `Ok(ExperimentTracker)` - The opened tracker
    /// * `Err(SynaError)` - If the database cannot be opened
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// ```
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = SynaDB::new(path)?;
        Ok(Self { db })
    }

    /// Returns a reference to the underlying database.
    ///
    /// This is useful for advanced operations or debugging.
    pub fn db(&self) -> &SynaDB {
        &self.db
    }

    /// Returns a mutable reference to the underlying database.
    ///
    /// This is useful for advanced operations or debugging.
    pub fn db_mut(&mut self) -> &mut SynaDB {
        &mut self.db
    }

    /// Starts a new experiment run.
    ///
    /// Creates a unique run ID and initializes the run metadata.
    ///
    /// # Arguments
    ///
    /// * `experiment` - Name of the experiment
    /// * `tags` - Optional tags for categorization
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - The unique run ID
    /// * `Err(SynaError)` - If storage fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// let run_id = tracker.start_run("mnist", vec!["baseline".to_string()]).unwrap();
    /// println!("Started run: {}", run_id);
    /// ```
    ///
    /// _Requirements: 5.1_
    pub fn start_run(&mut self, experiment: &str, tags: Vec<String>) -> Result<String> {
        // Generate unique run ID
        let run_id = Uuid::new_v4().to_string();

        // Get current timestamp
        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Create run metadata
        let run = Run::new(run_id.clone(), experiment.to_string(), started_at, tags);

        // Store run metadata
        let meta_key = format!("exp/{}/run/{}/meta", experiment, run_id);
        let meta_json =
            serde_json::to_string(&run).map_err(|e| SynaError::InvalidPath(e.to_string()))?;
        self.db.append(&meta_key, Atom::Text(meta_json))?;

        Ok(run_id)
    }

    /// Logs a parameter (hyperparameter or config value) for a run.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run ID
    /// * `key` - Parameter name
    /// * `value` - Parameter value (as string)
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the parameter was logged successfully
    /// * `Err(SynaError::RunNotFound)` - If the run doesn't exist
    /// * `Err(SynaError::RunAlreadyEnded)` - If the run has already ended
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// let run_id = tracker.start_run("mnist", vec![]).unwrap();
    ///
    /// tracker.log_param(&run_id, "learning_rate", "0.001").unwrap();
    /// tracker.log_param(&run_id, "batch_size", "32").unwrap();
    /// ```
    ///
    /// _Requirements: 5.2_
    pub fn log_param(&mut self, run_id: &str, key: &str, value: &str) -> Result<()> {
        // Get run metadata to verify it exists and is running
        let run = self.get_run_internal(run_id)?;

        if run.status != RunStatus::Running {
            return Err(SynaError::RunAlreadyEnded(run_id.to_string()));
        }

        // Store parameter
        let param_key = format!("exp/{}/run/{}/param/{}", run.experiment_name, run_id, key);
        self.db.append(&param_key, Atom::Text(value.to_string()))?;

        // Update run metadata with new param
        let mut updated_run = run;
        updated_run
            .params
            .insert(key.to_string(), value.to_string());

        let meta_key = format!("exp/{}/run/{}/meta", updated_run.experiment_name, run_id);
        let meta_json = serde_json::to_string(&updated_run)
            .map_err(|e| SynaError::InvalidPath(e.to_string()))?;
        self.db.append(&meta_key, Atom::Text(meta_json))?;

        Ok(())
    }

    /// Logs a metric value for a run.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run ID
    /// * `key` - Metric name
    /// * `value` - Metric value
    /// * `step` - Optional step number for time-series metrics
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the metric was logged successfully
    /// * `Err(SynaError::RunNotFound)` - If the run doesn't exist
    /// * `Err(SynaError::RunAlreadyEnded)` - If the run has already ended
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// let run_id = tracker.start_run("mnist", vec![]).unwrap();
    ///
    /// // Log metrics with step numbers
    /// for epoch in 0..10 {
    ///     tracker.log_metric(&run_id, "loss", 1.0 / (epoch + 1) as f64, Some(epoch as u64)).unwrap();
    ///     tracker.log_metric(&run_id, "accuracy", 0.5 + 0.05 * epoch as f64, Some(epoch as u64)).unwrap();
    /// }
    /// ```
    ///
    /// _Requirements: 5.3_
    pub fn log_metric(
        &mut self,
        run_id: &str,
        key: &str,
        value: f64,
        step: Option<u64>,
    ) -> Result<()> {
        // Get run metadata to verify it exists and is running
        let run = self.get_run_internal(run_id)?;

        if run.status != RunStatus::Running {
            return Err(SynaError::RunAlreadyEnded(run_id.to_string()));
        }

        // Store metric with step number
        let step_num = step.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64
        });

        let metric_key = format!(
            "exp/{}/run/{}/metric/{}/{}",
            run.experiment_name, run_id, key, step_num
        );
        self.db.append(&metric_key, Atom::Float(value))?;

        Ok(())
    }

    /// Logs an artifact (file, plot, model) for a run.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run ID
    /// * `name` - Artifact name
    /// * `data` - Artifact data as bytes
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the artifact was logged successfully
    /// * `Err(SynaError::RunNotFound)` - If the run doesn't exist
    /// * `Err(SynaError::RunAlreadyEnded)` - If the run has already ended
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// let run_id = tracker.start_run("mnist", vec![]).unwrap();
    ///
    /// // Log a model artifact
    /// let model_data = vec![0u8; 1024]; // Your model bytes
    /// tracker.log_artifact(&run_id, "model.pt", &model_data).unwrap();
    /// ```
    ///
    /// _Requirements: 5.4_
    pub fn log_artifact(&mut self, run_id: &str, name: &str, data: &[u8]) -> Result<()> {
        // Get run metadata to verify it exists and is running
        let run = self.get_run_internal(run_id)?;

        if run.status != RunStatus::Running {
            return Err(SynaError::RunAlreadyEnded(run_id.to_string()));
        }

        // Store artifact
        let artifact_key = format!(
            "exp/{}/run/{}/artifact/{}",
            run.experiment_name, run_id, name
        );
        self.db.append(&artifact_key, Atom::Bytes(data.to_vec()))?;

        Ok(())
    }

    /// Gets an artifact by name for a run.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run ID
    /// * `name` - Artifact name
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Vec<u8>))` - The artifact data if it exists
    /// * `Ok(None)` - If the artifact doesn't exist
    /// * `Err(SynaError::RunNotFound)` - If the run doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// let run_id = tracker.start_run("mnist", vec![]).unwrap();
    ///
    /// // Log an artifact
    /// let model_data = vec![0u8; 1024];
    /// tracker.log_artifact(&run_id, "model.pt", &model_data).unwrap();
    ///
    /// // Retrieve the artifact
    /// if let Some(data) = tracker.get_artifact(&run_id, "model.pt").unwrap() {
    ///     println!("Retrieved {} bytes", data.len());
    /// }
    /// ```
    ///
    /// _Requirements: 5.4_
    pub fn get_artifact(&mut self, run_id: &str, name: &str) -> Result<Option<Vec<u8>>> {
        // Get run to verify it exists and get experiment name
        let run = self.get_run_internal(run_id)?;

        let artifact_key = format!(
            "exp/{}/run/{}/artifact/{}",
            run.experiment_name, run_id, name
        );

        match self.db.get(&artifact_key)? {
            Some(Atom::Bytes(data)) => Ok(Some(data)),
            _ => Ok(None),
        }
    }

    /// Lists all artifacts for a run.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run ID
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - List of artifact names
    /// * `Err(SynaError::RunNotFound)` - If the run doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// let run_id = tracker.start_run("mnist", vec![]).unwrap();
    ///
    /// // Log some artifacts
    /// tracker.log_artifact(&run_id, "model.pt", &[1, 2, 3]).unwrap();
    /// tracker.log_artifact(&run_id, "config.json", &[4, 5, 6]).unwrap();
    ///
    /// // List all artifacts
    /// let artifacts = tracker.list_artifacts(&run_id).unwrap();
    /// for name in artifacts {
    ///     println!("Artifact: {}", name);
    /// }
    /// ```
    ///
    /// _Requirements: 5.4_
    pub fn list_artifacts(&mut self, run_id: &str) -> Result<Vec<String>> {
        // Get run to verify it exists and get experiment name
        let run = self.get_run_internal(run_id)?;

        let prefix = format!("exp/{}/run/{}/artifact/", run.experiment_name, run_id);

        Ok(self
            .db
            .keys()
            .into_iter()
            .filter(|k| k.starts_with(&prefix))
            .filter_map(|k| k.strip_prefix(&prefix).map(|s| s.to_string()))
            .collect())
    }

    /// Ends a run with the given status.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run ID
    /// * `status` - Final status of the run
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the run was ended successfully
    /// * `Err(SynaError::RunNotFound)` - If the run doesn't exist
    /// * `Err(SynaError::RunAlreadyEnded)` - If the run has already ended
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::{ExperimentTracker, RunStatus};
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// let run_id = tracker.start_run("mnist", vec![]).unwrap();
    ///
    /// // ... do training ...
    ///
    /// tracker.end_run(&run_id, RunStatus::Completed).unwrap();
    /// ```
    ///
    /// _Requirements: 5.8_
    pub fn end_run(&mut self, run_id: &str, status: RunStatus) -> Result<()> {
        // Get run metadata
        let mut run = self.get_run_internal(run_id)?;

        if run.status != RunStatus::Running {
            return Err(SynaError::RunAlreadyEnded(run_id.to_string()));
        }

        // Update run metadata
        run.status = status;
        run.ended_at = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );

        // Store updated metadata
        let meta_key = format!("exp/{}/run/{}/meta", run.experiment_name, run_id);
        let meta_json =
            serde_json::to_string(&run).map_err(|e| SynaError::InvalidPath(e.to_string()))?;
        self.db.append(&meta_key, Atom::Text(meta_json))?;

        Ok(())
    }

    /// Gets a run by ID.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run ID
    ///
    /// # Returns
    ///
    /// * `Ok(Run)` - The run metadata
    /// * `Err(SynaError::RunNotFound)` - If the run doesn't exist
    pub fn get_run(&mut self, run_id: &str) -> Result<Run> {
        self.get_run_internal(run_id)
    }

    /// Internal method to get run metadata by searching all experiments.
    fn get_run_internal(&mut self, run_id: &str) -> Result<Run> {
        // Search for the run across all experiments
        let prefix = format!("/run/{}/meta", run_id);

        for key in self.db.keys() {
            if key.contains(&prefix) {
                if let Some(Atom::Text(json)) = self.db.get(&key)? {
                    let run: Run = serde_json::from_str(&json)
                        .map_err(|e| SynaError::InvalidPath(e.to_string()))?;
                    return Ok(run);
                }
            }
        }

        Err(SynaError::RunNotFound(run_id.to_string()))
    }

    /// Lists all runs for an experiment.
    ///
    /// # Arguments
    ///
    /// * `experiment` - Name of the experiment
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Run>)` - List of all runs sorted by start time
    /// * `Err(SynaError)` - If reading from the database fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    ///
    /// let runs = tracker.list_runs("mnist").unwrap();
    /// for run in runs {
    ///     println!("Run {}: status={}", run.id, run.status);
    /// }
    /// ```
    ///
    /// _Requirements: 5.5_
    pub fn list_runs(&mut self, experiment: &str) -> Result<Vec<Run>> {
        let prefix = format!("exp/{}/run/", experiment);
        let suffix = "/meta";
        let mut runs = Vec::new();

        for key in self.db.keys() {
            if key.starts_with(&prefix) && key.ends_with(suffix) {
                if let Some(Atom::Text(json)) = self.db.get(&key)? {
                    let run: Run = serde_json::from_str(&json)
                        .map_err(|e| SynaError::InvalidPath(e.to_string()))?;
                    runs.push(run);
                }
            }
        }

        // Sort by start time
        runs.sort_by_key(|r| r.started_at);
        Ok(runs)
    }

    /// Gets all parameters for a run.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run ID
    ///
    /// # Returns
    ///
    /// * `Ok(HashMap<String, String>)` - Map of parameter name to value
    /// * `Err(SynaError::RunNotFound)` - If the run doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// let run_id = tracker.start_run("mnist", vec![]).unwrap();
    ///
    /// tracker.log_param(&run_id, "learning_rate", "0.001").unwrap();
    /// tracker.log_param(&run_id, "batch_size", "32").unwrap();
    ///
    /// // Get all parameters
    /// let params = tracker.get_params(&run_id).unwrap();
    /// for (key, value) in params {
    ///     println!("{}: {}", key, value);
    /// }
    /// ```
    ///
    /// _Requirements: 5.2_
    pub fn get_params(&mut self, run_id: &str) -> Result<HashMap<String, String>> {
        // Get run to verify it exists and get experiment name
        let run = self.get_run_internal(run_id)?;

        let prefix = format!("exp/{}/run/{}/param/", run.experiment_name, run_id);
        let mut params = HashMap::new();

        for key in self.db.keys() {
            if key.starts_with(&prefix) {
                if let Some(param_name) = key.strip_prefix(&prefix) {
                    if let Some(Atom::Text(value)) = self.db.get(&key)? {
                        params.insert(param_name.to_string(), value);
                    }
                }
            }
        }

        Ok(params)
    }

    /// Gets a specific parameter for a run.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run ID
    /// * `param_name` - Name of the parameter to retrieve
    ///
    /// # Returns
    ///
    /// * `Ok(Option<String>)` - The parameter value if it exists
    /// * `Err(SynaError::RunNotFound)` - If the run doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// let run_id = tracker.start_run("mnist", vec![]).unwrap();
    ///
    /// tracker.log_param(&run_id, "learning_rate", "0.001").unwrap();
    ///
    /// // Get specific parameter
    /// if let Some(lr) = tracker.get_param(&run_id, "learning_rate").unwrap() {
    ///     println!("Learning rate: {}", lr);
    /// }
    /// ```
    ///
    /// _Requirements: 5.2_
    pub fn get_param(&mut self, run_id: &str, param_name: &str) -> Result<Option<String>> {
        // Get run to verify it exists and get experiment name
        let run = self.get_run_internal(run_id)?;

        let param_key = format!(
            "exp/{}/run/{}/param/{}",
            run.experiment_name, run_id, param_name
        );

        if let Some(Atom::Text(value)) = self.db.get(&param_key)? {
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    /// Gets a specific metric for a run.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run ID
    /// * `metric_name` - Name of the metric to retrieve
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<(u64, f64)>)` - List of (step, value) pairs sorted by step
    /// * `Err(SynaError::RunNotFound)` - If the run doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// let run_id = tracker.start_run("mnist", vec![]).unwrap();
    ///
    /// // Log some metrics
    /// tracker.log_metric(&run_id, "loss", 0.5, Some(1)).unwrap();
    /// tracker.log_metric(&run_id, "loss", 0.3, Some(2)).unwrap();
    ///
    /// // Get specific metric
    /// let loss_values = tracker.get_metric(&run_id, "loss").unwrap();
    /// for (step, value) in loss_values {
    ///     println!("Step {}: loss = {}", step, value);
    /// }
    /// ```
    ///
    /// _Requirements: 5.3_
    pub fn get_metric(&mut self, run_id: &str, metric_name: &str) -> Result<Vec<(u64, f64)>> {
        // Get run to verify it exists and get experiment name
        let run = self.get_run_internal(run_id)?;

        let prefix = format!(
            "exp/{}/run/{}/metric/{}/",
            run.experiment_name, run_id, metric_name
        );

        let mut metrics = Vec::new();
        for key in self.db.keys() {
            if key.starts_with(&prefix) {
                if let Some(step_str) = key.strip_prefix(&prefix) {
                    if let Ok(step) = step_str.parse::<u64>() {
                        if let Some(Atom::Float(v)) = self.db.get(&key)? {
                            metrics.push((step, v));
                        }
                    }
                }
            }
        }

        // Sort by step
        metrics.sort_by_key(|(step, _)| *step);
        Ok(metrics)
    }

    /// Gets all metrics for a run.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run ID
    ///
    /// # Returns
    ///
    /// * `Ok(HashMap<String, Vec<(u64, f64)>>)` - Map of metric name to (step, value) pairs
    /// * `Err(SynaError::RunNotFound)` - If the run doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::ExperimentTracker;
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    /// let run_id = tracker.start_run("mnist", vec![]).unwrap();
    ///
    /// // Log some metrics
    /// tracker.log_metric(&run_id, "loss", 0.5, Some(1)).unwrap();
    /// tracker.log_metric(&run_id, "loss", 0.3, Some(2)).unwrap();
    ///
    /// // Get all metrics
    /// let metrics = tracker.get_all_metrics(&run_id).unwrap();
    /// if let Some(loss_values) = metrics.get("loss") {
    ///     for (step, value) in loss_values {
    ///         println!("Step {}: loss = {}", step, value);
    ///     }
    /// }
    /// ```
    ///
    /// _Requirements: 5.3_
    pub fn get_all_metrics(&mut self, run_id: &str) -> Result<HashMap<String, Vec<(u64, f64)>>> {
        // Get run to verify it exists and get experiment name
        let run = self.get_run_internal(run_id)?;

        let prefix = format!("exp/{}/run/{}/metric/", run.experiment_name, run_id);
        let mut metrics: HashMap<String, Vec<(u64, f64)>> = HashMap::new();

        for key in self.db.keys() {
            if key.starts_with(&prefix) {
                // Parse metric name and step from key
                let suffix = key.strip_prefix(&prefix).unwrap_or("");
                let parts: Vec<&str> = suffix.split('/').collect();

                if parts.len() == 2 {
                    let metric_name = parts[0];
                    if let Ok(step) = parts[1].parse::<u64>() {
                        if let Some(Atom::Float(value)) = self.db.get(&key)? {
                            metrics
                                .entry(metric_name.to_string())
                                .or_default()
                                .push((step, value));
                        }
                    }
                }
            }
        }

        // Sort each metric's values by step
        for values in metrics.values_mut() {
            values.sort_by_key(|(step, _)| *step);
        }

        Ok(metrics)
    }

    /// Query runs with optional filtering.
    ///
    /// Returns all runs that match the specified filter criteria.
    /// Results are sorted by start time (newest first).
    ///
    /// # Arguments
    ///
    /// * `filter` - Filter criteria for querying runs
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Run>)` - List of matching runs sorted by start time (newest first)
    /// * `Err(SynaError)` - If reading from the database fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::experiment::{ExperimentTracker, RunFilter, RunStatus};
    ///
    /// let mut tracker = ExperimentTracker::new("experiments.db").unwrap();
    ///
    /// // Query all runs for an experiment
    /// let filter = RunFilter {
    ///     experiment: Some("mnist".to_string()),
    ///     ..Default::default()
    /// };
    /// let runs = tracker.query_runs(filter).unwrap();
    ///
    /// // Query completed runs
    /// let filter = RunFilter {
    ///     status: Some(RunStatus::Completed),
    ///     ..Default::default()
    /// };
    /// let completed_runs = tracker.query_runs(filter).unwrap();
    ///
    /// // Query runs with specific tags
    /// let filter = RunFilter {
    ///     tags: Some(vec!["baseline".to_string()]),
    ///     ..Default::default()
    /// };
    /// let tagged_runs = tracker.query_runs(filter).unwrap();
    ///
    /// // Query runs with specific parameter value
    /// let filter = RunFilter {
    ///     param_filter: Some(("learning_rate".to_string(), "0.001".to_string())),
    ///     ..Default::default()
    /// };
    /// let filtered_runs = tracker.query_runs(filter).unwrap();
    /// ```
    ///
    /// _Requirements: 5.5, 5.6_
    pub fn query_runs(&mut self, filter: RunFilter) -> Result<Vec<Run>> {
        let mut runs = Vec::new();

        // Find all run metadata keys
        for key in self.db.keys() {
            if key.contains("/run/") && key.ends_with("/meta") {
                if let Some(Atom::Text(json)) = self.db.get(&key)? {
                    if let Ok(run) = serde_json::from_str::<Run>(&json) {
                        // Apply filters
                        if let Some(ref exp) = filter.experiment {
                            if &run.experiment_name != exp {
                                continue;
                            }
                        }
                        if let Some(status) = filter.status {
                            if run.status != status {
                                continue;
                            }
                        }
                        if let Some(ref tags) = filter.tags {
                            if !tags.iter().all(|t| run.tags.contains(t)) {
                                continue;
                            }
                        }
                        if let Some((ref param_key, ref param_value)) = filter.param_filter {
                            match run.params.get(param_key) {
                                Some(value) if value == param_value => {}
                                _ => continue,
                            }
                        }
                        runs.push(run);
                    }
                }
            }
        }

        // Sort by start time (newest first)
        runs.sort_by(|a: &Run, b: &Run| b.started_at.cmp(&a.started_at));
        Ok(runs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_run_status_default() {
        let status = RunStatus::default();
        assert_eq!(status, RunStatus::Running);
    }

    #[test]
    fn test_run_status_display() {
        assert_eq!(format!("{}", RunStatus::Running), "Running");
        assert_eq!(format!("{}", RunStatus::Completed), "Completed");
        assert_eq!(format!("{}", RunStatus::Failed), "Failed");
        assert_eq!(format!("{}", RunStatus::Killed), "Killed");
    }

    #[test]
    fn test_run_new() {
        let run = Run::new(
            "run_123".to_string(),
            "mnist".to_string(),
            1234567890,
            vec!["baseline".to_string()],
        );

        assert_eq!(run.id, "run_123");
        assert_eq!(run.experiment_name, "mnist");
        assert_eq!(run.started_at, 1234567890);
        assert_eq!(run.ended_at, None);
        assert_eq!(run.status, RunStatus::Running);
        assert!(run.params.is_empty());
        assert_eq!(run.tags, vec!["baseline".to_string()]);
    }

    #[test]
    fn test_experiment_tracker_new() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_tracker.db");

        let tracker = ExperimentTracker::new(&db_path);
        assert!(tracker.is_ok());
    }

    #[test]
    fn test_start_run() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_start_run.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker
            .start_run("mnist", vec!["baseline".to_string()])
            .unwrap();

        // Verify run ID is a valid UUID
        assert!(uuid::Uuid::parse_str(&run_id).is_ok());

        // Verify run can be retrieved
        let run = tracker.get_run(&run_id).unwrap();
        assert_eq!(run.id, run_id);
        assert_eq!(run.experiment_name, "mnist");
        assert_eq!(run.status, RunStatus::Running);
        assert_eq!(run.tags, vec!["baseline".to_string()]);
    }

    #[test]
    fn test_log_param() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_log_param.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("mnist", vec![]).unwrap();

        // Log parameters
        tracker
            .log_param(&run_id, "learning_rate", "0.001")
            .unwrap();
        tracker.log_param(&run_id, "batch_size", "32").unwrap();

        // Verify parameters are stored
        let run = tracker.get_run(&run_id).unwrap();
        assert_eq!(run.params.get("learning_rate"), Some(&"0.001".to_string()));
        assert_eq!(run.params.get("batch_size"), Some(&"32".to_string()));
    }

    #[test]
    fn test_log_metric() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_log_metric.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("mnist", vec![]).unwrap();

        // Log metrics with step numbers
        tracker.log_metric(&run_id, "loss", 0.5, Some(1)).unwrap();
        tracker.log_metric(&run_id, "loss", 0.3, Some(2)).unwrap();
        tracker
            .log_metric(&run_id, "accuracy", 0.85, Some(1))
            .unwrap();

        // Verify metrics can be retrieved using get_all_metrics
        let metrics = tracker.get_all_metrics(&run_id).unwrap();

        let loss_values = metrics.get("loss").unwrap();
        assert_eq!(loss_values.len(), 2);
        assert_eq!(loss_values[0], (1, 0.5));
        assert_eq!(loss_values[1], (2, 0.3));

        let accuracy_values = metrics.get("accuracy").unwrap();
        assert_eq!(accuracy_values.len(), 1);
        assert_eq!(accuracy_values[0], (1, 0.85));
    }

    #[test]
    fn test_get_metric() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_get_metric.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("mnist", vec![]).unwrap();

        // Log metrics with step numbers
        tracker.log_metric(&run_id, "loss", 0.5, Some(1)).unwrap();
        tracker.log_metric(&run_id, "loss", 0.3, Some(2)).unwrap();
        tracker.log_metric(&run_id, "loss", 0.1, Some(3)).unwrap();
        tracker
            .log_metric(&run_id, "accuracy", 0.85, Some(1))
            .unwrap();

        // Verify specific metric can be retrieved
        let loss_values = tracker.get_metric(&run_id, "loss").unwrap();
        assert_eq!(loss_values.len(), 3);
        assert_eq!(loss_values[0], (1, 0.5));
        assert_eq!(loss_values[1], (2, 0.3));
        assert_eq!(loss_values[2], (3, 0.1));

        // Verify another metric
        let accuracy_values = tracker.get_metric(&run_id, "accuracy").unwrap();
        assert_eq!(accuracy_values.len(), 1);
        assert_eq!(accuracy_values[0], (1, 0.85));

        // Verify non-existent metric returns empty
        let empty_values = tracker.get_metric(&run_id, "nonexistent").unwrap();
        assert!(empty_values.is_empty());
    }

    #[test]
    fn test_get_params() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_get_params.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("mnist", vec![]).unwrap();

        // Log parameters
        tracker
            .log_param(&run_id, "learning_rate", "0.001")
            .unwrap();
        tracker.log_param(&run_id, "batch_size", "32").unwrap();
        tracker.log_param(&run_id, "optimizer", "adam").unwrap();

        // Verify all parameters can be retrieved
        let params = tracker.get_params(&run_id).unwrap();
        assert_eq!(params.len(), 3);
        assert_eq!(params.get("learning_rate"), Some(&"0.001".to_string()));
        assert_eq!(params.get("batch_size"), Some(&"32".to_string()));
        assert_eq!(params.get("optimizer"), Some(&"adam".to_string()));
    }

    #[test]
    fn test_get_param() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_get_param.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("mnist", vec![]).unwrap();

        // Log parameters
        tracker
            .log_param(&run_id, "learning_rate", "0.001")
            .unwrap();
        tracker.log_param(&run_id, "batch_size", "32").unwrap();

        // Verify specific parameter can be retrieved
        let lr = tracker.get_param(&run_id, "learning_rate").unwrap();
        assert_eq!(lr, Some("0.001".to_string()));

        let batch_size = tracker.get_param(&run_id, "batch_size").unwrap();
        assert_eq!(batch_size, Some("32".to_string()));

        // Verify non-existent parameter returns None
        let nonexistent = tracker.get_param(&run_id, "nonexistent").unwrap();
        assert_eq!(nonexistent, None);
    }

    #[test]
    fn test_log_artifact() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_log_artifact.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("mnist", vec![]).unwrap();

        // Log an artifact
        let artifact_data = vec![0u8, 1, 2, 3, 4, 5];
        tracker
            .log_artifact(&run_id, "model.pt", &artifact_data)
            .unwrap();

        // Verify artifact is stored (by checking the key exists)
        let run = tracker.get_run(&run_id).unwrap();
        let artifact_key = format!(
            "exp/{}/run/{}/artifact/model.pt",
            run.experiment_name, run_id
        );

        let stored = tracker.db_mut().get(&artifact_key).unwrap();
        assert!(matches!(stored, Some(Atom::Bytes(data)) if data == artifact_data));
    }

    #[test]
    fn test_get_artifact() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_get_artifact.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("mnist", vec![]).unwrap();

        // Log an artifact
        let artifact_data = vec![0u8, 1, 2, 3, 4, 5];
        tracker
            .log_artifact(&run_id, "model.pt", &artifact_data)
            .unwrap();

        // Retrieve the artifact
        let retrieved = tracker.get_artifact(&run_id, "model.pt").unwrap();
        assert_eq!(retrieved, Some(artifact_data));

        // Try to get non-existent artifact
        let nonexistent = tracker.get_artifact(&run_id, "nonexistent.pt").unwrap();
        assert_eq!(nonexistent, None);
    }

    #[test]
    fn test_list_artifacts() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_list_artifacts.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("mnist", vec![]).unwrap();

        // Log multiple artifacts
        tracker
            .log_artifact(&run_id, "model.pt", &[1, 2, 3])
            .unwrap();
        tracker
            .log_artifact(&run_id, "config.json", &[4, 5, 6])
            .unwrap();
        tracker
            .log_artifact(&run_id, "weights.bin", &[7, 8, 9])
            .unwrap();

        // List all artifacts
        let artifacts = tracker.list_artifacts(&run_id).unwrap();
        assert_eq!(artifacts.len(), 3);
        assert!(artifacts.contains(&"model.pt".to_string()));
        assert!(artifacts.contains(&"config.json".to_string()));
        assert!(artifacts.contains(&"weights.bin".to_string()));
    }

    #[test]
    fn test_list_artifacts_empty() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_list_artifacts_empty.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("mnist", vec![]).unwrap();

        // List artifacts when none exist
        let artifacts = tracker.list_artifacts(&run_id).unwrap();
        assert!(artifacts.is_empty());
    }

    #[test]
    fn test_end_run() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_end_run.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("mnist", vec![]).unwrap();

        // End the run
        tracker.end_run(&run_id, RunStatus::Completed).unwrap();

        // Verify run status is updated
        let run = tracker.get_run(&run_id).unwrap();
        assert_eq!(run.status, RunStatus::Completed);
        assert!(run.ended_at.is_some());
    }

    #[test]
    fn test_cannot_log_to_ended_run() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_ended_run.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("mnist", vec![]).unwrap();

        // End the run
        tracker.end_run(&run_id, RunStatus::Completed).unwrap();

        // Try to log param - should fail
        let result = tracker.log_param(&run_id, "key", "value");
        assert!(matches!(result, Err(SynaError::RunAlreadyEnded(_))));

        // Try to log metric - should fail
        let result = tracker.log_metric(&run_id, "loss", 0.5, Some(1));
        assert!(matches!(result, Err(SynaError::RunAlreadyEnded(_))));

        // Try to log artifact - should fail
        let result = tracker.log_artifact(&run_id, "file", &[1, 2, 3]);
        assert!(matches!(result, Err(SynaError::RunAlreadyEnded(_))));

        // Try to end again - should fail
        let result = tracker.end_run(&run_id, RunStatus::Failed);
        assert!(matches!(result, Err(SynaError::RunAlreadyEnded(_))));
    }

    #[test]
    fn test_list_runs() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_list_runs.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();

        // Start multiple runs
        let run1 = tracker.start_run("mnist", vec!["v1".to_string()]).unwrap();
        let run2 = tracker.start_run("mnist", vec!["v2".to_string()]).unwrap();
        let run3 = tracker.start_run("cifar", vec![]).unwrap(); // Different experiment

        // List runs for mnist
        let mnist_runs = tracker.list_runs("mnist").unwrap();
        assert_eq!(mnist_runs.len(), 2);
        assert!(mnist_runs.iter().any(|r| r.id == run1));
        assert!(mnist_runs.iter().any(|r| r.id == run2));

        // List runs for cifar
        let cifar_runs = tracker.list_runs("cifar").unwrap();
        assert_eq!(cifar_runs.len(), 1);
        assert_eq!(cifar_runs[0].id, run3);
    }

    #[test]
    fn test_run_not_found() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_not_found.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();

        // Try to get a non-existent run
        let result = tracker.get_run("nonexistent");
        assert!(matches!(result, Err(SynaError::RunNotFound(_))));
    }

    #[test]
    fn test_run_status_serialization() {
        // Test that RunStatus can be serialized and deserialized
        let status = RunStatus::Completed;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: RunStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_run_serialization() {
        let run = Run::new(
            "run_123".to_string(),
            "mnist".to_string(),
            1234567890,
            vec!["baseline".to_string()],
        );

        let serialized = serde_json::to_string(&run).unwrap();
        let deserialized: Run = serde_json::from_str(&serialized).unwrap();

        assert_eq!(run.id, deserialized.id);
        assert_eq!(run.experiment_name, deserialized.experiment_name);
        assert_eq!(run.status, deserialized.status);
    }

    #[test]
    fn test_query_runs_no_filter() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_query_no_filter.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();

        // Start multiple runs
        let _run1 = tracker.start_run("mnist", vec![]).unwrap();
        let _run2 = tracker.start_run("mnist", vec![]).unwrap();
        let _run3 = tracker.start_run("cifar", vec![]).unwrap();

        // Query all runs with no filter
        let filter = RunFilter::default();
        let runs = tracker.query_runs(filter).unwrap();
        assert_eq!(runs.len(), 3);
    }

    #[test]
    fn test_query_runs_by_experiment() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_query_by_exp.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();

        // Start multiple runs
        let run1 = tracker.start_run("mnist", vec![]).unwrap();
        let run2 = tracker.start_run("mnist", vec![]).unwrap();
        let _run3 = tracker.start_run("cifar", vec![]).unwrap();

        // Query runs for mnist experiment
        let filter = RunFilter {
            experiment: Some("mnist".to_string()),
            ..Default::default()
        };
        let runs = tracker.query_runs(filter).unwrap();
        assert_eq!(runs.len(), 2);
        assert!(runs.iter().any(|r| r.id == run1));
        assert!(runs.iter().any(|r| r.id == run2));
    }

    #[test]
    fn test_query_runs_by_status() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_query_by_status.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();

        // Start multiple runs
        let run1 = tracker.start_run("mnist", vec![]).unwrap();
        let run2 = tracker.start_run("mnist", vec![]).unwrap();
        let run3 = tracker.start_run("mnist", vec![]).unwrap();

        // End some runs with different statuses
        tracker.end_run(&run1, RunStatus::Completed).unwrap();
        tracker.end_run(&run2, RunStatus::Failed).unwrap();
        // run3 stays Running

        // Query completed runs
        let filter = RunFilter {
            status: Some(RunStatus::Completed),
            ..Default::default()
        };
        let runs = tracker.query_runs(filter).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, run1);

        // Query running runs
        let filter = RunFilter {
            status: Some(RunStatus::Running),
            ..Default::default()
        };
        let runs = tracker.query_runs(filter).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, run3);
    }

    #[test]
    fn test_query_runs_by_tags() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_query_by_tags.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();

        // Start runs with different tags
        let run1 = tracker
            .start_run("mnist", vec!["baseline".to_string()])
            .unwrap();
        let run2 = tracker
            .start_run("mnist", vec!["baseline".to_string(), "v2".to_string()])
            .unwrap();
        let _run3 = tracker
            .start_run("mnist", vec!["experimental".to_string()])
            .unwrap();

        // Query runs with "baseline" tag
        let filter = RunFilter {
            tags: Some(vec!["baseline".to_string()]),
            ..Default::default()
        };
        let runs = tracker.query_runs(filter).unwrap();
        assert_eq!(runs.len(), 2);
        assert!(runs.iter().any(|r| r.id == run1));
        assert!(runs.iter().any(|r| r.id == run2));

        // Query runs with both "baseline" and "v2" tags
        let filter = RunFilter {
            tags: Some(vec!["baseline".to_string(), "v2".to_string()]),
            ..Default::default()
        };
        let runs = tracker.query_runs(filter).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, run2);
    }

    #[test]
    fn test_query_runs_by_param() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_query_by_param.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();

        // Start runs with different parameters
        let run1 = tracker.start_run("mnist", vec![]).unwrap();
        tracker.log_param(&run1, "learning_rate", "0.001").unwrap();

        let run2 = tracker.start_run("mnist", vec![]).unwrap();
        tracker.log_param(&run2, "learning_rate", "0.01").unwrap();

        let run3 = tracker.start_run("mnist", vec![]).unwrap();
        tracker.log_param(&run3, "learning_rate", "0.001").unwrap();

        // Query runs with learning_rate = 0.001
        let filter = RunFilter {
            param_filter: Some(("learning_rate".to_string(), "0.001".to_string())),
            ..Default::default()
        };
        let runs = tracker.query_runs(filter).unwrap();
        assert_eq!(runs.len(), 2);
        assert!(runs.iter().any(|r| r.id == run1));
        assert!(runs.iter().any(|r| r.id == run3));
    }

    #[test]
    fn test_query_runs_combined_filters() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_query_combined.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();

        // Start runs with various attributes
        let run1 = tracker
            .start_run("mnist", vec!["baseline".to_string()])
            .unwrap();
        tracker.log_param(&run1, "lr", "0.001").unwrap();
        tracker.end_run(&run1, RunStatus::Completed).unwrap();

        let run2 = tracker
            .start_run("mnist", vec!["baseline".to_string()])
            .unwrap();
        tracker.log_param(&run2, "lr", "0.001").unwrap();
        // run2 stays Running

        let run3 = tracker
            .start_run("cifar", vec!["baseline".to_string()])
            .unwrap();
        tracker.log_param(&run3, "lr", "0.001").unwrap();
        tracker.end_run(&run3, RunStatus::Completed).unwrap();

        // Query: mnist + completed + baseline tag + lr=0.001
        let filter = RunFilter {
            experiment: Some("mnist".to_string()),
            status: Some(RunStatus::Completed),
            tags: Some(vec!["baseline".to_string()]),
            param_filter: Some(("lr".to_string(), "0.001".to_string())),
        };
        let runs = tracker.query_runs(filter).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, run1);
    }

    #[test]
    fn test_query_runs_sorted_by_start_time() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_query_sorted.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();

        // Start runs (they will have increasing timestamps)
        // Use 1 second sleep to ensure different timestamps (timestamps are in seconds)
        let run1 = tracker.start_run("mnist", vec![]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(1));
        let run2 = tracker.start_run("mnist", vec![]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(1));
        let run3 = tracker.start_run("mnist", vec![]).unwrap();

        // Query all runs
        let filter = RunFilter::default();
        let runs = tracker.query_runs(filter).unwrap();

        // Should be sorted newest first
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].id, run3); // newest
        assert_eq!(runs[1].id, run2);
        assert_eq!(runs[2].id, run1); // oldest
    }

    #[test]
    fn test_run_filter_default() {
        let filter = RunFilter::default();
        assert!(filter.experiment.is_none());
        assert!(filter.status.is_none());
        assert!(filter.tags.is_none());
        assert!(filter.param_filter.is_none());
    }
}
