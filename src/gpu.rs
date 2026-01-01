// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

//! GPU Direct memory access (optional feature)
//!
//! Enables zero-copy loading of tensors directly to GPU memory.
//! Requires CUDA toolkit installed.
//!
//! # Feature Flag
//!
//! This module requires the `gpu` feature to be enabled:
//!
//! ```toml
//! [dependencies]
//! synadb = { version = "0.5", features = ["gpu"] }
//! ```
//!
//! # Requirements
//!
//! - CUDA toolkit installed
//! - Compatible NVIDIA GPU
//!
//! # Example (when gpu feature is enabled)
//!
//! ```rust,ignore
//! use synadb::gpu::{GpuContext, GpuTensor};
//!
//! // Initialize GPU context for device 0
//! let ctx = GpuContext::new(0)?;
//!
//! // Upload data to GPU
//! let data = vec![1.0f32, 2.0, 3.0, 4.0];
//! let gpu_tensor = ctx.upload(&data)?;
//!
//! // Get raw pointer for use with PyTorch/TensorFlow
//! let ptr = gpu_tensor.as_ptr();
//! ```

use crate::error::{Result, SynaError};

// =============================================================================
// GPU Feature Enabled Implementation
// =============================================================================

#[cfg(feature = "gpu")]
mod gpu_impl {
    use super::*;
    use std::ffi::c_void;

    // CUDA runtime API bindings
    // These are the minimal bindings needed for GPU memory operations
    #[link(name = "cudart")]
    extern "C" {
        fn cudaSetDevice(device: i32) -> i32;
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
        fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
        fn cudaFree(devPtr: *mut c_void) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
        fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
        fn cudaFreeHost(ptr: *mut c_void) -> i32;
        fn cudaDeviceSynchronize() -> i32;
        fn cudaGetErrorString(error: i32) -> *const std::ffi::c_char;
    }

    // CUDA memory copy kinds
    const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
    const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

    /// GPU device context for managing CUDA operations.
    ///
    /// Provides methods for uploading tensors to GPU memory and managing
    /// GPU resources. Each context is associated with a specific GPU device.
    pub struct GpuContext {
        device: i32,
    }

    impl GpuContext {
        /// Initialize GPU context for a specific device.
        ///
        /// # Arguments
        ///
        /// * `device` - CUDA device index (0-based)
        ///
        /// # Errors
        ///
        /// Returns `SynaError::GpuUnavailable` if:
        /// - No CUDA devices are available
        /// - The specified device index is invalid
        /// - CUDA initialization fails
        ///
        /// # Example
        ///
        /// ```rust,ignore
        /// let ctx = GpuContext::new(0)?; // Use first GPU
        /// ```
        pub fn new(device: i32) -> Result<Self> {
            unsafe {
                // Check device count
                let mut count: i32 = 0;
                let result = cudaGetDeviceCount(&mut count);
                if result != 0 {
                    return Err(SynaError::GpuUnavailable(format!(
                        "Failed to get device count: {}",
                        get_cuda_error_string(result)
                    )));
                }

                if count == 0 {
                    return Err(SynaError::GpuUnavailable(
                        "No CUDA devices available".to_string(),
                    ));
                }

                if device < 0 || device >= count {
                    return Err(SynaError::GpuUnavailable(format!(
                        "Invalid device {}: only {} devices available",
                        device, count
                    )));
                }

                // Set the device
                let result = cudaSetDevice(device);
                if result != 0 {
                    return Err(SynaError::GpuUnavailable(format!(
                        "Failed to set device {}: {}",
                        device,
                        get_cuda_error_string(result)
                    )));
                }
            }

            Ok(Self { device })
        }

        /// Get the number of available CUDA devices.
        ///
        /// # Returns
        ///
        /// The number of CUDA-capable devices, or 0 if CUDA is not available.
        pub fn device_count() -> i32 {
            unsafe {
                let mut count: i32 = 0;
                let result = cudaGetDeviceCount(&mut count);
                if result != 0 {
                    return 0;
                }
                count
            }
        }

        /// Get the device index for this context.
        pub fn device(&self) -> i32 {
            self.device
        }

        /// Upload data to GPU memory.
        ///
        /// Allocates GPU memory and copies the provided data from host to device.
        ///
        /// # Arguments
        ///
        /// * `data` - Slice of f32 values to upload
        ///
        /// # Returns
        ///
        /// A `GpuTensor` containing the device pointer and metadata.
        ///
        /// # Errors
        ///
        /// Returns `SynaError::GpuOutOfMemory` if GPU memory allocation fails.
        ///
        /// # Example
        ///
        /// ```rust,ignore
        /// let ctx = GpuContext::new(0)?;
        /// let data = vec![1.0f32, 2.0, 3.0, 4.0];
        /// let gpu_tensor = ctx.upload(&data)?;
        /// ```
        pub fn upload(&self, data: &[f32]) -> Result<GpuTensor> {
            if data.is_empty() {
                return Ok(GpuTensor {
                    ptr: std::ptr::null_mut(),
                    len: 0,
                    device: self.device,
                });
            }

            let size = data.len() * std::mem::size_of::<f32>();
            let mut device_ptr: *mut c_void = std::ptr::null_mut();

            unsafe {
                // Ensure we're on the right device
                cudaSetDevice(self.device);

                // Allocate device memory
                let result = cudaMalloc(&mut device_ptr, size);
                if result != 0 {
                    return Err(SynaError::GpuOutOfMemory(format!(
                        "Failed to allocate {} bytes: {}",
                        size,
                        get_cuda_error_string(result)
                    )));
                }

                // Copy data to device
                let result = cudaMemcpy(
                    device_ptr,
                    data.as_ptr() as *const c_void,
                    size,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                );
                if result != 0 {
                    // Free allocated memory on error
                    cudaFree(device_ptr);
                    return Err(SynaError::GpuUnavailable(format!(
                        "Failed to copy data to device: {}",
                        get_cuda_error_string(result)
                    )));
                }
            }

            Ok(GpuTensor {
                ptr: device_ptr as *mut f32,
                len: data.len(),
                device: self.device,
            })
        }

        /// Upload data to GPU memory using pinned (page-locked) host memory.
        ///
        /// This method uses pinned memory for faster async transfers.
        /// Useful for large tensors where transfer speed is critical.
        ///
        /// # Arguments
        ///
        /// * `data` - Slice of f32 values to upload
        ///
        /// # Returns
        ///
        /// A `GpuTensor` containing the device pointer and metadata.
        ///
        /// # Errors
        ///
        /// Returns `SynaError::GpuOutOfMemory` if memory allocation fails.
        pub fn upload_pinned(&self, data: &[f32]) -> Result<GpuTensor> {
            if data.is_empty() {
                return Ok(GpuTensor {
                    ptr: std::ptr::null_mut(),
                    len: 0,
                    device: self.device,
                });
            }

            let size = data.len() * std::mem::size_of::<f32>();
            let mut device_ptr: *mut c_void = std::ptr::null_mut();
            let mut pinned_ptr: *mut c_void = std::ptr::null_mut();

            unsafe {
                // Ensure we're on the right device
                cudaSetDevice(self.device);

                // Allocate pinned host memory
                let result = cudaMallocHost(&mut pinned_ptr, size);
                if result != 0 {
                    return Err(SynaError::GpuOutOfMemory(format!(
                        "Failed to allocate pinned memory: {}",
                        get_cuda_error_string(result)
                    )));
                }

                // Copy data to pinned memory
                std::ptr::copy_nonoverlapping(data.as_ptr(), pinned_ptr as *mut f32, data.len());

                // Allocate device memory
                let result = cudaMalloc(&mut device_ptr, size);
                if result != 0 {
                    cudaFreeHost(pinned_ptr);
                    return Err(SynaError::GpuOutOfMemory(format!(
                        "Failed to allocate device memory: {}",
                        get_cuda_error_string(result)
                    )));
                }

                // Async copy from pinned to device
                let result = cudaMemcpyAsync(
                    device_ptr,
                    pinned_ptr,
                    size,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    std::ptr::null_mut(), // default stream
                );
                if result != 0 {
                    cudaFree(device_ptr);
                    cudaFreeHost(pinned_ptr);
                    return Err(SynaError::GpuUnavailable(format!(
                        "Failed to copy data to device: {}",
                        get_cuda_error_string(result)
                    )));
                }

                // Synchronize to ensure copy is complete
                cudaDeviceSynchronize();

                // Free pinned memory
                cudaFreeHost(pinned_ptr);
            }

            Ok(GpuTensor {
                ptr: device_ptr as *mut f32,
                len: data.len(),
                device: self.device,
            })
        }

        /// Synchronize the GPU device.
        ///
        /// Blocks until all previously issued CUDA commands have completed.
        pub fn synchronize(&self) -> Result<()> {
            unsafe {
                cudaSetDevice(self.device);
                let result = cudaDeviceSynchronize();
                if result != 0 {
                    return Err(SynaError::GpuUnavailable(format!(
                        "Device synchronization failed: {}",
                        get_cuda_error_string(result)
                    )));
                }
            }
            Ok(())
        }
    }

    /// Tensor stored in GPU memory.
    ///
    /// Provides access to GPU memory for use with deep learning frameworks
    /// like PyTorch and TensorFlow. The memory is automatically freed when
    /// the tensor is dropped.
    ///
    /// # Safety
    ///
    /// The raw pointer returned by `as_ptr()` is only valid while this
    /// `GpuTensor` is alive. Do not use the pointer after dropping the tensor.
    pub struct GpuTensor {
        ptr: *mut f32,
        len: usize,
        device: i32,
    }

    impl GpuTensor {
        /// Get raw device pointer for use with PyTorch/TensorFlow.
        ///
        /// # Safety
        ///
        /// The returned pointer is only valid while this `GpuTensor` exists.
        /// Using the pointer after the tensor is dropped is undefined behavior.
        ///
        /// # Example
        ///
        /// ```rust,ignore
        /// let gpu_tensor = ctx.upload(&data)?;
        /// let ptr = gpu_tensor.as_ptr();
        /// // Use ptr with PyTorch: torch.from_dlpack(...)
        /// ```
        pub fn as_ptr(&self) -> *mut f32 {
            self.ptr
        }

        /// Get tensor length (number of f32 elements).
        pub fn len(&self) -> usize {
            self.len
        }

        /// Check if tensor is empty.
        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// Get the device index this tensor is stored on.
        pub fn device(&self) -> i32 {
            self.device
        }

        /// Get the size in bytes.
        pub fn size_bytes(&self) -> usize {
            self.len * std::mem::size_of::<f32>()
        }

        /// Download tensor data from GPU to host memory.
        ///
        /// # Returns
        ///
        /// A `Vec<f32>` containing the tensor data.
        ///
        /// # Errors
        ///
        /// Returns an error if the memory copy fails.
        pub fn download(&self) -> Result<Vec<f32>> {
            if self.len == 0 || self.ptr.is_null() {
                return Ok(Vec::new());
            }

            let mut data = vec![0.0f32; self.len];
            let size = self.len * std::mem::size_of::<f32>();

            unsafe {
                cudaSetDevice(self.device);
                let result = cudaMemcpy(
                    data.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    size,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                );
                if result != 0 {
                    return Err(SynaError::GpuUnavailable(format!(
                        "Failed to copy data from device: {}",
                        get_cuda_error_string(result)
                    )));
                }
            }

            Ok(data)
        }
    }

    impl Drop for GpuTensor {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe {
                    cudaSetDevice(self.device);
                    cudaFree(self.ptr as *mut c_void);
                }
            }
        }
    }

    // GpuTensor is Send because CUDA memory can be accessed from any thread
    // (as long as the device is set correctly)
    unsafe impl Send for GpuTensor {}

    /// Get CUDA error string from error code.
    fn get_cuda_error_string(error: i32) -> String {
        unsafe {
            let ptr = cudaGetErrorString(error);
            if ptr.is_null() {
                return format!("Unknown error ({})", error);
            }
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

// Re-export GPU types when feature is enabled
#[cfg(feature = "gpu")]
pub use gpu_impl::{GpuContext, GpuTensor};

// =============================================================================
// Stub Implementation (GPU feature disabled)
// =============================================================================

/// Stub GPU context when GPU feature is disabled.
///
/// All methods return `SynaError::GpuUnavailable` with instructions
/// to rebuild with the `gpu` feature enabled.
#[cfg(not(feature = "gpu"))]
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    /// Attempt to create a GPU context (always fails without gpu feature).
    ///
    /// # Errors
    ///
    /// Always returns `SynaError::GpuUnavailable` with instructions to
    /// rebuild with the `gpu` feature.
    pub fn new(_device: i32) -> Result<Self> {
        Err(SynaError::GpuUnavailable(
            "GPU support not compiled. Rebuild with --features gpu".to_string(),
        ))
    }

    /// Get device count (always returns 0 without gpu feature).
    pub fn device_count() -> i32 {
        0
    }
}

/// Stub GPU tensor when GPU feature is disabled.
#[cfg(not(feature = "gpu"))]
pub struct GpuTensor;

#[cfg(not(feature = "gpu"))]
impl GpuTensor {
    /// Get raw pointer (always returns null without gpu feature).
    pub fn as_ptr(&self) -> *mut f32 {
        std::ptr::null_mut()
    }

    /// Get length (always returns 0 without gpu feature).
    pub fn len(&self) -> usize {
        0
    }

    /// Check if empty (always returns true without gpu feature).
    pub fn is_empty(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "gpu"))]
    fn test_gpu_unavailable_without_feature() {
        let result = GpuContext::new(0);
        assert!(result.is_err());
        match result {
            Err(SynaError::GpuUnavailable(msg)) => {
                assert!(msg.contains("GPU support not compiled"));
            }
            _ => panic!("Expected GpuUnavailable error"),
        }
    }

    #[test]
    #[cfg(not(feature = "gpu"))]
    fn test_device_count_without_feature() {
        assert_eq!(GpuContext::device_count(), 0);
    }

    #[test]
    #[cfg(not(feature = "gpu"))]
    fn test_stub_tensor() {
        let tensor = GpuTensor;
        assert!(tensor.as_ptr().is_null());
        assert_eq!(tensor.len(), 0);
        assert!(tensor.is_empty());
    }
}
