//! Training runs in bfloat16, the same storage type as the Spark checkpoint.
//!
//! The device default has to be set before the first tensor on that device.
//! Metric code can cast a copy back to f32 when it needs a host number.

#[cfg(target_os = "macos")]
use burn::tensor::DeviceKind;
use burn::tensor::{Device, DeviceError, FloatDType};

/// CUDA on Linux, Metal on macOS. Other hosts stay on the Flex CPU backend.
pub fn accelerator_name() -> &'static str {
    #[cfg(target_os = "linux")]
    {
        "cuda"
    }
    #[cfg(target_os = "macos")]
    {
        "metal"
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        "flex"
    }
}

fn accelerator_device() -> Device {
    #[cfg(target_os = "linux")]
    {
        Device::cuda(0)
    }
    #[cfg(target_os = "macos")]
    {
        Device::metal(DeviceKind::DefaultDevice)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        Device::flex()
    }
}

pub fn bf16_device(autodiff: bool) -> Result<Device, String> {
    let mut device = accelerator_device();
    if autodiff {
        device = device.autodiff();
    }
    if !device.supports_dtype(FloatDType::BF16) {
        return Err("this device cannot compute in bfloat16".into());
    }
    match device.configure(FloatDType::BF16) {
        Ok(()) => Ok(device),
        Err(DeviceError::AlreadyInitialized { .. }) => {
            if device.settings().float_dtype == FloatDType::BF16 {
                Ok(device)
            } else {
                Err(
                    "bfloat16 training must configure the device before any other tensor is created"
                        .into(),
                )
            }
        }
        Err(err) => Err(err.to_string()),
    }
}
