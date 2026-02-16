use std::fs;

use crate::env::env_bool;

#[cfg(any(feature = "ort-coreml", feature = "ort-cuda", feature = "ort-tensorrt"))]
use ort::execution_providers::ExecutionProvider;
use ort::execution_providers::ExecutionProviderDispatch;
use ort::execution_providers::cpu::CPUExecutionProvider;

use super::InferenceError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceBackend {
    Auto,
    Cpu,
    CoreMLGpu,
    Cuda,
    TensorRT,
}

impl InferenceBackend {
    pub fn auto() -> Self {
        Self::Auto
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "auto" => Some(InferenceBackend::Auto),
            "cpu" => Some(InferenceBackend::Cpu),
            "coreml" | "coreml-gpu" => Some(InferenceBackend::CoreMLGpu),
            "cuda" => Some(InferenceBackend::Cuda),
            "tensorrt" => Some(InferenceBackend::TensorRT),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendSelection {
    selected: InferenceBackend,
    providers: Vec<ExecutionProviderDispatch>,
}

impl BackendSelection {
    pub fn selected(&self) -> InferenceBackend {
        self.selected
    }

    pub fn providers(&self) -> &[ExecutionProviderDispatch] {
        &self.providers
    }
}

pub(crate) fn select_backend(
    requested: InferenceBackend,
) -> Result<BackendSelection, InferenceError> {
    let desired = match requested {
        InferenceBackend::Auto => detect_backend(),
        other => other,
    };

    let mut providers = Vec::new();
    let mut selected = InferenceBackend::Cpu;

    match desired {
        InferenceBackend::CoreMLGpu => {
            if let Some(ep) = coreml_provider()? {
                providers.push(ep);
                selected = InferenceBackend::CoreMLGpu;
            }
        }
        InferenceBackend::Cuda => {
            if let Some(ep) = cuda_provider()? {
                providers.push(ep);
                selected = InferenceBackend::Cuda;
            }
        }
        InferenceBackend::TensorRT => {
            if let Some(ep) = tensorrt_provider()? {
                providers.push(ep);
                selected = InferenceBackend::TensorRT;
            } else if let Some(ep) = cuda_provider()? {
                providers.push(ep);
                selected = InferenceBackend::Cuda;
            }
        }
        InferenceBackend::Cpu => {
            selected = InferenceBackend::Cpu;
        }
        InferenceBackend::Auto => {
            selected = InferenceBackend::Cpu;
        }
    }

    if providers.is_empty() {
        selected = InferenceBackend::Cpu;
    }

    let use_cpu_arena = env_bool("KIKO_ORT_CPU_ARENA").unwrap_or(true);
    providers.push(
        CPUExecutionProvider::default()
            .with_arena_allocator(use_cpu_arena)
            .build(),
    );

    Ok(BackendSelection {
        selected,
        providers,
    })
}

fn detect_backend() -> InferenceBackend {
    if cfg!(target_vendor = "apple") {
        return InferenceBackend::CoreMLGpu;
    }

    if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
        if is_jetson() {
            return InferenceBackend::TensorRT;
        }
        return InferenceBackend::Cuda;
    }

    InferenceBackend::Cpu
}

fn is_jetson() -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }

    match fs::read_to_string("/proc/device-tree/model") {
        Ok(model) => model.to_lowercase().contains("jetson"),
        Err(_) => false,
    }
}

fn coreml_provider() -> Result<Option<ExecutionProviderDispatch>, InferenceError> {
    #[cfg(feature = "ort-coreml")]
    {
        use ort::execution_providers::coreml::{CoreMLComputeUnits, CoreMLExecutionProvider};

        let ep =
            CoreMLExecutionProvider::default().with_compute_units(CoreMLComputeUnits::CPUAndGPU);
        if !ep.supported_by_platform() {
            return Ok(None);
        }
        if !ep.is_available().map_err(InferenceError::Execution)? {
            return Ok(None);
        }
        Ok(Some(ep.build()))
    }

    #[cfg(not(feature = "ort-coreml"))]
    {
        Ok(None)
    }
}

fn cuda_provider() -> Result<Option<ExecutionProviderDispatch>, InferenceError> {
    #[cfg(feature = "ort-cuda")]
    {
        use ort::execution_providers::CUDAExecutionProvider;

        let ep = CUDAExecutionProvider::default();
        if !ep.supported_by_platform() {
            return Ok(None);
        }
        if !ep.is_available().map_err(InferenceError::Execution)? {
            return Ok(None);
        }
        Ok(Some(ep.build()))
    }

    #[cfg(not(feature = "ort-cuda"))]
    {
        Ok(None)
    }
}

fn tensorrt_provider() -> Result<Option<ExecutionProviderDispatch>, InferenceError> {
    #[cfg(feature = "ort-tensorrt")]
    {
        use ort::execution_providers::TensorRTExecutionProvider;

        let ep = TensorRTExecutionProvider::default();
        if !ep.supported_by_platform() {
            return Ok(None);
        }
        if !ep.is_available().map_err(InferenceError::Execution)? {
            return Ok(None);
        }
        Ok(Some(ep.build()))
    }

    #[cfg(not(feature = "ort-tensorrt"))]
    {
        Ok(None)
    }
}
