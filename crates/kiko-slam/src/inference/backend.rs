use std::fs;

#[cfg(any(feature = "ort-coreml", feature = "ort-cuda", feature = "ort-tensorrt"))]
use ort::ep::ExecutionProvider;
use ort::ep::{CPU, ExecutionProviderDispatch};

use super::InferenceError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceBackend {
    Auto,
    Cpu,
    CoreMLGpu,
    Cuda,
    /// CUDA is the primary execution provider and ONNX Runtime may assign
    /// unsupported graph nodes to the explicitly configured CPU provider.
    CudaCpuHybrid,
    TensorRT,
}

impl InferenceBackend {
    pub fn auto() -> Self {
        Self::Auto
    }

    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("auto") {
            return Some(Self::Auto);
        }
        if value.eq_ignore_ascii_case("cpu") {
            return Some(Self::Cpu);
        }
        if value.eq_ignore_ascii_case("coreml") || value.eq_ignore_ascii_case("coreml-gpu") {
            return Some(Self::CoreMLGpu);
        }
        if value.eq_ignore_ascii_case("cuda") {
            return Some(Self::Cuda);
        }
        if value.eq_ignore_ascii_case("cuda-cpu-hybrid")
            || value.eq_ignore_ascii_case("cuda_cpu_hybrid")
        {
            return Some(Self::CudaCpuHybrid);
        }
        if value.eq_ignore_ascii_case("tensorrt") {
            return Some(Self::TensorRT);
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct BackendSelection {
    selected: InferenceBackend,
    providers: Vec<ExecutionProviderDispatch>,
    strict_accelerator: bool,
}

impl BackendSelection {
    pub fn selected(&self) -> InferenceBackend {
        self.selected
    }

    pub fn providers(&self) -> &[ExecutionProviderDispatch] {
        &self.providers
    }

    pub fn strict_accelerator(&self) -> bool {
        self.strict_accelerator
    }
}

pub(crate) fn select_backend(
    requested: InferenceBackend,
    use_cpu_arena: bool,
) -> Result<BackendSelection, InferenceError> {
    let explicit = requested != InferenceBackend::Auto;
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
            } else if explicit {
                return Err(InferenceError::BackendUnavailable { requested });
            }
        }
        InferenceBackend::Cuda => {
            if let Some(ep) = cuda_provider()? {
                providers.push(ep);
                selected = InferenceBackend::Cuda;
            } else if explicit {
                return Err(InferenceError::BackendUnavailable { requested });
            }
        }
        InferenceBackend::CudaCpuHybrid => {
            if let Some(ep) = cuda_provider()? {
                providers.push(ep);
                selected = InferenceBackend::CudaCpuHybrid;
            } else {
                return Err(InferenceError::BackendUnavailable { requested });
            }
        }
        InferenceBackend::TensorRT => {
            if let Some(ep) = tensorrt_provider()? {
                providers.push(ep);
                selected = InferenceBackend::TensorRT;
            } else if explicit {
                return Err(InferenceError::BackendUnavailable { requested });
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

    let strict_accelerator = explicit
        && !matches!(
            selected,
            InferenceBackend::Cpu | InferenceBackend::CudaCpuHybrid
        );
    if !strict_accelerator {
        providers.push(CPU::default().with_arena_allocator(use_cpu_arena).build());
    }

    Ok(BackendSelection {
        selected,
        providers,
        strict_accelerator,
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
        use ort::ep::CoreML;
        use ort::ep::coreml::ComputeUnits;

        let ep = CoreML::default().with_compute_units(ComputeUnits::CPUAndGPU);
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
        use ort::ep::CUDA;

        let ep = CUDA::default();
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
        use ort::ep::TensorRT;

        let ep = TensorRT::default();
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

#[cfg(test)]
mod tests {
    use super::InferenceBackend;

    #[test]
    fn backend_parser_is_ascii_case_insensitive() {
        assert_eq!(
            InferenceBackend::parse(" AUTO "),
            Some(InferenceBackend::Auto)
        );
        assert_eq!(InferenceBackend::parse("Cpu"), Some(InferenceBackend::Cpu));
        assert_eq!(
            InferenceBackend::parse("COREML-GPU"),
            Some(InferenceBackend::CoreMLGpu)
        );
        assert_eq!(
            InferenceBackend::parse("TensorRT"),
            Some(InferenceBackend::TensorRT)
        );
        assert_eq!(
            InferenceBackend::parse("CUDA_CPU_HYBRID"),
            Some(InferenceBackend::CudaCpuHybrid)
        );
        assert_eq!(InferenceBackend::parse("gpu"), None);
    }
}
