use std::fs;
#[cfg(feature = "ort-tensorrt")]
use std::path::PathBuf;

use ort::execution_providers::CPUExecutionProvider;
#[cfg(any(feature = "ort-coreml", feature = "ort-cuda", feature = "ort-tensorrt"))]
use ort::execution_providers::ExecutionProvider;
use ort::execution_providers::ExecutionProviderDispatch;

use super::{InferenceError, inference_env};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceBackend {
    Auto,
    Cpu,
    CoreMLCpuAndGpu,
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
            "coreml" | "coreml-gpu" | "coreml-cpu-gpu" => Some(InferenceBackend::CoreMLCpuAndGpu),
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
    provider_names: Vec<&'static str>,
    strict_accelerator: bool,
}

impl BackendSelection {
    pub fn selected(&self) -> InferenceBackend {
        self.selected
    }

    pub fn providers(&self) -> &[ExecutionProviderDispatch] {
        &self.providers
    }

    pub fn provider_names(&self) -> &[&'static str] {
        &self.provider_names
    }

    pub fn strict_accelerator(&self) -> bool {
        self.strict_accelerator
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BackendPolicy {
    strict_accelerator: bool,
}

pub(crate) fn select_backend(
    requested: InferenceBackend,
) -> Result<BackendSelection, InferenceError> {
    let jetson = is_jetson();
    let desired = match requested {
        InferenceBackend::Auto => detect_backend(),
        other => other,
    };
    let allow_fallback =
        inference_env(crate::env::try_env_bool("KIKO_ALLOW_BACKEND_FALLBACK"))?.unwrap_or(false);

    let mut providers = Vec::new();
    let mut provider_names = Vec::new();
    let mut selected = InferenceBackend::Cpu;

    match desired {
        InferenceBackend::CoreMLCpuAndGpu => {
            if let Some(ep) = coreml_provider()? {
                providers.push(ep);
                provider_names.push("CoreML(CPUAndGPU)");
                selected = InferenceBackend::CoreMLCpuAndGpu;
            }
        }
        InferenceBackend::Cuda => {
            if let Some(ep) = cuda_provider(jetson)? {
                providers.push(ep);
                provider_names.push("CUDA");
                selected = InferenceBackend::Cuda;
            }
        }
        InferenceBackend::TensorRT => {
            if let Some(ep) = tensorrt_provider()? {
                providers.push(ep);
                provider_names.push("TensorRT");
                selected = InferenceBackend::TensorRT;
                if let Some(ep) = cuda_provider(jetson)? {
                    providers.push(ep);
                    provider_names.push("CUDA");
                }
            } else if requested == InferenceBackend::Auto || allow_fallback {
                if let Some(ep) = cuda_provider(jetson)? {
                    providers.push(ep);
                    provider_names.push("CUDA");
                    selected = InferenceBackend::Cuda;
                }
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

    let policy = validate_backend_selection(requested, desired, selected, allow_fallback, jetson)?;
    providers = configure_provider_registration(providers, policy.strict_accelerator);
    if !policy.strict_accelerator {
        let use_cpu_arena =
            inference_env(crate::env::try_env_bool("KIKO_ORT_CPU_ARENA"))?.unwrap_or(true);
        providers.push(
            CPUExecutionProvider::default()
                .with_arena_allocator(use_cpu_arena)
                .build(),
        );
        provider_names.push("CPU");
    }

    Ok(BackendSelection {
        selected,
        providers,
        provider_names,
        strict_accelerator: policy.strict_accelerator,
    })
}

fn configure_provider_registration(
    providers: Vec<ExecutionProviderDispatch>,
    strict_accelerator: bool,
) -> Vec<ExecutionProviderDispatch> {
    if strict_accelerator {
        providers
            .into_iter()
            .map(ExecutionProviderDispatch::error_on_failure)
            .collect()
    } else {
        providers
    }
}

fn validate_backend_selection(
    requested: InferenceBackend,
    desired: InferenceBackend,
    selected: InferenceBackend,
    allow_fallback: bool,
    jetson: bool,
) -> Result<BackendPolicy, InferenceError> {
    let explicit_accelerator = matches!(
        requested,
        InferenceBackend::CoreMLCpuAndGpu | InferenceBackend::Cuda | InferenceBackend::TensorRT
    );
    let auto_jetson_accelerator =
        requested == InferenceBackend::Auto && jetson && desired != InferenceBackend::Cpu;

    if explicit_accelerator && selected != requested && !allow_fallback {
        return Err(InferenceError::BackendUnavailable {
            requested,
            selected,
        });
    }

    if auto_jetson_accelerator && selected == InferenceBackend::Cpu && !allow_fallback {
        return Err(InferenceError::BackendUnavailable {
            requested: desired,
            selected,
        });
    }

    Ok(BackendPolicy {
        strict_accelerator: (explicit_accelerator || auto_jetson_accelerator)
            && selected != InferenceBackend::Cpu
            && !allow_fallback,
    })
}

fn detect_backend() -> InferenceBackend {
    if cfg!(target_vendor = "apple") {
        return InferenceBackend::CoreMLCpuAndGpu;
    }

    if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
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
        use ort::execution_providers::CoreML;
        use ort::execution_providers::coreml::ComputeUnits;

        let ep = CoreML::default().with_compute_units(ComputeUnits::CPUAndGPU);
        if !ep.supported_by_platform() {
            return Ok(None);
        }
        if !ep
            .is_available()
            .map_err(|source| InferenceError::BackendProbe {
                backend: InferenceBackend::CoreMLCpuAndGpu,
                source,
            })?
        {
            return Ok(None);
        }
        Ok(Some(ep.build()))
    }

    #[cfg(not(feature = "ort-coreml"))]
    {
        Ok(None)
    }
}

fn cuda_provider(jetson: bool) -> Result<Option<ExecutionProviderDispatch>, InferenceError> {
    #[cfg(feature = "ort-cuda")]
    {
        use ort::execution_providers::CUDAExecutionProvider;

        let default_conv_search = if jetson { "heuristic" } else { "exhaustive" };
        let conv_search_raw = inference_env(crate::env::try_env_string("KIKO_CUDA_CONV_SEARCH"))?
            .unwrap_or_else(|| default_conv_search.to_string());
        let conv_search = match conv_search_raw.trim().to_lowercase().as_str() {
            "heuristic" => ort::execution_providers::cuda::ConvAlgorithmSearch::Heuristic,
            "default" => ort::execution_providers::cuda::ConvAlgorithmSearch::Default,
            "exhaustive" => ort::execution_providers::cuda::ConvAlgorithmSearch::Exhaustive,
            _ => {
                return Err(InferenceError::InvalidSetting {
                    key: "KIKO_CUDA_CONV_SEARCH",
                    value: conv_search_raw,
                    expected: "heuristic, default, or exhaustive",
                });
            }
        };
        let prefer_nhwc =
            inference_env(crate::env::try_env_bool("KIKO_CUDA_PREFER_NHWC"))?.unwrap_or(false);
        let fuse_conv_bias =
            inference_env(crate::env::try_env_bool("KIKO_CUDA_FUSE_CONV_BIAS"))?.unwrap_or(false);
        let cuda_graph =
            inference_env(crate::env::try_env_bool("KIKO_CUDA_GRAPH"))?.unwrap_or(false);
        let ep = CUDAExecutionProvider::default()
            .with_conv_algorithm_search(conv_search)
            .with_conv_max_workspace(true)
            .with_tf32(true)
            .with_prefer_nhwc(prefer_nhwc)
            .with_fuse_conv_bias(fuse_conv_bias)
            .with_cuda_graph(cuda_graph)
            .with_attention_backend(ort::execution_providers::cuda::AttentionBackend::all());
        if !ep.supported_by_platform() {
            return Ok(None);
        }
        if !ep
            .is_available()
            .map_err(|source| InferenceError::BackendProbe {
                backend: InferenceBackend::Cuda,
                source,
            })?
        {
            return Ok(None);
        }
        Ok(Some(ep.build()))
    }

    #[cfg(not(feature = "ort-cuda"))]
    {
        let _ = jetson;
        Ok(None)
    }
}

fn tensorrt_provider() -> Result<Option<ExecutionProviderDispatch>, InferenceError> {
    #[cfg(feature = "ort-tensorrt")]
    {
        use ort::execution_providers::TensorRTExecutionProvider;

        let cache_dir_value = inference_env(crate::env::try_env_string("KIKO_TRT_CACHE_DIR"))?
            .unwrap_or_else(|| "/home/makerspace/.cache/kiko-trt-engines".to_string());
        let cache_dir = PathBuf::from(&cache_dir_value);
        std::fs::create_dir_all(&cache_dir).map_err(|source| InferenceError::CacheDirectory {
            path: cache_dir.clone(),
            source,
        })?;
        let dump_subgraphs =
            inference_env(crate::env::try_env_bool("KIKO_TRT_DUMP_SUBGRAPHS"))?.unwrap_or(false);
        let detailed_build_log =
            inference_env(crate::env::try_env_bool("KIKO_TRT_DETAILED_BUILD_LOG"))?
                .unwrap_or(false);
        let cuda_graph =
            inference_env(crate::env::try_env_bool("KIKO_TRT_CUDA_GRAPH"))?.unwrap_or(false);

        let ep = TensorRTExecutionProvider::default()
            .with_fp16(true)
            .with_engine_cache(true)
            .with_engine_cache_path(&cache_dir_value)
            .with_timing_cache(true)
            .with_timing_cache_path(&cache_dir_value)
            .with_build_heuristics(true)
            .with_builder_optimization_level(5)
            .with_cuda_graph(cuda_graph)
            .with_detailed_build_log(detailed_build_log)
            .with_dump_subgraphs(dump_subgraphs);
        // Do not set explicit TRT shape profiles here. SuperPoint partitions at
        // dynamic NonZero/Where/Gather outputs, and ORT requires profiles for
        // every dynamic subgraph input once any profile is specified. Let ORT/TRT
        // infer profiles from the first inference instead of maintaining brittle
        // intermediate tensor names.
        if !ep.supported_by_platform() {
            return Ok(None);
        }
        if !ep
            .is_available()
            .map_err(|source| InferenceError::BackendProbe {
                backend: InferenceBackend::TensorRT,
                source,
            })?
        {
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
    #[cfg(not(feature = "ort-cuda"))]
    use super::configure_provider_registration;
    use super::{BackendPolicy, InferenceBackend, validate_backend_selection};
    use crate::inference::InferenceError;

    #[test]
    fn explicit_accelerator_request_rejects_fallback_without_opt_in() {
        let err = validate_backend_selection(
            InferenceBackend::TensorRT,
            InferenceBackend::TensorRT,
            InferenceBackend::Cpu,
            false,
            true,
        )
        .expect_err("fallback should fail");
        assert!(matches!(
            err,
            InferenceError::BackendUnavailable {
                requested: InferenceBackend::TensorRT,
                selected: InferenceBackend::Cpu,
            }
        ));
    }

    #[test]
    fn explicit_unavailable_backend_rejects_fallback_without_opt_in() {
        let err = validate_backend_selection(
            InferenceBackend::Cuda,
            InferenceBackend::Cuda,
            InferenceBackend::Cpu,
            false,
            true,
        )
        .expect_err("an explicit unavailable backend must fail");
        assert!(matches!(
            err,
            InferenceError::BackendUnavailable {
                requested: InferenceBackend::Cuda,
                selected: InferenceBackend::Cpu,
            }
        ));
    }

    #[test]
    fn selected_accelerators_are_strict_without_fallback_opt_in() {
        for backend in [
            InferenceBackend::CoreMLCpuAndGpu,
            InferenceBackend::Cuda,
            InferenceBackend::TensorRT,
        ] {
            assert_eq!(
                validate_backend_selection(backend, backend, backend, false, true)
                    .expect("available explicit accelerator"),
                BackendPolicy {
                    strict_accelerator: true,
                }
            );
        }
    }

    #[test]
    fn selected_accelerator_allows_cpu_fallback_only_with_opt_in() {
        assert_eq!(
            validate_backend_selection(
                InferenceBackend::Cuda,
                InferenceBackend::Cuda,
                InferenceBackend::Cuda,
                true,
                true,
            )
            .expect("available explicit accelerator"),
            BackendPolicy {
                strict_accelerator: false,
            }
        );
    }

    #[test]
    fn jetson_auto_accelerator_is_strict_without_fallback_opt_in() {
        assert_eq!(
            validate_backend_selection(
                InferenceBackend::Auto,
                InferenceBackend::Cuda,
                InferenceBackend::Cuda,
                false,
                true,
            )
            .expect("available auto-selected Jetson accelerator"),
            BackendPolicy {
                strict_accelerator: true,
            }
        );
    }

    #[test]
    fn non_jetson_auto_accelerator_retains_cpu_fallback() {
        assert_eq!(
            validate_backend_selection(
                InferenceBackend::Auto,
                InferenceBackend::CoreMLCpuAndGpu,
                InferenceBackend::CoreMLCpuAndGpu,
                false,
                false,
            )
            .expect("available auto-selected non-Jetson accelerator"),
            BackendPolicy {
                strict_accelerator: false,
            }
        );
    }

    #[test]
    fn jetson_auto_rejects_cpu_fallback_without_opt_in() {
        let err = validate_backend_selection(
            InferenceBackend::Auto,
            InferenceBackend::TensorRT,
            InferenceBackend::Cpu,
            false,
            true,
        )
        .expect_err("cpu fallback should fail on jetson auto");
        assert!(matches!(
            err,
            InferenceError::BackendUnavailable {
                requested: InferenceBackend::TensorRT,
                selected: InferenceBackend::Cpu,
            }
        ));
    }

    #[test]
    fn fallback_opt_in_allows_backend_downgrade() {
        let policy = validate_backend_selection(
            InferenceBackend::TensorRT,
            InferenceBackend::TensorRT,
            InferenceBackend::Cpu,
            true,
            true,
        )
        .expect("auto fallback opt-in should allow downgrade");
        assert_eq!(
            policy,
            BackendPolicy {
                strict_accelerator: false,
            }
        );
    }

    #[cfg(not(feature = "ort-cuda"))]
    #[test]
    fn strict_dispatch_surfaces_provider_registration_failure() {
        use ort::execution_providers::CUDAExecutionProvider;

        let providers =
            configure_provider_registration(vec![CUDAExecutionProvider::default().build()], true);
        let builder = ort::session::Session::builder().expect("session builder");
        assert!(builder.with_execution_providers(providers).is_err());
    }
}
