# Kiko inference model contracts

Kiko's host-side `SuperPoint` adapter implements the concrete profile exported by `sp.onnx`
(SHA-256 `aaefb94ad6dd3624fe4300b39f0f1a77e8739ed6d5430162729fd6a72c265431`):

- input `image`: `f32 [1, 1, height, width]`, with both spatial dimensions at least 8;
- output `keypoints`: `i64 [1, N, 2]` absolute `(x, y)` model-raster pixels;
- output `scores`: `f32 [1, N]`, finite, non-increasing, and within `(0.0005, 1]`;
- output `descriptors`: runtime `f32 [1, N, 256]`; and
- `0 <= N <= 512`, with row `i` aligned across all three outputs.

The graph uses an effective grid of `8 * floor(extent / 8)` and excludes a four-pixel border,
so each coordinate is in `[4, effective_extent - 4)`. A host downscale factor maps a model
coordinate to the top-left-sampled original pixel `coordinate * factor`; there is no half-pixel
offset. The descriptor output's serialized symbolic shape metadata is inaccurate, so Kiko checks
the concrete runtime shape rather than accepting the metadata as authority.

Kiko stores detection pixels as `f32`. Every scaled integer coordinate must therefore be exactly
representable as `f32`; the adapter rejects an unrepresentable pixel instead of silently rounding it
to a neighboring coordinate. Models or image domains that require wider integer/subpixel coordinates
need an explicit wider detection type.

`--superpoint-model` and `KIKO_SUPERPOINT_MODEL` select a path, not a coordinate/layout adapter.
A replacement model must implement the profile above. Supporting normalized, subpixel, transposed,
different-border, or larger-output models requires an explicit adapter/profile rather than
value-based inference.

The host currently converts grayscale bytes to `f32` by dividing by 255. The ONNX artifact contains
no source, license, training-preprocessing provenance, documentation string, or metadata properties,
so this repository cannot establish that the host normalization matches the model's training
pipeline. `superpoint_512.onnx` is byte-identical to `sp.onnx`; it remains present until external
packaging and Nano path compatibility are audited separately.

An empty candidate set makes this graph invoke `TopK` with `K=0`. ONNX Runtime's CPU provider was
observed to return the documented empty tensors, which Kiko accepts. Portability of that graph path
to CUDA, TensorRT, CoreML, and other providers has not been established; this is a model/provider
uncertainty, not a claimed cross-provider guarantee.
