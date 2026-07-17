# Kiko SuperPoint model contracts

The Jetson profile's active sparse detector is `sp_topk2048.onnx`, SHA-256
`d3efd8800e34c3f08a4c5718cc31eaa8244e43deb80c4fc9db1ebae6f8c08ed4`. Direct inspection of
that ONNX graph establishes this profile:

- input `image`: `f32 [1, 1, height, width]`; the graph casts the input to `f16`, and both spatial
  dimensions must be at least 8;
- output `keypoints`: `i64 [1, N, 2]` absolute integer `(x, y)` model-raster pixels;
- output `scores`: `f32 [1, N]`, selected strictly above the graph's `f16` threshold `0x1019`
  (`0.0005002021789550781` as `f32`), in non-increasing `TopK` order and within the Softmax domain
  `(threshold, 1]`;
- output `descriptors`: concrete runtime `f32 [1, N, 256]`, with row `i` aligned to keypoint and
  score row `i`; and
- `0 <= N <= 2048`. The graph computes `N = min(2048, candidate_count)` and requests sorted,
  largest-first `TopK` output.

The graph has three stride-2 pools, so its effective raster is
`8 * floor(width / 8)` by `8 * floor(height / 8)`. Its coordinate path takes candidate `(y, x)`
indices, removes the batch column, reverses the remaining axis to emit `(x, y)`, and excludes a
four-pixel border. Therefore each model coordinate is in `[4, effective_extent - 4)`. Kiko maps a
downscaled integer coordinate back to the top-left-sampled source pixel as `coordinate * factor`,
with no inferred normalization, axis swap, or half-pixel offset. A mapped integer must be exactly
representable by Kiko's `f32` keypoint type.

The serialized symbolic descriptor shape in this family is inaccurate. A matching session
interface makes Kiko select the declared sparse adapter once at construction, but that boundary
cannot prove the graph's internal coordinate, threshold, ordering, or border semantics. Kiko does
**not** hash-attest the selected file at runtime. The supplied model therefore promises the active
profile, and Kiko requires its exact concrete `[1, N, 256]` descriptor layout at runtime.
The regular constructors and Jetson CLI default to the typed `SuperPointSparseProfile::CanonicalFp16`
contract (2048-row adapter capacity and threshold bits `0x3a032000`). Library callers using the
archived `superpoint_512.onnx` must explicitly select `SuperPointSparseProfile::LegacyFp32` through
`SuperPoint::new_with_sparse_profile` or `SuperPoint::new_with_backend_and_sparse_profile`; that
contract has a 512-row capacity and threshold bits `0x3a03126f`. The profile declaration is not
filename inference.

`--superpoint-model` and `KIKO_SUPERPOINT_MODEL` select a model promising the canonical contract;
they do not request value-based layout, coordinate, or score inference. Sparse `f32` keypoints,
normalized/subpixel coordinates, `(y, x)` output, a different border, an unsorted score vector, or
a different descriptor layout still require a separate explicit adapter.

The related tracked artifacts are distinct files:

- `sp.onnx`: `cdc58afc1de1af44f440ca49f5aefd6cad465a70f6be75d215cbe34a195cae06`, `f16` weights,
  graph cap 512;
- `superpoint_512.onnx`:
  `aaefb94ad6dd3624fe4300b39f0f1a77e8739ed6d5430162729fd6a72c265431`, `f32` weights, graph cap
  512, and threshold `0.0005000000237487257`; it is **not** byte-identical to `sp.onnx` and does not
  conform to the active strict-threshold profile despite exposing the same session interface, so it
  requires the typed legacy profile;
- `sp_topk1024.onnx`:
  `1a892261f19315bbeb1c38154d2e559fb673ac703aaa62b14a85cfc33142ebe5`, graph cap 1024;
- `sp_topk1536.onnx`:
  `3085af65ab8a18a24fd6205b7f53598f83abd0c8f6444cd7c2268549ed5cb1bd`, graph cap 1536; and
- `sp_topk2048_u8.onnx`:
  `dbd6cd5f93406049a1d6ac882eaf1047e49ffc9e00323cf10671d3ae7ab0b9f0`, `u8` image input and
  graph cap 2048.

The listed `f16` sparse exports, including the `u8` input variant, use the active exact threshold;
the archived `f32` `superpoint_512.onnx` is the exception described above.

The active artifact identifies PyTorch 2.0.1 as its producer but contains no embedded source,
license, training-preprocessing provenance, documentation string, or metadata properties. The host
currently converts grayscale bytes to `f32` by dividing by 255; the artifact alone does not prove
that this matches its training pipeline. An empty candidate set drives `TopK` with `K=0`. Kiko
accepts the exact empty output tensors, but provider portability of that graph path must be
established by the Jetson CUDA validation lane rather than inferred from the graph.
