# Legacy eye-source reconciliation

The source drop inspected on 2026-07-20 was
`/Users/ttrb/Downloads/Kiko_expression_engine/eyes`. It is evidence, not a
runtime dependency and not a canonical protocol authority.

Relevant source digests (SHA-256):

| Legacy file | SHA-256 |
| --- | --- |
| `src/main.rs` | `543adcf15281046248a5d49fae92050b886193a89ae6e7057c370c3a0a818b15` |
| `src/host.rs` | `14b7afd7b3d5972177b85a7a2f22cd2bdca8b123d170d9f5543c66c95bbc7853` |
| `src/geometry.rs` | `38fbbf822516f2172976346a29badae3053d4d6c7124c99e1d0bc8720b5a706d` |
| `src/animation/host_eye.rs` | `65bcf56c9e779ca9ed8244756d969ab10e2d2b6ce15067c2454014a07a4c7901` |
| `src/panel.rs` | `9a1051e823313e143f96e5889bd0dfec6c727f0bea111e2c26461c6cbbc3b60c` |
| `src/color.rs` | `f284a3d78d8bd0e8966e6bac2bbcbaf0472ff29fad698922d7868b099d18caf8` |
| `src/framebuffer.rs` | `6d1aa24c98406b1e62f190cf1504c0aa9fa01e2617625f8189bf00ae93fc6588` |
| `Cargo.toml` | `5ffc8a41d522a6c59c478488a06300e78fe55a40b8b49a90e8311aaf8174436f` |
| `memory.x` | `9401f0e66043283bbad3b2f7618bf2a261c27eb46d6cdfb526d895f8518931ee` |

## Retained

- RP2350A / Pico 2 W target and Embassy PIO + DMA + USB CDC architecture.
- Two 56-LED WS2812 panels, GP15 left and GP16 right as used by the actual
  legacy executable source.
- The exact measured 56-position 1/16 mm geometry and chain order.
- ±18 mm gaze range, 11 mm neutral pupil radius, 60 Hz render cadence,
  200 ms blink duration, Kiko green, dark pupil, gamma-2 curve, and 56/255
  channel ceiling.
- Double-buffered concurrent panel DMA and render structure.

## Replaced or corrected

- ASCII `E/B/P` lines and a frame-count TTL were removed. Canonical KEP2 now
  supplies bounded COBS/CRC records, typed identity, acquisition, exact
  sequencing, finite millisecond leases, results, and release.
- The legacy command implemented gaze/lid/blink but could not truthfully claim
  KEP2 pupil, RGB, brightness, expression, identity, or applied-report
  capabilities. Every advertised KEP2 field now reaches the renderer.
- Lease time is a device monotonic timestamp with an exclusive boundary, not
  `120 frames == approximately 2 seconds`.
- Duplicate messages are content-checked and cached; a reused sequence with
  changed content cannot partially apply.
- Disconnect and malformed-frame paths relinquish immediately rather than
  relying on a later frame TTL.
- Gaze/lid smoothing uses elapsed milliseconds and labelled physical rates.
- The legacy framebuffer's unchecked indexing and layout reinterpretation were
  not copied. The canonical renderer uses safe iteration and the driver's
  logical RGB type directly.
- Release keeps overflow checks enabled. No unmeasured “zero-copy” or speed
  claim was carried forward.

## Contradictions and uncertainty retained as explicit gates

Legacy top-level comments/docs named GP2/GP3 while executable source and its
newer config named GP15/GP16. This image follows executable source GP15/GP16,
but wiring remains physically unverified. The legacy right-eye sign was `1`
and explicitly called uncalibrated, so the canonical build refuses to choose a
default. The supplied `kiko.uf2` was not treated as evidence of its source,
identity, or installed behavior and was not flashed.
