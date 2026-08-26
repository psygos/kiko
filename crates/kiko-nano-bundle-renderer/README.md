# Kiko Nano bundle renderer

`kiko-nano-bundle-renderer` is the canonical offline builder for production
and wheels-off-qualification configuration bundles. It has exactly two
operations:

- `check` parses the strict render input, reads each exact source once, hashes
  it, renders every derived document in memory, and writes nothing.
- `stage` performs the same work and writes one empty, read-only staging tree.

There is no install, flash, SSH, process-control, serial, USB, camera, or
systemd operation in this crate. A successfully rendered bundle is not
hardware evidence and is not permission to attach wheels.

The strict render input also chooses either no map warm start or production
dataset replay. Dataset replay emits the canonical persisted snapshot/dataset
paths, but it does not claim that replay or live relocalization succeeded.
Qualification bundles reject persisted warm start.

Qualification render-input V4 additionally requires distinct exact frontal
and profile cascades. Its qualification-only head-gaze policy is optional:
absence emits no policy claim, while supplied exact bytes are retained,
size/hash bound into qualification launch V4, and rejected on path/content
aliasing. Bootstrap admits a supplied qualification policy only as
proposal-only. Production render-input V2 emits production launch V4 and
requires two separate exact inputs: the physically reviewed head-gaze policy
and its attended review-evidence record. Offline qualification and runtime
bootstrap cross-bind that evidence digest before physical head gaze can start.

The storage input requires independent nonzero ceilings for cumulative dataset
logical bytes, regular files, and navigation-ingress records, plus a
descriptor-relative post-write free-space floor and terminal reserve. The
renderer rejects a reserve that reaches the dataset byte ceiling or is below
the checked 4096-byte-fragment-rounded sum of the map ceiling, bounded 64 MiB
manifest, and 4 KiB selection. The file ceiling is capped at 65,536 while
finalization retains one bounded monolithic manifest; longer sessions require
a reviewed chunked-manifest format. Rendering proves only contract
consistency; it does not measure filesystem capacity or retention behavior.

See [`docs/nano-bundle-renderer.md`](../../docs/nano-bundle-renderer.md) for
the input contract, deterministic output order, evidence format, and
production motion gate.
