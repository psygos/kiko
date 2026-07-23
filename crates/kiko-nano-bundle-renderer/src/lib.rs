//! Offline-only, content-addressed Kiko Nano deployment bundle rendering.
//!
//! The renderer never installs a bundle, opens a device, starts a process, or
//! mutates an existing nonempty destination. It turns one strict discovery
//! record and exact retained source bytes into either an in-memory dry-run plan
//! or a read-only staging tree.

#![forbid(unsafe_code)]

mod input;
mod renderer;

pub use renderer::{
    BundleFileEvidence, BundlePlanEvidence, RenderError, RenderMode, render_bundle,
};
