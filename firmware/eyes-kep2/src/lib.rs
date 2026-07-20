#![no_std]
#![forbid(unsafe_code)]

//! Canonical KEP2 eye-firmware boundary and deterministic renderer.
//!
//! The library is host-testable and contains no allocator, USB driver, clock,
//! or RP2350 register access. The hardware image in `src/bin/rp2350.rs` owns
//! those boundaries and parses bytes into these domain types exactly once.

pub mod controller;
pub mod endpoint;
pub mod geometry;
pub mod renderer;

pub use controller::{
    ACQUIRE_TO_FIRST_INTENT_MS, Controller, ControllerError, FallbackCause, FirmwareIdentity,
    IdentityError, InboundKind, OutputState, SUPPORTED_CAPABILITIES_BITS,
};
pub use endpoint::{EncodedResponse, EndpointEvent, EndpointFault, Kep2Endpoint};
pub use renderer::{
    BRIGHTNESS_CEILING, EyeFrame, EyeRenderer, FRAME_RATE_HZ, LEDS_PER_EYE, MountingSign,
    MountingSignError, RenderError,
};
