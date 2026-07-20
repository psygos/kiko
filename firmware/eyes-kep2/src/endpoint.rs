//! Bounded COBS/CRC byte-stream endpoint around the device controller.

use kiko_eye_protocol::{
    EncodeError, FrameError, MAX_ENCODED_FRAME_BYTES, Message, StreamDecoder, StreamEvent, encode,
};

use crate::controller::{Controller, ControllerError, FirmwareIdentity};
use kiko_eye_protocol::DeviceTimestampMs;

/// One complete encoded KEP2 response, including its trailing zero delimiter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodedResponse {
    bytes: [u8; MAX_ENCODED_FRAME_BYTES],
    length: usize,
}

impl EncodedResponse {
    pub fn try_new(message: Message) -> Result<Self, EncodeError> {
        let mut bytes = [0_u8; MAX_ENCODED_FRAME_BYTES];
        let length = encode(message, &mut bytes)?;
        Ok(Self { bytes, length })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.length]
    }

    pub const fn len(&self) -> usize {
        self.length
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndpointFault {
    Controller(ControllerError),
    Encode(EncodeError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndpointEvent {
    Pending,
    Response(EncodedResponse),
    Dropped(FrameError),
    Fault(EndpointFault),
}

/// The complete allocation-free firmware protocol boundary.
pub struct Kep2Endpoint {
    decoder: StreamDecoder,
    controller: Controller,
}

impl Kep2Endpoint {
    pub const fn new(identity: FirmwareIdentity) -> Self {
        Self {
            decoder: StreamDecoder::new(),
            controller: Controller::new(identity),
        }
    }

    pub const fn controller(&self) -> &Controller {
        &self.controller
    }

    pub fn controller_mut(&mut self) -> &mut Controller {
        &mut self.controller
    }

    /// Feed one byte with distinct receipt and handling timestamps.
    ///
    /// `received_at` is the monotonic timestamp captured when the USB packet
    /// containing this byte became available. `handled_at` is sampled when
    /// the byte reaches this parser. Keeping both prevents a complete frame
    /// that waited behind an earlier response from receiving a fresh lease.
    pub fn push(
        &mut self,
        byte: u8,
        received_at: DeviceTimestampMs,
        handled_at: DeviceTimestampMs,
    ) -> EndpointEvent {
        match self.decoder.push(byte) {
            StreamEvent::Pending => EndpointEvent::Pending,
            StreamEvent::Dropped(error) => match self.controller.on_malformed_frame(handled_at) {
                Ok(()) => EndpointEvent::Dropped(error),
                Err(controller) => EndpointEvent::Fault(EndpointFault::Controller(controller)),
            },
            StreamEvent::Frame(message) => {
                let response =
                    match self
                        .controller
                        .handle_received(message, received_at, handled_at)
                    {
                        Ok(response) => response,
                        Err(error) => {
                            return EndpointEvent::Fault(EndpointFault::Controller(error));
                        }
                    };
                match EncodedResponse::try_new(response) {
                    Ok(encoded) => EndpointEvent::Response(encoded),
                    Err(error) => match self.controller.on_internal_fault(handled_at) {
                        Ok(()) => EndpointEvent::Fault(EndpointFault::Encode(error)),
                        Err(controller) => {
                            EndpointEvent::Fault(EndpointFault::Controller(controller))
                        }
                    },
                }
            }
        }
    }

    /// Reset stream framing and relinquish control on USB disconnect.
    pub fn on_disconnect(&mut self, now: DeviceTimestampMs) -> Result<(), ControllerError> {
        self.decoder = StreamDecoder::new();
        self.controller.on_disconnect(now)
    }
}
