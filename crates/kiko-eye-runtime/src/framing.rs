use std::fmt;
use std::time::Duration;

use kiko_expression_core::MonotonicTimestamp;
use kiko_eye_protocol::{FrameError, MAX_ENCODED_FRAME_BYTES, Message, StreamDecoder, StreamEvent};

use crate::config::OperationTimeout;
use crate::transport::{
    AsyncByteTransport, ClockError, MonotonicClock, TransportFailure, TransportOperation,
};

/// Fixed read capacity; bytes after the first complete record are retained for
/// the next response.
pub const MAX_READ_CHUNK_BYTES: usize = 64;

pub(crate) struct FrameReader {
    decoder: StreamDecoder,
    pending: [u8; MAX_READ_CHUNK_BYTES],
    pending_start: usize,
    pending_end: usize,
    pending_received_at: MonotonicTimestamp,
    record_bytes: usize,
    record_started_at: Option<MonotonicTimestamp>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReceivedMessage {
    pub(crate) message: Message,
    pub(crate) started_at: MonotonicTimestamp,
    pub(crate) received_at: MonotonicTimestamp,
}

impl FrameReader {
    pub(crate) const fn new() -> Self {
        Self {
            decoder: StreamDecoder::new(),
            pending: [0; MAX_READ_CHUNK_BYTES],
            pending_start: 0,
            pending_end: 0,
            pending_received_at: MonotonicTimestamp::ZERO,
            record_bytes: 0,
            record_started_at: None,
        }
    }

    pub(crate) async fn read_message<T, C>(
        &mut self,
        transport: &mut T,
        clock: &C,
        timeout: OperationTimeout,
        empty_delimiter_budget: u8,
    ) -> Result<ReceivedMessage, FrameReadError>
    where
        T: AsyncByteTransport,
        C: MonotonicClock,
    {
        let started_at = clock.now().map_err(FrameReadError::Clock)?;
        let timeout_ns = u64::try_from(timeout.get().as_nanos()).map_err(|_| {
            FrameReadError::DeadlineOverflow {
                started_at_ns: started_at.nanos_since_epoch(),
                timeout,
            }
        })?;
        let deadline_ns = started_at
            .nanos_since_epoch()
            .checked_add(timeout_ns)
            .ok_or(FrameReadError::DeadlineOverflow {
                started_at_ns: started_at.nanos_since_epoch(),
                timeout,
            })?;
        let mut empty_delimiters = 0_u8;

        loop {
            while self.pending_start < self.pending_end {
                let byte = self.pending[self.pending_start];
                self.pending_start += 1;
                if byte != 0 {
                    if self.record_bytes == 0 {
                        self.record_started_at = Some(self.pending_received_at);
                    }
                    self.record_bytes = self
                        .record_bytes
                        .checked_add(1)
                        .ok_or(FrameReadError::EncodedRecordLengthCounterOverflow)?;
                    if self.record_bytes > MAX_ENCODED_FRAME_BYTES - 1 {
                        return Err(FrameReadError::Malformed {
                            source: FrameError::EncodedRecordTooLong {
                                observed_at_least: self.record_bytes,
                                maximum: MAX_ENCODED_FRAME_BYTES - 1,
                            },
                        });
                    }
                }
                let event = self.decoder.push(byte);
                let record_started_at = if byte == 0 {
                    self.record_bytes = 0;
                    self.record_started_at.take()
                } else {
                    self.record_started_at
                };
                match event {
                    StreamEvent::Pending => {}
                    StreamEvent::Frame(message) => {
                        return Ok(ReceivedMessage {
                            message,
                            started_at: record_started_at
                                .ok_or(FrameReadError::FrameWithoutStartTimestamp)?,
                            received_at: self.pending_received_at,
                        });
                    }
                    StreamEvent::Dropped(FrameError::EmptyRecord) => {
                        empty_delimiters = empty_delimiters.checked_add(1).ok_or(
                            FrameReadError::EmptyDelimiterBudgetExceeded {
                                budget: empty_delimiter_budget,
                                observed: u8::MAX,
                            },
                        )?;
                        if empty_delimiters > empty_delimiter_budget {
                            return Err(FrameReadError::EmptyDelimiterBudgetExceeded {
                                budget: empty_delimiter_budget,
                                observed: empty_delimiters,
                            });
                        }
                    }
                    StreamEvent::Dropped(source) => {
                        return Err(FrameReadError::Malformed { source });
                    }
                }
            }

            self.pending_start = 0;
            self.pending_end = 0;
            let remaining = remaining(clock, deadline_ns)?;
            match transport.read_some(&mut self.pending, remaining).await {
                Ok(0) => {
                    return Err(FrameReadError::EndOfStream {
                        encoded_record_bytes: self.record_bytes,
                    });
                }
                Ok(read) if read <= self.pending.len() => {
                    let received_at = clock.now().map_err(FrameReadError::Clock)?;
                    if received_at.nanos_since_epoch() >= deadline_ns {
                        return Err(FrameReadError::Transport {
                            source: TransportFailure::timed_out(TransportOperation::Read, read),
                            encoded_record_bytes: self.record_bytes,
                        });
                    }
                    self.pending_received_at = received_at;
                    self.pending_end = read;
                }
                Ok(read) => {
                    return Err(FrameReadError::TransportContractViolation {
                        reported_bytes: read,
                        provided_capacity: self.pending.len(),
                    });
                }
                Err(source) => {
                    if source.operation() != TransportOperation::Read {
                        return Err(FrameReadError::TransportOperationMismatch {
                            expected: TransportOperation::Read,
                            source,
                        });
                    }
                    return Err(FrameReadError::Transport {
                        source,
                        encoded_record_bytes: self.record_bytes,
                    });
                }
            }
        }
    }
}

fn remaining<C: MonotonicClock>(clock: &C, deadline_ns: u64) -> Result<Duration, FrameReadError> {
    let now = clock.now().map_err(FrameReadError::Clock)?;
    let Some(remaining_ns) = deadline_ns.checked_sub(now.nanos_since_epoch()) else {
        return Err(FrameReadError::Transport {
            source: TransportFailure::timed_out(TransportOperation::Read, 0),
            encoded_record_bytes: 0,
        });
    };
    if remaining_ns == 0 {
        return Err(FrameReadError::Transport {
            source: TransportFailure::timed_out(TransportOperation::Read, 0),
            encoded_record_bytes: 0,
        });
    }
    Ok(Duration::from_nanos(remaining_ns))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameReadError {
    Clock(ClockError),
    DeadlineOverflow {
        started_at_ns: u64,
        timeout: OperationTimeout,
    },
    Transport {
        source: TransportFailure,
        encoded_record_bytes: usize,
    },
    EndOfStream {
        encoded_record_bytes: usize,
    },
    EmptyDelimiterBudgetExceeded {
        budget: u8,
        observed: u8,
    },
    Malformed {
        source: FrameError,
    },
    TransportContractViolation {
        reported_bytes: usize,
        provided_capacity: usize,
    },
    TransportOperationMismatch {
        expected: TransportOperation,
        source: TransportFailure,
    },
    EncodedRecordLengthCounterOverflow,
    FrameWithoutStartTimestamp,
}

impl fmt::Display for FrameReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not read one bounded KEP2 record: {self:?}"
        )
    }
}

impl std::error::Error for FrameReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Clock(source) => Some(source),
            Self::Transport { source, .. } => Some(source),
            Self::TransportOperationMismatch { source, .. } => Some(source),
            Self::Malformed { source } => Some(source),
            _ => None,
        }
    }
}

impl Default for FrameReader {
    fn default() -> Self {
        Self::new()
    }
}
