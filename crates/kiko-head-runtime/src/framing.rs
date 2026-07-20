use std::fmt;
use std::time::Duration;

use crate::config::OperationTimeout;
use crate::transport::{
    AsyncByteTransport, MonotonicClock, MonotonicTime, TransportFailure, TransportOperation,
};

const HEADER_BYTE: u8 = 0xff;
const MIN_RESPONSE_BYTES: usize = 6;
/// Largest response used by the qualified protocol: 15 parameters plus the
/// six-byte STS envelope.
pub const MAX_RESPONSE_BYTES: usize = 21;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResponseFrame {
    bytes: [u8; MAX_RESPONSE_BYTES],
    len: u8,
    discarded_noise_bytes: u16,
}

impl ResponseFrame {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes[..usize::from(self.len)]
    }

    pub(crate) const fn discarded_noise_bytes(&self) -> u16 {
        self.discarded_noise_bytes
    }
}

/// Read one exact response without reading beyond it. Prefix noise is bounded;
/// once `FF FF` is observed, identity and declared length belong to that frame
/// and are never silently skipped.
pub(crate) async fn read_response_frame<T, C>(
    transport: &mut T,
    clock: &C,
    timeout: OperationTimeout,
    noise_budget_bytes: u16,
) -> Result<ResponseFrame, FrameReadError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let started_at = clock.now();
    let deadline = started_at
        .duration_since_origin()
        .checked_add(timeout.get())
        .map(MonotonicTime::from_duration_since_origin)
        .ok_or(FrameReadError::ClockOverflow {
            started_at,
            timeout: timeout.get(),
        })?;
    let mut bytes = [0_u8; MAX_RESPONSE_BYTES];
    let mut len = 0_usize;
    let mut discarded = 0_u16;
    let mut last_observed = started_at;

    while len < 2 {
        let byte = read_one(transport, clock, deadline, &mut last_observed, len, None).await?;
        match (len, byte) {
            (0, HEADER_BYTE) => {
                bytes[0] = byte;
                len = 1;
            }
            (0, _) => add_noise(&mut discarded, 1, noise_budget_bytes)?,
            (1, HEADER_BYTE) => {
                bytes[1] = byte;
                len = 2;
            }
            (1, _) => {
                // The retained FF and the mismatching byte are both noise.
                add_noise(&mut discarded, 2, noise_budget_bytes)?;
                len = 0;
            }
            _ => unreachable!("header scanner has only two states"),
        }
    }

    fill_exact(
        transport,
        clock,
        deadline,
        &mut last_observed,
        &mut bytes,
        &mut len,
        4,
    )
    .await?;
    let declared_bytes = usize::from(bytes[3]) + 4;
    if !(MIN_RESPONSE_BYTES..=MAX_RESPONSE_BYTES).contains(&declared_bytes) {
        return Err(FrameReadError::DeclaredLengthOutOfRange {
            length_byte: bytes[3],
            declared_bytes,
            minimum_bytes: MIN_RESPONSE_BYTES,
            maximum_bytes: MAX_RESPONSE_BYTES,
        });
    }
    fill_exact(
        transport,
        clock,
        deadline,
        &mut last_observed,
        &mut bytes,
        &mut len,
        declared_bytes,
    )
    .await?;

    Ok(ResponseFrame {
        bytes,
        len: u8::try_from(len).expect("bounded response capacity fits u8"),
        discarded_noise_bytes: discarded,
    })
}

async fn fill_exact<T, C>(
    transport: &mut T,
    clock: &C,
    deadline: MonotonicTime,
    last_observed: &mut MonotonicTime,
    bytes: &mut [u8; MAX_RESPONSE_BYTES],
    len: &mut usize,
    target: usize,
) -> Result<(), FrameReadError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    while *len < target {
        let remaining = remaining_time(clock, deadline, last_observed, *len, Some(target))?;
        match transport
            .read_some(&mut bytes[*len..target], remaining)
            .await
        {
            Ok(0) => {
                return Err(FrameReadError::Truncated {
                    buffered_bytes: *len,
                    expected_bytes: Some(target),
                });
            }
            Ok(read) => {
                if read > target - *len {
                    return Err(FrameReadError::TransportContractViolation {
                        reported_bytes: read,
                        provided_capacity: target - *len,
                    });
                }
                *len += read;
            }
            Err(source) => {
                return Err(FrameReadError::Transport {
                    source,
                    buffered_bytes: *len,
                    expected_bytes: Some(target),
                });
            }
        }
    }
    Ok(())
}

async fn read_one<T, C>(
    transport: &mut T,
    clock: &C,
    deadline: MonotonicTime,
    last_observed: &mut MonotonicTime,
    buffered_bytes: usize,
    expected_bytes: Option<usize>,
) -> Result<u8, FrameReadError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let remaining = remaining_time(
        clock,
        deadline,
        last_observed,
        buffered_bytes,
        expected_bytes,
    )?;
    let mut byte = [0_u8; 1];
    match transport.read_some(&mut byte, remaining).await {
        Ok(0) => Err(FrameReadError::Truncated {
            buffered_bytes,
            expected_bytes,
        }),
        Ok(1) => Ok(byte[0]),
        Ok(reported_bytes) => Err(FrameReadError::TransportContractViolation {
            reported_bytes,
            provided_capacity: 1,
        }),
        Err(source) => Err(FrameReadError::Transport {
            source,
            buffered_bytes,
            expected_bytes,
        }),
    }
}

fn remaining_time<C: MonotonicClock>(
    clock: &C,
    deadline: MonotonicTime,
    last_observed: &mut MonotonicTime,
    buffered_bytes: usize,
    expected_bytes: Option<usize>,
) -> Result<Duration, FrameReadError> {
    let now = clock.now();
    if now < *last_observed {
        return Err(FrameReadError::NonMonotonicClock {
            previous: *last_observed,
            actual: now,
        });
    }
    *last_observed = now;
    let remaining = deadline
        .duration_since_origin()
        .checked_sub(now.duration_since_origin())
        .ok_or_else(|| FrameReadError::Transport {
            source: TransportFailure::timed_out(TransportOperation::Read, 0),
            buffered_bytes,
            expected_bytes,
        })?;
    if remaining.is_zero() {
        return Err(FrameReadError::Transport {
            source: TransportFailure::timed_out(TransportOperation::Read, 0),
            buffered_bytes,
            expected_bytes,
        });
    }
    Ok(remaining)
}

fn add_noise(discarded: &mut u16, additional: u16, budget: u16) -> Result<(), FrameReadError> {
    let observed = discarded
        .checked_add(additional)
        .expect("configured noise budget bounds the counter");
    if observed > budget {
        return Err(FrameReadError::NoiseBudgetExceeded {
            budget_bytes: budget,
            observed_noise_bytes: observed,
        });
    }
    *discarded = observed;
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameReadError {
    ClockOverflow {
        started_at: MonotonicTime,
        timeout: Duration,
    },
    NonMonotonicClock {
        previous: MonotonicTime,
        actual: MonotonicTime,
    },
    NoiseBudgetExceeded {
        budget_bytes: u16,
        observed_noise_bytes: u16,
    },
    DeclaredLengthOutOfRange {
        length_byte: u8,
        declared_bytes: usize,
        minimum_bytes: usize,
        maximum_bytes: usize,
    },
    Truncated {
        buffered_bytes: usize,
        expected_bytes: Option<usize>,
    },
    Transport {
        source: TransportFailure,
        buffered_bytes: usize,
        expected_bytes: Option<usize>,
    },
    TransportContractViolation {
        reported_bytes: usize,
        provided_capacity: usize,
    },
}

impl fmt::Display for FrameReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not delimit one bounded STS response: {self:?}"
        )
    }
}

impl std::error::Error for FrameReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport { source, .. } => Some(source),
            Self::ClockOverflow { .. }
            | Self::NonMonotonicClock { .. }
            | Self::NoiseBudgetExceeded { .. }
            | Self::DeclaredLengthOutOfRange { .. }
            | Self::Truncated { .. }
            | Self::TransportContractViolation { .. } => None,
        }
    }
}
