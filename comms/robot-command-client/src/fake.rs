use crate::{MonotonicClock, MonotonicInstant, V2CommandTransport};
use robot_protocol::v2::{Message, MessageKind};
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

#[derive(Clone, Debug, Default)]
pub struct FakeClock {
    nanoseconds: Arc<AtomicU64>,
}

impl FakeClock {
    pub fn new(nanoseconds: u64) -> Self {
        Self {
            nanoseconds: Arc::new(AtomicU64::new(nanoseconds)),
        }
    }

    pub fn set_nanos(&self, nanoseconds: u64) {
        self.nanoseconds.store(nanoseconds, Ordering::SeqCst);
    }

    pub fn advance(&self, duration: Duration) -> Result<(), FakeTransportError> {
        let delta = u64::try_from(duration.as_nanos())
            .map_err(|_| FakeTransportError::ClockArithmeticOverflow)?;
        self.nanoseconds
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                current.checked_add(delta)
            })
            .map(|_| ())
            .map_err(|_| FakeTransportError::ClockArithmeticOverflow)
    }
}

impl MonotonicClock for FakeClock {
    fn now(&self) -> MonotonicInstant {
        MonotonicInstant::from_nanos_since_clock_start(u128::from(
            self.nanoseconds.load(Ordering::SeqCst),
        ))
    }
}

#[derive(Clone, Debug)]
pub enum FakeStep {
    Respond {
        expected_request: MessageKind,
        after: Duration,
        response: Message,
    },
    Fail {
        expected_request: MessageKind,
        after: Duration,
        detail: &'static str,
    },
}

impl FakeStep {
    pub const fn respond(
        expected_request: MessageKind,
        after: Duration,
        response: Message,
    ) -> Self {
        Self::Respond {
            expected_request,
            after,
            response,
        }
    }

    pub const fn fail(
        expected_request: MessageKind,
        after: Duration,
        detail: &'static str,
    ) -> Self {
        Self::Fail {
            expected_request,
            after,
            detail,
        }
    }

    const fn expected_request(&self) -> MessageKind {
        match self {
            Self::Respond {
                expected_request, ..
            }
            | Self::Fail {
                expected_request, ..
            } => *expected_request,
        }
    }

    const fn after(&self) -> Duration {
        match self {
            Self::Respond { after, .. } | Self::Fail { after, .. } => *after,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FakeExchangeRecord {
    request: Message,
    timeout: Duration,
}

impl FakeExchangeRecord {
    pub const fn request(self) -> Message {
        self.request
    }

    pub const fn timeout(self) -> Duration {
        self.timeout
    }
}

#[derive(Debug, Default)]
struct FakeState {
    steps: VecDeque<FakeStep>,
    exchanges: Vec<FakeExchangeRecord>,
}

#[derive(Debug)]
pub struct FakeTransport {
    clock: FakeClock,
    state: Arc<Mutex<FakeState>>,
}

#[derive(Clone, Debug)]
pub struct FakeTransportProbe {
    state: Arc<Mutex<FakeState>>,
}

impl FakeTransport {
    pub fn scripted(
        clock: FakeClock,
        steps: impl IntoIterator<Item = FakeStep>,
    ) -> (Self, FakeTransportProbe) {
        let state = Arc::new(Mutex::new(FakeState {
            steps: steps.into_iter().collect(),
            exchanges: Vec::new(),
        }));
        (
            Self {
                clock,
                state: Arc::clone(&state),
            },
            FakeTransportProbe { state },
        )
    }
}

impl FakeTransportProbe {
    pub fn push(&self, step: FakeStep) {
        lock(&self.state).steps.push_back(step);
    }

    pub fn exchanges(&self) -> Vec<FakeExchangeRecord> {
        lock(&self.state).exchanges.clone()
    }

    pub fn remaining_steps(&self) -> usize {
        lock(&self.state).steps.len()
    }
}

impl V2CommandTransport for FakeTransport {
    type Error = FakeTransportError;

    fn exchange_once(
        &mut self,
        request: Message,
        timeout: Duration,
    ) -> Result<Message, Self::Error> {
        let step = {
            let mut state = lock(&self.state);
            state
                .exchanges
                .push(FakeExchangeRecord { request, timeout });
            state
                .steps
                .pop_front()
                .ok_or(FakeTransportError::UnexpectedRequest {
                    actual: request.kind(),
                })?
        };
        if step.expected_request() != request.kind() {
            return Err(FakeTransportError::RequestKindMismatch {
                expected: step.expected_request(),
                actual: request.kind(),
            });
        }
        self.clock.advance(step.after())?;
        match step {
            FakeStep::Respond { response, .. } => Ok(response),
            FakeStep::Fail { detail, .. } => Err(FakeTransportError::Injected { detail }),
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FakeTransportError {
    UnexpectedRequest {
        actual: MessageKind,
    },
    RequestKindMismatch {
        expected: MessageKind,
        actual: MessageKind,
    },
    Injected {
        detail: &'static str,
    },
    ClockArithmeticOverflow,
}

impl fmt::Display for FakeTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedRequest { actual } => {
                write!(formatter, "unexpected unscripted {actual:?} request")
            }
            Self::RequestKindMismatch { expected, actual } => write!(
                formatter,
                "fake transport expected {expected:?}, received {actual:?}"
            ),
            Self::Injected { detail } => write!(formatter, "injected transport failure: {detail}"),
            Self::ClockArithmeticOverflow => {
                formatter.write_str("fake clock arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for FakeTransportError {}
