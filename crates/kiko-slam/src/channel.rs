use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropPolicy {
    DropNewest,
    DropOldest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SendOutcome {
    Enqueued,
    DroppedNewest,
    DroppedOldest,
    Disconnected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelCapacity(NonZeroUsize);

impl ChannelCapacity {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self(capacity)
    }

    pub fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug)]
pub enum ChannelCapacityError {
    Zero,
}

impl std::fmt::Display for ChannelCapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelCapacityError::Zero => write!(f, "channel capacity must be > 0"),
        }
    }
}

impl std::error::Error for ChannelCapacityError {}

impl TryFrom<usize> for ChannelCapacity {
    type Error = ChannelCapacityError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::new(value)
            .map(ChannelCapacity)
            .ok_or(ChannelCapacityError::Zero)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ChannelStats {
    pub enqueued: u64,
    pub dropped_newest: u64,
    pub dropped_oldest: u64,
    pub disconnected: u64,
}

#[derive(Debug)]
struct ChannelStatsInner {
    enqueued: AtomicU64,
    dropped_newest: AtomicU64,
    dropped_oldest: AtomicU64,
    disconnected: AtomicU64,
}

impl ChannelStatsInner {
    fn snapshot(&self) -> ChannelStats {
        ChannelStats {
            enqueued: self.enqueued.load(Ordering::Relaxed),
            dropped_newest: self.dropped_newest.load(Ordering::Relaxed),
            dropped_oldest: self.dropped_oldest.load(Ordering::Relaxed),
            disconnected: self.disconnected.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChannelStatsHandle {
    inner: Arc<ChannelStatsInner>,
}

impl ChannelStatsHandle {
    pub fn snapshot(&self) -> ChannelStats {
        self.inner.snapshot()
    }
}

#[derive(Debug)]
pub struct DropSender<T> {
    tx: Sender<T>,
    drop_rx: Receiver<T>,
    policy: DropPolicy,
    stats: Arc<ChannelStatsInner>,
}

impl<T> DropSender<T> {
    pub fn try_send(&self, value: T) -> SendOutcome {
        match self.tx.try_send(value) {
            Ok(()) => {
                self.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                SendOutcome::Enqueued
            }
            Err(TrySendError::Full(value)) => match self.policy {
                DropPolicy::DropNewest => {
                    self.stats.dropped_newest.fetch_add(1, Ordering::Relaxed);
                    SendOutcome::DroppedNewest
                }
                DropPolicy::DropOldest => {
                    match self.drop_rx.try_recv() {
                        Ok(_) => {
                            self.stats.dropped_oldest.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(TryRecvError::Empty) => {
                            // A racing receiver may have drained between `Full` and this
                            // eviction attempt. Fall through and retry the send.
                        }
                        Err(TryRecvError::Disconnected) => {
                            self.stats.disconnected.fetch_add(1, Ordering::Relaxed);
                            return SendOutcome::Disconnected;
                        }
                    }
                    match self.tx.try_send(value) {
                        Ok(()) => {
                            self.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                            SendOutcome::Enqueued
                        }
                        Err(TrySendError::Full(_)) => {
                            self.stats.dropped_newest.fetch_add(1, Ordering::Relaxed);
                            SendOutcome::DroppedNewest
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            self.stats.disconnected.fetch_add(1, Ordering::Relaxed);
                            SendOutcome::Disconnected
                        }
                    }
                }
            },
            Err(TrySendError::Disconnected(_)) => {
                self.stats.disconnected.fetch_add(1, Ordering::Relaxed);
                SendOutcome::Disconnected
            }
        }
    }
}

#[derive(Debug)]
pub struct DropReceiver<T> {
    rx: Receiver<T>,
}

impl<T> DropReceiver<T> {
    pub fn recv(&self) -> Result<T, crossbeam_channel::RecvError> {
        self.rx.recv()
    }

    pub fn try_recv(&self) -> Result<T, crossbeam_channel::TryRecvError> {
        self.rx.try_recv()
    }

    pub fn iter(&self) -> crossbeam_channel::Iter<'_, T> {
        self.rx.iter()
    }

    /// Expose the inner crossbeam receiver for use in `crossbeam_channel::select!`.
    pub fn as_receiver(&self) -> &Receiver<T> {
        &self.rx
    }
}

pub fn bounded_channel<T>(
    capacity: ChannelCapacity,
    policy: DropPolicy,
) -> (DropSender<T>, DropReceiver<T>, ChannelStatsHandle) {
    let (tx, rx) = crossbeam_channel::bounded(capacity.get());
    let stats = Arc::new(ChannelStatsInner {
        enqueued: AtomicU64::new(0),
        dropped_newest: AtomicU64::new(0),
        dropped_oldest: AtomicU64::new(0),
        disconnected: AtomicU64::new(0),
    });
    let drop_rx = rx.clone();
    let sender = DropSender {
        tx,
        drop_rx,
        policy,
        stats: stats.clone(),
    };
    let receiver = DropReceiver { rx };
    let handle = ChannelStatsHandle { inner: stats };
    (sender, receiver, handle)
}
