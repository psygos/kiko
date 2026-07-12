use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

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
    pub current_depth: usize,
    pub max_depth: usize,
}

#[derive(Debug)]
struct ChannelStatsInner {
    enqueued: AtomicU64,
    dropped_newest: AtomicU64,
    dropped_oldest: AtomicU64,
    disconnected: AtomicU64,
    current_depth: AtomicUsize,
    max_depth: AtomicUsize,
}

#[derive(Debug)]
struct ReceiverLease {
    stats: Arc<ChannelStatsInner>,
}

impl Drop for ReceiverLease {
    fn drop(&mut self) {
        self.stats.current_depth.store(0, Ordering::Relaxed);
    }
}

impl ChannelStatsInner {
    fn snapshot(&self) -> ChannelStats {
        ChannelStats {
            enqueued: self.enqueued.load(Ordering::Relaxed),
            dropped_newest: self.dropped_newest.load(Ordering::Relaxed),
            dropped_oldest: self.dropped_oldest.load(Ordering::Relaxed),
            disconnected: self.disconnected.load(Ordering::Relaxed),
            current_depth: self.current_depth.load(Ordering::Relaxed),
            max_depth: self.max_depth.load(Ordering::Relaxed),
        }
    }

    fn on_enqueue(&self) {
        let depth = self.current_depth.fetch_add(1, Ordering::Relaxed) + 1;
        let mut observed_max = self.max_depth.load(Ordering::Relaxed);
        while depth > observed_max {
            match self.max_depth.compare_exchange_weak(
                observed_max,
                depth,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => observed_max = actual,
            }
        }
    }

    fn on_dequeue(&self) {
        let _ = self
            .current_depth
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |depth| {
                Some(depth.saturating_sub(1))
            });
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
    receiver_lease: Weak<ReceiverLease>,
    policy: DropPolicy,
    stats: Arc<ChannelStatsInner>,
}

impl<T> DropSender<T> {
    pub fn try_send(&self, value: T) -> SendOutcome {
        let Some(_receiver_lease) = self.receiver_lease.upgrade() else {
            self.stats.disconnected.fetch_add(1, Ordering::Relaxed);
            return SendOutcome::Disconnected;
        };

        match self.tx.try_send(value) {
            Ok(()) => {
                self.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                self.stats.on_enqueue();
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
                            self.stats.on_dequeue();
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
                            self.stats.on_enqueue();
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
    _lease: Arc<ReceiverLease>,
    stats: Arc<ChannelStatsInner>,
}

impl<T> DropReceiver<T> {
    pub fn recv(&self) -> Result<T, crossbeam_channel::RecvError> {
        let value = self.rx.recv()?;
        self.stats.on_dequeue();
        Ok(value)
    }

    pub fn try_recv(&self) -> Result<T, crossbeam_channel::TryRecvError> {
        let value = self.rx.try_recv()?;
        self.stats.on_dequeue();
        Ok(value)
    }

    pub fn iter(&self) -> DropIter<'_, T> {
        DropIter {
            inner: self.rx.iter(),
            stats: Arc::clone(&self.stats),
        }
    }
}

pub struct DropIter<'a, T> {
    inner: crossbeam_channel::Iter<'a, T>,
    stats: Arc<ChannelStatsInner>,
}

impl<'a, T> Iterator for DropIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.inner.next()?;
        self.stats.on_dequeue();
        Some(value)
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
        current_depth: AtomicUsize::new(0),
        max_depth: AtomicUsize::new(0),
    });
    let receiver_lease = Arc::new(ReceiverLease {
        stats: Arc::clone(&stats),
    });
    let drop_rx = rx.clone();
    let sender = DropSender {
        tx,
        drop_rx,
        receiver_lease: Arc::downgrade(&receiver_lease),
        policy,
        stats: stats.clone(),
    };
    let receiver = DropReceiver {
        rx,
        _lease: receiver_lease,
        stats: Arc::clone(&stats),
    };
    let handle = ChannelStatsHandle { inner: stats };
    (sender, receiver, handle)
}

#[cfg(test)]
mod tests {
    use super::{ChannelCapacity, DropPolicy, SendOutcome, bounded_channel};

    #[test]
    fn channel_stats_track_depth_and_high_water_mark() {
        let (tx, rx, stats) = bounded_channel(
            ChannelCapacity::try_from(2).expect("capacity"),
            DropPolicy::DropNewest,
        );
        assert!(matches!(tx.try_send(1_u8), SendOutcome::Enqueued));
        assert!(matches!(tx.try_send(2_u8), SendOutcome::Enqueued));
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.current_depth, 2);
        assert_eq!(snapshot.max_depth, 2);

        let value = rx.recv().expect("recv");
        assert_eq!(value, 1);
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.current_depth, 1);
        assert_eq!(snapshot.max_depth, 2);
    }

    #[test]
    fn sender_reports_disconnected_after_consumer_is_dropped() {
        for policy in [DropPolicy::DropNewest, DropPolicy::DropOldest] {
            let (tx, rx, stats) =
                bounded_channel(ChannelCapacity::try_from(1).expect("capacity"), policy);
            assert!(matches!(tx.try_send(1_u8), SendOutcome::Enqueued));

            drop(rx);

            assert!(matches!(tx.try_send(2_u8), SendOutcome::Disconnected));
            let snapshot = stats.snapshot();
            assert_eq!(snapshot.disconnected, 1);
            assert_eq!(snapshot.current_depth, 0);
        }
    }
}
