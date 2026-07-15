use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropPolicy {
    DropNewest,
    DropOldest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SendOutcome {
    /// The submitted value was enqueued without evicting a queued value.
    Enqueued,
    /// The submitted value was not enqueued because the channel was full.
    DroppedNewest,
    /// The submitted value was enqueued after this attempt evicted the oldest
    /// queued value.
    DroppedOldest,
    /// The submitted value was not enqueued because the consumer disconnected.
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

/// Monotonic channel event counters that saturate at `u64::MAX`.
///
/// A successful `DropOldest` attempt increments both `enqueued` and
/// `dropped_oldest`, so these fields are not mutually exclusive outcome
/// counts. A snapshot loads each counter independently; cross-field identities
/// are reliable after senders and receivers have been quiesced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChannelStats {
    pub enqueued: u64,
    pub dropped_newest: u64,
    pub dropped_oldest: u64,
    pub disconnected: u64,
}

#[derive(Debug)]
struct ChannelState {
    receiver_alive: AtomicBool,
    enqueued: AtomicU64,
    dropped_newest: AtomicU64,
    dropped_oldest: AtomicU64,
    disconnected: AtomicU64,
}

impl ChannelState {
    fn new() -> Self {
        Self {
            receiver_alive: AtomicBool::new(true),
            enqueued: AtomicU64::new(0),
            dropped_newest: AtomicU64::new(0),
            dropped_oldest: AtomicU64::new(0),
            disconnected: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> ChannelStats {
        ChannelStats {
            enqueued: self.enqueued.load(Ordering::Relaxed),
            dropped_newest: self.dropped_newest.load(Ordering::Relaxed),
            dropped_oldest: self.dropped_oldest.load(Ordering::Relaxed),
            disconnected: self.disconnected.load(Ordering::Relaxed),
        }
    }

    fn record_enqueued(&self) {
        saturating_increment(&self.enqueued);
    }

    fn record_dropped_newest(&self) {
        saturating_increment(&self.dropped_newest);
    }

    fn record_dropped_oldest(&self) {
        saturating_increment(&self.dropped_oldest);
    }

    fn record_disconnected(&self) {
        saturating_increment(&self.disconnected);
    }
}

fn saturating_increment(counter: &AtomicU64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        if value == u64::MAX {
            None
        } else {
            Some(value + 1)
        }
    });
}

#[derive(Clone, Debug)]
pub struct ChannelStatsHandle {
    inner: Arc<ChannelState>,
}

impl ChannelStatsHandle {
    pub fn snapshot(&self) -> ChannelStats {
        self.inner.snapshot()
    }
}

#[derive(Debug)]
enum SenderBehavior<T> {
    DropNewest,
    DropOldest { drop_rx: Receiver<T> },
}

#[derive(Debug)]
pub struct DropSender<T> {
    tx: Sender<T>,
    behavior: SenderBehavior<T>,
    state: Arc<ChannelState>,
}

impl<T> DropSender<T> {
    /// Attempt a nonblocking send.
    ///
    /// The outcome describes the submitted value. Under producer contention,
    /// an attempt can evict an old value and still lose its submitted value to
    /// a racing producer; in that case the outcome is `DroppedNewest` and the
    /// `dropped_oldest` statistic still records the separate eviction event.
    pub fn try_send(&self, value: T) -> SendOutcome {
        if !self.state.receiver_alive.load(Ordering::Acquire) {
            self.state.record_disconnected();
            return SendOutcome::Disconnected;
        }

        match self.tx.try_send(value) {
            Ok(()) => {
                self.state.record_enqueued();
                SendOutcome::Enqueued
            }
            Err(TrySendError::Full(value)) => match &self.behavior {
                SenderBehavior::DropNewest => {
                    self.state.record_dropped_newest();
                    SendOutcome::DroppedNewest
                }
                SenderBehavior::DropOldest { drop_rx } => {
                    if !self.state.receiver_alive.load(Ordering::Acquire) {
                        self.state.record_disconnected();
                        return SendOutcome::Disconnected;
                    }
                    let evicted_oldest = match drop_rx.try_recv() {
                        Ok(_) => {
                            self.state.record_dropped_oldest();
                            true
                        }
                        Err(TryRecvError::Empty) => {
                            // A racing receiver may have drained between `Full` and this
                            // eviction attempt. Fall through and retry the send.
                            false
                        }
                        Err(TryRecvError::Disconnected) => {
                            self.state.record_disconnected();
                            return SendOutcome::Disconnected;
                        }
                    };
                    match self.tx.try_send(value) {
                        Ok(()) => {
                            self.state.record_enqueued();
                            if evicted_oldest {
                                SendOutcome::DroppedOldest
                            } else {
                                SendOutcome::Enqueued
                            }
                        }
                        Err(TrySendError::Full(_)) => {
                            self.state.record_dropped_newest();
                            SendOutcome::DroppedNewest
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            self.state.record_disconnected();
                            SendOutcome::Disconnected
                        }
                    }
                }
            },
            Err(TrySendError::Disconnected(_)) => {
                self.state.record_disconnected();
                SendOutcome::Disconnected
            }
        }
    }
}

#[derive(Debug)]
pub struct DropReceiver<T> {
    rx: Receiver<T>,
    state: Arc<ChannelState>,
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
    ///
    /// Do not retain a clone of this receiver beyond the `DropReceiver`: logical
    /// consumer liveness is tied to this wrapper.
    pub fn as_receiver(&self) -> &Receiver<T> {
        &self.rx
    }
}

impl<T> Drop for DropReceiver<T> {
    fn drop(&mut self) {
        self.state.receiver_alive.store(false, Ordering::Release);
    }
}

pub fn bounded_channel<T>(
    capacity: ChannelCapacity,
    policy: DropPolicy,
) -> (DropSender<T>, DropReceiver<T>, ChannelStatsHandle) {
    let (tx, rx) = crossbeam_channel::bounded(capacity.get());
    let state = Arc::new(ChannelState::new());
    let behavior = match policy {
        DropPolicy::DropNewest => SenderBehavior::DropNewest,
        DropPolicy::DropOldest => SenderBehavior::DropOldest {
            drop_rx: rx.clone(),
        },
    };
    let sender = DropSender {
        tx,
        behavior,
        state: Arc::clone(&state),
    };
    let receiver = DropReceiver {
        rx,
        state: Arc::clone(&state),
    };
    let handle = ChannelStatsHandle { inner: state };
    (sender, receiver, handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    fn capacity(value: usize) -> ChannelCapacity {
        ChannelCapacity::try_from(value).expect("nonzero test capacity")
    }

    fn outcome_index(outcome: SendOutcome) -> usize {
        match outcome {
            SendOutcome::Enqueued => 0,
            SendOutcome::DroppedNewest => 1,
            SendOutcome::DroppedOldest => 2,
            SendOutcome::Disconnected => 3,
        }
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(matches!(
            ChannelCapacity::try_from(0),
            Err(ChannelCapacityError::Zero)
        ));
    }

    #[test]
    fn drop_policy_outcomes_and_event_counters_are_truthful() {
        let (newest_tx, newest_rx, newest_stats) =
            bounded_channel(capacity(1), DropPolicy::DropNewest);
        assert_eq!(newest_tx.try_send(1), SendOutcome::Enqueued);
        assert_eq!(newest_tx.try_send(2), SendOutcome::DroppedNewest);
        assert_eq!(newest_rx.try_recv(), Ok(1));
        assert_eq!(
            newest_stats.snapshot(),
            ChannelStats {
                enqueued: 1,
                dropped_newest: 1,
                ..ChannelStats::default()
            }
        );

        let (oldest_tx, oldest_rx, oldest_stats) =
            bounded_channel(capacity(1), DropPolicy::DropOldest);
        assert_eq!(oldest_tx.try_send(1), SendOutcome::Enqueued);
        assert_eq!(oldest_tx.try_send(2), SendOutcome::DroppedOldest);
        assert_eq!(oldest_rx.try_recv(), Ok(2));
        assert_eq!(
            oldest_stats.snapshot(),
            ChannelStats {
                enqueued: 2,
                dropped_oldest: 1,
                ..ChannelStats::default()
            }
        );
    }

    #[test]
    fn counters_saturate_under_contention() {
        let counter = Arc::new(AtomicU64::new(u64::MAX - 1));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let counter = Arc::clone(&counter);
            workers.push(thread::spawn(move || {
                for _ in 0..64 {
                    saturating_increment(&counter);
                }
            }));
        }
        for worker in workers {
            worker.join().expect("counter worker");
        }
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn contending_drop_oldest_producers_conserve_all_events() {
        const PRODUCERS: usize = 8;
        const SENDS_PER_PRODUCER: usize = 1_000;

        let (sender, receiver, stats) = bounded_channel(capacity(1), DropPolicy::DropOldest);
        let sender = Arc::new(sender);
        let barrier = Arc::new(Barrier::new(PRODUCERS));
        let mut workers = Vec::new();
        for producer in 0..PRODUCERS {
            let sender = Arc::clone(&sender);
            let barrier = Arc::clone(&barrier);
            workers.push(thread::spawn(move || {
                let mut outcomes = [0_u64; 4];
                barrier.wait();
                for sequence in 0..SENDS_PER_PRODUCER {
                    let outcome = sender.try_send((producer, sequence));
                    outcomes[outcome_index(outcome)] += 1;
                }
                outcomes
            }));
        }

        let mut outcomes = [0_u64; 4];
        for worker in workers {
            for (total, count) in outcomes
                .iter_mut()
                .zip(worker.join().expect("producer worker"))
            {
                *total += count;
            }
        }

        let stats = stats.snapshot();
        let remaining = receiver.as_receiver().len() as u64;
        let attempts = (PRODUCERS * SENDS_PER_PRODUCER) as u64;
        assert_eq!(remaining, 1);
        assert_eq!(stats.enqueued, outcomes[0] + outcomes[2]);
        assert_eq!(stats.dropped_newest, outcomes[1]);
        assert_eq!(stats.disconnected, outcomes[3]);
        assert_eq!(stats.disconnected, 0);
        assert!(stats.dropped_oldest >= outcomes[2]);
        assert_eq!(stats.enqueued, stats.dropped_oldest + remaining);
        assert_eq!(
            attempts,
            stats.enqueued + stats.dropped_newest + stats.disconnected
        );
    }

    #[test]
    fn producer_consumer_races_conserve_enqueues_and_losses() {
        const PRODUCERS: usize = 4;
        const SENDS_PER_PRODUCER: usize = 1_000;

        for policy in [DropPolicy::DropNewest, DropPolicy::DropOldest] {
            let (sender, receiver, stats) = bounded_channel(capacity(4), policy);
            let sender = Arc::new(sender);
            let done = Arc::new(AtomicBool::new(false));
            let receiver_done = Arc::clone(&done);
            let consumer = thread::spawn(move || {
                let mut received = 0_u64;
                loop {
                    match receiver.try_recv() {
                        Ok(_) => received += 1,
                        Err(TryRecvError::Empty) if receiver_done.load(Ordering::Acquire) => break,
                        Err(TryRecvError::Empty) => thread::yield_now(),
                        Err(TryRecvError::Disconnected) => {
                            panic!("sender remains alive until consumer completion")
                        }
                    }
                }
                received
            });

            let barrier = Arc::new(Barrier::new(PRODUCERS));
            let mut producers = Vec::new();
            for producer in 0..PRODUCERS {
                let sender = Arc::clone(&sender);
                let barrier = Arc::clone(&barrier);
                producers.push(thread::spawn(move || {
                    barrier.wait();
                    for sequence in 0..SENDS_PER_PRODUCER {
                        sender.try_send((producer, sequence));
                    }
                }));
            }
            for producer in producers {
                producer.join().expect("producer worker");
            }
            done.store(true, Ordering::Release);
            let received = consumer.join().expect("consumer worker");

            let stats = stats.snapshot();
            let attempts = (PRODUCERS * SENDS_PER_PRODUCER) as u64;
            assert_eq!(stats.disconnected, 0);
            assert_eq!(attempts, stats.enqueued + stats.dropped_newest);
            assert_eq!(stats.enqueued, received + stats.dropped_oldest);
            if policy == DropPolicy::DropNewest {
                assert_eq!(stats.dropped_oldest, 0);
            }
        }
    }

    #[test]
    fn receiver_drop_races_preserve_submitted_value_accounting() {
        const PRODUCERS: usize = 4;
        const SENDS_PER_PRODUCER: usize = 500;

        for policy in [DropPolicy::DropNewest, DropPolicy::DropOldest] {
            let (sender, receiver, stats) = bounded_channel(capacity(1), policy);
            let sender = Arc::new(sender);
            let mut outcomes = [0_u64; 4];
            outcomes[outcome_index(sender.try_send((usize::MAX, 0)))] += 1;

            let barrier = Arc::new(Barrier::new(PRODUCERS + 1));
            let drop_barrier = Arc::clone(&barrier);
            let dropper = thread::spawn(move || {
                drop_barrier.wait();
                drop(receiver);
            });
            let mut producers = Vec::new();
            for producer in 0..PRODUCERS {
                let sender = Arc::clone(&sender);
                let barrier = Arc::clone(&barrier);
                producers.push(thread::spawn(move || {
                    let mut outcomes = [0_u64; 4];
                    barrier.wait();
                    for sequence in 0..SENDS_PER_PRODUCER {
                        outcomes[outcome_index(sender.try_send((producer, sequence)))] += 1;
                    }
                    outcomes
                }));
            }
            for producer in producers {
                for (total, count) in outcomes
                    .iter_mut()
                    .zip(producer.join().expect("producer worker"))
                {
                    *total += count;
                }
            }
            dropper.join().expect("receiver dropper");
            outcomes[outcome_index(sender.try_send((usize::MAX, 1)))] += 1;

            let stats = stats.snapshot();
            let attempts = (PRODUCERS * SENDS_PER_PRODUCER + 2) as u64;
            assert_eq!(stats.enqueued, outcomes[0] + outcomes[2]);
            assert_eq!(stats.dropped_newest, outcomes[1]);
            assert_eq!(stats.disconnected, outcomes[3]);
            assert!(stats.disconnected >= 1);
            assert!(stats.dropped_oldest >= outcomes[2]);
            assert_eq!(
                attempts,
                stats.enqueued + stats.dropped_newest + stats.disconnected
            );
        }
    }

    #[test]
    fn sender_reports_disconnect_after_consumer_is_dropped() {
        for policy in [DropPolicy::DropNewest, DropPolicy::DropOldest] {
            let (sender, receiver, stats) = bounded_channel(capacity(1), policy);

            drop(receiver);

            assert_eq!(sender.try_send(1), SendOutcome::Disconnected);
            assert_eq!(
                stats.snapshot(),
                ChannelStats {
                    disconnected: 1,
                    ..ChannelStats::default()
                }
            );
        }
    }
}
