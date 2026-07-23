//! Allocation-free, record-preserving UART transmit admission.
//!
//! Stop confirmations and applied-control receipts own independent capacity
//! from periodic telemetry. Selection changes only after a COBS delimiter, so
//! priority never interleaves two records on the wire.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxTrafficClass {
    HostStopResult,
    AppliedControl,
    BestEffort,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxAdmissionError {
    EmptyRecord,
    MissingFinalDelimiter,
    EmbeddedDelimiter,
    QueueFull {
        class: TxTrafficClass,
        required_bytes: usize,
        available_bytes: usize,
    },
}

struct FixedByteQueue<const BYTES: usize> {
    bytes: [u8; BYTES],
    head: usize,
    len: usize,
}

impl<const BYTES: usize> FixedByteQueue<BYTES> {
    const fn new() -> Self {
        Self {
            bytes: [0; BYTES],
            head: 0,
            len: 0,
        }
    }

    const fn len(&self) -> usize {
        self.len
    }

    const fn available(&self) -> usize {
        BYTES - self.len
    }

    fn push_record(&mut self, record: &[u8]) -> bool {
        if record.len() > self.available() || BYTES == 0 {
            return false;
        }
        for &byte in record {
            let tail = (self.head + self.len) % BYTES;
            self.bytes[tail] = byte;
            self.len += 1;
        }
        true
    }

    fn pop(&mut self) -> Option<u8> {
        if self.len == 0 || BYTES == 0 {
            return None;
        }
        let byte = self.bytes[self.head];
        self.head = (self.head + 1) % BYTES;
        self.len -= 1;
        Some(byte)
    }

    const fn peek(&self) -> Option<u8> {
        if self.len == 0 || BYTES == 0 {
            None
        } else {
            Some(self.bytes[self.head])
        }
    }
}

pub struct PriorityTxScheduler<
    const STOP_BYTES: usize,
    const APPLIED_BYTES: usize,
    const BEST_EFFORT_BYTES: usize,
> {
    stop: FixedByteQueue<STOP_BYTES>,
    applied: FixedByteQueue<APPLIED_BYTES>,
    best_effort: FixedByteQueue<BEST_EFFORT_BYTES>,
    active: Option<TxTrafficClass>,
}

impl<const STOP_BYTES: usize, const APPLIED_BYTES: usize, const BEST_EFFORT_BYTES: usize>
    PriorityTxScheduler<STOP_BYTES, APPLIED_BYTES, BEST_EFFORT_BYTES>
{
    pub const fn new() -> Self {
        Self {
            stop: FixedByteQueue::new(),
            applied: FixedByteQueue::new(),
            best_effort: FixedByteQueue::new(),
            active: None,
        }
    }

    pub fn try_enqueue_record(
        &mut self,
        class: TxTrafficClass,
        record: &[u8],
    ) -> Result<(), TxAdmissionError> {
        validate_record(record)?;
        let (available_bytes, queued) = match class {
            TxTrafficClass::HostStopResult => {
                let available = self.stop.available();
                (available, self.stop.push_record(record))
            }
            TxTrafficClass::AppliedControl => {
                let available = self.applied.available();
                (available, self.applied.push_record(record))
            }
            TxTrafficClass::BestEffort => {
                let available = self.best_effort.available();
                (available, self.best_effort.push_record(record))
            }
        };
        if queued {
            Ok(())
        } else {
            Err(TxAdmissionError::QueueFull {
                class,
                required_bytes: record.len(),
                available_bytes,
            })
        }
    }

    const fn selected_class(&self) -> TxTrafficClass {
        match self.active {
            Some(class) => class,
            None if self.stop.len() != 0 => TxTrafficClass::HostStopResult,
            None if self.applied.len() != 0 => TxTrafficClass::AppliedControl,
            None => TxTrafficClass::BestEffort,
        }
    }

    pub const fn peek_byte(&self) -> Option<u8> {
        match self.selected_class() {
            TxTrafficClass::HostStopResult => self.stop.peek(),
            TxTrafficClass::AppliedControl => self.applied.peek(),
            TxTrafficClass::BestEffort => self.best_effort.peek(),
        }
    }

    pub fn consume_byte(&mut self) -> Option<u8> {
        let class = self.selected_class();
        let byte = match class {
            TxTrafficClass::HostStopResult => self.stop.pop(),
            TxTrafficClass::AppliedControl => self.applied.pop(),
            TxTrafficClass::BestEffort => self.best_effort.pop(),
        };
        match byte {
            Some(0) => self.active = None,
            Some(_) => self.active = Some(class),
            None => self.active = None,
        }
        byte
    }

    pub fn dequeue_byte(&mut self) -> Option<u8> {
        self.consume_byte()
    }

    pub const fn queued_bytes(&self) -> usize {
        self.stop.len() + self.applied.len() + self.best_effort.len()
    }

    pub const fn queued_bytes_for(&self, class: TxTrafficClass) -> usize {
        match class {
            TxTrafficClass::HostStopResult => self.stop.len(),
            TxTrafficClass::AppliedControl => self.applied.len(),
            TxTrafficClass::BestEffort => self.best_effort.len(),
        }
    }
}

impl<const STOP_BYTES: usize, const APPLIED_BYTES: usize, const BEST_EFFORT_BYTES: usize> Default
    for PriorityTxScheduler<STOP_BYTES, APPLIED_BYTES, BEST_EFFORT_BYTES>
{
    fn default() -> Self {
        Self::new()
    }
}

fn validate_record(record: &[u8]) -> Result<(), TxAdmissionError> {
    let Some((&delimiter, body)) = record.split_last() else {
        return Err(TxAdmissionError::EmptyRecord);
    };
    if delimiter != 0 {
        return Err(TxAdmissionError::MissingFinalDelimiter);
    }
    if body.contains(&0) {
        return Err(TxAdmissionError::EmbeddedDelimiter);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;

    type Scheduler = PriorityTxScheduler<8, 12, 8>;

    fn drain(scheduler: &mut Scheduler) -> std::vec::Vec<u8> {
        let mut bytes = std::vec::Vec::new();
        while let Some(byte) = scheduler.dequeue_byte() {
            bytes.push(byte);
        }
        bytes
    }

    #[test]
    fn best_effort_saturation_cannot_consume_stop_or_applied_capacity() {
        let mut scheduler = Scheduler::new();
        scheduler
            .try_enqueue_record(TxTrafficClass::BestEffort, &[1, 2, 3, 4, 5, 6, 7, 0])
            .expect("best-effort queue exactly fills");
        assert!(matches!(
            scheduler.try_enqueue_record(TxTrafficClass::BestEffort, &[8, 0]),
            Err(TxAdmissionError::QueueFull {
                class: TxTrafficClass::BestEffort,
                ..
            })
        ));
        scheduler
            .try_enqueue_record(TxTrafficClass::AppliedControl, &[9, 10, 0])
            .expect("applied-control capacity is independent");
        scheduler
            .try_enqueue_record(TxTrafficClass::HostStopResult, &[11, 12, 0])
            .expect("stop-result capacity is independent");

        assert_eq!(
            drain(&mut scheduler),
            [11, 12, 0, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 0]
        );
    }

    #[test]
    fn a_started_record_is_never_interleaved_but_next_record_observes_priority() {
        let mut scheduler = Scheduler::new();
        scheduler
            .try_enqueue_record(TxTrafficClass::BestEffort, &[1, 2, 3, 0])
            .expect("best-effort record");
        assert_eq!(scheduler.dequeue_byte(), Some(1));
        scheduler
            .try_enqueue_record(TxTrafficClass::HostStopResult, &[9, 8, 0])
            .expect("stop record");
        assert_eq!(drain(&mut scheduler), [2, 3, 0, 9, 8, 0]);
    }

    #[test]
    fn applied_result_saturation_still_preserves_the_stop_result_lane() {
        let mut scheduler = Scheduler::new();
        scheduler
            .try_enqueue_record(
                TxTrafficClass::AppliedControl,
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
            )
            .expect("applied lane exactly fills");
        assert!(matches!(
            scheduler.try_enqueue_record(TxTrafficClass::AppliedControl, &[12, 0]),
            Err(TxAdmissionError::QueueFull {
                class: TxTrafficClass::AppliedControl,
                ..
            })
        ));
        scheduler
            .try_enqueue_record(TxTrafficClass::HostStopResult, &[99, 0])
            .expect("stop lane remains independent");
        assert_eq!(
            drain(&mut scheduler),
            [99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
        );
    }

    #[test]
    fn malformed_records_never_partially_enter_a_queue() {
        let mut scheduler = Scheduler::new();
        assert_eq!(
            scheduler.try_enqueue_record(TxTrafficClass::HostStopResult, &[]),
            Err(TxAdmissionError::EmptyRecord)
        );
        assert_eq!(
            scheduler.try_enqueue_record(TxTrafficClass::AppliedControl, &[1, 2]),
            Err(TxAdmissionError::MissingFinalDelimiter)
        );
        assert_eq!(
            scheduler.try_enqueue_record(TxTrafficClass::BestEffort, &[1, 0, 2, 0]),
            Err(TxAdmissionError::EmbeddedDelimiter)
        );
        assert_eq!(scheduler.queued_bytes(), 0);
    }
}
