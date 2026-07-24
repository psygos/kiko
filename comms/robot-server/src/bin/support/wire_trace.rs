const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WireTraceSnapshot {
    pub(crate) total_bytes: usize,
    pub(crate) fnv1a64: u64,
    pub(crate) initial_synchronization_delimiter_offset: Option<usize>,
    pub(crate) retained_delimiter_offsets: Vec<usize>,
    pub(crate) current_nonzero_run: usize,
    pub(crate) maximum_completed_nonzero_run: usize,
    pub(crate) retained_start_offset: usize,
    pub(crate) retained_bytes: Vec<u8>,
}

pub(crate) struct WireTrace<const BYTES: usize, const DELIMITERS: usize> {
    bytes: [u8; BYTES],
    next_byte: usize,
    retained_bytes: usize,
    delimiter_offsets: [usize; DELIMITERS],
    next_delimiter: usize,
    retained_delimiters: usize,
    total_bytes: usize,
    current_nonzero_run: usize,
    maximum_completed_nonzero_run: usize,
    initial_synchronization_delimiter_offset: Option<usize>,
    fnv1a64: u64,
}

impl<const BYTES: usize, const DELIMITERS: usize> WireTrace<BYTES, DELIMITERS> {
    pub(crate) const fn new() -> Self {
        Self {
            bytes: [0; BYTES],
            next_byte: 0,
            retained_bytes: 0,
            delimiter_offsets: [0; DELIMITERS],
            next_delimiter: 0,
            retained_delimiters: 0,
            total_bytes: 0,
            current_nonzero_run: 0,
            maximum_completed_nonzero_run: 0,
            initial_synchronization_delimiter_offset: None,
            fnv1a64: FNV1A64_OFFSET_BASIS,
        }
    }

    pub(crate) fn observe(&mut self, byte: u8) -> Option<usize> {
        let offset = self.total_bytes;
        self.total_bytes += 1;
        self.fnv1a64 ^= u64::from(byte);
        self.fnv1a64 = self.fnv1a64.wrapping_mul(FNV1A64_PRIME);

        if BYTES != 0 {
            self.bytes[self.next_byte] = byte;
            self.next_byte = (self.next_byte + 1) % BYTES;
            if self.retained_bytes < BYTES {
                self.retained_bytes += 1;
            }
        }

        if byte == 0 {
            let completed_nonzero_run = self.current_nonzero_run;
            self.maximum_completed_nonzero_run = self
                .maximum_completed_nonzero_run
                .max(self.current_nonzero_run);
            self.current_nonzero_run = 0;
            if DELIMITERS != 0 {
                self.delimiter_offsets[self.next_delimiter] = offset;
                self.next_delimiter = (self.next_delimiter + 1) % DELIMITERS;
                if self.retained_delimiters < DELIMITERS {
                    self.retained_delimiters += 1;
                }
            }
            Some(completed_nonzero_run)
        } else {
            self.current_nonzero_run += 1;
            None
        }
    }

    pub(crate) fn note_initial_synchronization_delimiter(&mut self) {
        self.initial_synchronization_delimiter_offset = self.total_bytes.checked_sub(1);
    }

    pub(crate) const fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub(crate) const fn current_nonzero_run(&self) -> usize {
        self.current_nonzero_run
    }

    pub(crate) fn snapshot(&self) -> WireTraceSnapshot {
        WireTraceSnapshot {
            total_bytes: self.total_bytes,
            fnv1a64: self.fnv1a64,
            initial_synchronization_delimiter_offset: self.initial_synchronization_delimiter_offset,
            retained_delimiter_offsets: self.retained_delimiter_offsets(),
            current_nonzero_run: self.current_nonzero_run,
            maximum_completed_nonzero_run: self.maximum_completed_nonzero_run,
            retained_start_offset: self.total_bytes - self.retained_bytes,
            retained_bytes: self.retained_bytes_in_order(),
        }
    }

    fn retained_bytes_in_order(&self) -> Vec<u8> {
        if BYTES == 0 || self.retained_bytes == 0 {
            return Vec::new();
        }
        let start = if self.retained_bytes == BYTES {
            self.next_byte
        } else {
            0
        };
        (0..self.retained_bytes)
            .map(|index| self.bytes[(start + index) % BYTES])
            .collect()
    }

    fn retained_delimiter_offsets(&self) -> Vec<usize> {
        if DELIMITERS == 0 || self.retained_delimiters == 0 {
            return Vec::new();
        }
        let start = if self.retained_delimiters == DELIMITERS {
            self.next_delimiter
        } else {
            0
        };
        (0..self.retained_delimiters)
            .map(|index| self.delimiter_offsets[(start + index) % DELIMITERS])
            .collect()
    }
}

impl<const BYTES: usize, const DELIMITERS: usize> Default for WireTrace<BYTES, DELIMITERS> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_byte_ring_wraps_in_wire_order() {
        let mut trace = WireTrace::<4, 1>::new();
        for byte in 1..=6 {
            assert_eq!(trace.observe(byte), None);
        }

        let snapshot = trace.snapshot();
        assert_eq!(snapshot.total_bytes, 6);
        assert_eq!(snapshot.retained_start_offset, 2);
        assert_eq!(snapshot.retained_bytes, [3, 4, 5, 6]);
        assert_eq!(snapshot.current_nonzero_run, 6);
    }

    #[test]
    fn delimiter_ring_retains_latest_offsets_and_run_bounds() {
        let mut trace = WireTrace::<8, 2>::new();
        assert_eq!(trace.observe(0), Some(0));
        trace.note_initial_synchronization_delimiter();
        for byte in [1, 0, 2, 3, 0] {
            let _completed_nonzero_run = trace.observe(byte);
        }

        let snapshot = trace.snapshot();
        assert_eq!(snapshot.initial_synchronization_delimiter_offset, Some(0));
        assert_eq!(snapshot.retained_delimiter_offsets, [2, 5]);
        assert_eq!(snapshot.current_nonzero_run, 0);
        assert_eq!(snapshot.maximum_completed_nonzero_run, 2);
    }

    #[test]
    fn fnv1a64_matches_the_standard_hello_vector() {
        let mut trace = WireTrace::<0, 0>::new();
        for byte in b"hello" {
            assert_eq!(trace.observe(*byte), None);
        }

        assert_eq!(trace.snapshot().fnv1a64, 0xa430_d846_80aa_bd0b);
    }

    #[test]
    fn zero_and_nonzero_capacities_remain_strictly_bounded() {
        let mut empty = WireTrace::<0, 0>::new();
        for byte in [0, 1, 0, 2, 0] {
            let _completed_nonzero_run = empty.observe(byte);
        }
        let empty_snapshot = empty.snapshot();
        assert!(empty_snapshot.retained_bytes.is_empty());
        assert!(empty_snapshot.retained_delimiter_offsets.is_empty());
        assert_eq!(empty_snapshot.retained_start_offset, 5);

        let mut bounded = WireTrace::<3, 2>::new();
        for byte in [0, 1, 0, 2, 0, 3] {
            let _completed_nonzero_run = bounded.observe(byte);
        }
        let bounded_snapshot = bounded.snapshot();
        assert_eq!(bounded_snapshot.retained_bytes.len(), 3);
        assert_eq!(bounded_snapshot.retained_delimiter_offsets.len(), 2);
        assert_eq!(bounded_snapshot.retained_start_offset, 3);
    }
}
