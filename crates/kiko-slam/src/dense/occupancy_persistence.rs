//! Versioned, bounded persistence for host-side occupancy snapshots.
//!
//! Version 1 is a deterministic little-endian format. Its fixed-size header
//! records the metric occupancy geometry, the rigid visual-world transform,
//! the accepted height interval, row order, cell encoding, and map revision.
//! The payload preserves each row-major cell class byte exactly and a CRC-32
//! trailer covers both header and payload.
//!
//! Version 1's 192-byte header is fixed as follows (all multi-byte scalars are
//! little-endian and every reserved byte must be zero):
//!
//! | Byte range | Field |
//! |---|---|
//! | `0..8` | `KIKO2DM\0` magic |
//! | `8..10`, `10..12` | format version, header length |
//! | `12..16` | reserved |
//! | `16`, `17`, `18`, `19` | metric `[x, y, height]` frame, increasing-`y` row order, `u8` class encoding, reserved |
//! | `20..28`, `28..32` | `u32` width and height, reserved |
//! | `32..48` | `u64` cell count and payload byte length |
//! | `48..72` | `f64` metres/cell resolution and lower `[x, y]` origin |
//! | `72..144` | row-major `f64` world-to-occupancy rotation |
//! | `144..168` | `f64` world-to-occupancy translation in metres |
//! | `168..184` | `f64` minimum and maximum accepted height in metres |
//! | `184..192` | `u64` map revision |
//!
//! The header is followed by exactly `width * height` class bytes and a
//! little-endian CRC-32/ISO-HDLC checksum over the header and class bytes.
//!
//! [`crate::map::MapInstanceId`] is deliberately absent. It is a process-local
//! freshness token, so a loaded snapshot has no map instance ID. It can be
//! rebound only after exact comparison with the final occupancy output of a
//! retained dataset replay; that comparison is not live visual relocalization.

use std::collections::TryReserveError;
use std::ffi::OsString;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::occupancy::{
    HeightRangeError, HeightRangeMeters, OccupancyGridGeometry, OccupancyGridGeometryError,
    OccupancyGridSnapshot, WorldToOccupancy, WorldToOccupancyError,
};
use crate::map::{MapInstanceId, MapSnapshot};

const MAGIC: [u8; 8] = *b"KIKO2DM\0";
pub const OCCUPANCY_MAP_FORMAT_VERSION: u16 = 1;
pub const OCCUPANCY_MAP_HEADER_BYTES: usize = 192;
pub const OCCUPANCY_MAP_CHECKSUM_BYTES: usize = 4;
pub const OCCUPANCY_MAP_FIXED_OVERHEAD_BYTES: usize =
    OCCUPANCY_MAP_HEADER_BYTES + OCCUPANCY_MAP_CHECKSUM_BYTES;

const COORDINATE_FRAME_METRIC_XY_HEIGHT: u8 = 1;
const ROW_ORDER_INCREASING_Y: u8 = 1;
const CELL_ENCODING_CLASS_U8: u8 = 1;
const TEMP_CREATE_ATTEMPTS: usize = 32;
static NEXT_TEMPORARY_FILE_NONCE: AtomicU64 = AtomicU64::new(0);

/// Explicit allocation bound for decoding one persisted occupancy map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccupancyMapLimits {
    maximum_cells: NonZeroUsize,
    maximum_encoded_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyMapLimitsError {
    ZeroMaximumCells,
    EncodedLengthOverflow { maximum_cells: usize },
}

impl fmt::Display for OccupancyMapLimitsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMaximumCells => {
                write!(
                    formatter,
                    "occupancy-map maximum cell count must be nonzero"
                )
            }
            Self::EncodedLengthOverflow { maximum_cells } => write!(
                formatter,
                "occupancy-map maximum of {maximum_cells} cells has no addressable encoded length"
            ),
        }
    }
}

impl std::error::Error for OccupancyMapLimitsError {}

impl OccupancyMapLimits {
    pub fn try_new(maximum_cells: usize) -> Result<Self, OccupancyMapLimitsError> {
        let maximum_cells =
            NonZeroUsize::new(maximum_cells).ok_or(OccupancyMapLimitsError::ZeroMaximumCells)?;
        let maximum_encoded_bytes = OCCUPANCY_MAP_FIXED_OVERHEAD_BYTES
            .checked_add(maximum_cells.get())
            .ok_or(OccupancyMapLimitsError::EncodedLengthOverflow {
                maximum_cells: maximum_cells.get(),
            })?;
        Ok(Self {
            maximum_cells,
            maximum_encoded_bytes,
        })
    }

    pub fn maximum_cells(self) -> usize {
        self.maximum_cells.get()
    }

    pub fn maximum_encoded_bytes(self) -> usize {
        self.maximum_encoded_bytes
    }
}

#[derive(Debug)]
pub enum OccupancyMapEncodeError {
    EncodedLengthOverflow {
        cells: usize,
    },
    AllocationFailed {
        requested_bytes: usize,
        source: TryReserveError,
    },
}

impl fmt::Display for OccupancyMapEncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EncodedLengthOverflow { cells } => write!(
                formatter,
                "occupancy snapshot with {cells} cells has no addressable encoded length"
            ),
            Self::AllocationFailed {
                requested_bytes, ..
            } => write!(
                formatter,
                "failed to allocate {requested_bytes} bytes for encoded occupancy map"
            ),
        }
    }
}

impl std::error::Error for OccupancyMapEncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AllocationFailed { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum OccupancyMapDecodeError {
    InputExceedsLimit {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    Truncated {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    TrailingBytes {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    MagicMismatch {
        actual: [u8; 8],
    },
    UnsupportedVersion {
        version: u16,
    },
    InvalidHeaderLength {
        bytes: u16,
    },
    NonZeroReservedByte {
        offset: usize,
        value: u8,
    },
    UnsupportedCoordinateFrame {
        code: u8,
    },
    UnsupportedRowOrder {
        code: u8,
    },
    UnsupportedCellEncoding {
        code: u8,
    },
    DeclaredCellCountMismatch {
        declared: u64,
        computed: u64,
    },
    PayloadLengthMismatch {
        declared: u64,
        computed: u64,
    },
    Geometry(OccupancyGridGeometryError),
    WorldToOccupancy(WorldToOccupancyError),
    HeightRange(HeightRangeError),
    ChecksumMismatch {
        stored: u32,
        computed: u32,
    },
    UnsupportedCellClass {
        index: usize,
        class_id: u8,
    },
    AllocationFailed {
        cells: usize,
        source: TryReserveError,
    },
    EncodedLengthOverflow {
        cells: usize,
    },
}

impl fmt::Display for OccupancyMapDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExceedsLimit {
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "occupancy-map input is {actual_bytes} bytes, exceeding configured maximum {maximum_bytes}"
            ),
            Self::Truncated {
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "occupancy-map input is truncated: expected {expected_bytes} bytes, got {actual_bytes}"
            ),
            Self::TrailingBytes {
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "occupancy-map input has trailing data: expected {expected_bytes} bytes, got {actual_bytes}"
            ),
            Self::MagicMismatch { actual } => write!(
                formatter,
                "occupancy-map magic does not match: got {actual:02x?}"
            ),
            Self::UnsupportedVersion { version } => {
                write!(formatter, "unsupported occupancy-map version {version}")
            }
            Self::InvalidHeaderLength { bytes } => write!(
                formatter,
                "occupancy-map version 1 header must be {OCCUPANCY_MAP_HEADER_BYTES} bytes, got {bytes}"
            ),
            Self::NonZeroReservedByte { offset, value } => write!(
                formatter,
                "occupancy-map reserved byte at offset {offset} must be zero, got {value:#04x}"
            ),
            Self::UnsupportedCoordinateFrame { code } => write!(
                formatter,
                "unsupported occupancy-map coordinate-frame code {code}"
            ),
            Self::UnsupportedRowOrder { code } => {
                write!(formatter, "unsupported occupancy-map row-order code {code}")
            }
            Self::UnsupportedCellEncoding { code } => write!(
                formatter,
                "unsupported occupancy-map cell-encoding code {code}"
            ),
            Self::DeclaredCellCountMismatch { declared, computed } => write!(
                formatter,
                "occupancy-map declares {declared} cells but dimensions contain {computed}"
            ),
            Self::PayloadLengthMismatch { declared, computed } => write!(
                formatter,
                "occupancy-map declares a {declared}-byte payload but its cell encoding requires {computed}"
            ),
            Self::Geometry(source) => write!(formatter, "invalid occupancy-map geometry: {source}"),
            Self::WorldToOccupancy(source) => write!(
                formatter,
                "invalid occupancy-map world-to-occupancy transform: {source}"
            ),
            Self::HeightRange(source) => {
                write!(formatter, "invalid occupancy-map height range: {source}")
            }
            Self::ChecksumMismatch { stored, computed } => write!(
                formatter,
                "occupancy-map checksum mismatch: stored {stored:#010x}, computed {computed:#010x}"
            ),
            Self::UnsupportedCellClass { index, class_id } => write!(
                formatter,
                "occupancy-map cell {index} has unsupported class ID {class_id}"
            ),
            Self::AllocationFailed { cells, .. } => write!(
                formatter,
                "failed to allocate storage for {cells} occupancy-map cells"
            ),
            Self::EncodedLengthOverflow { cells } => write!(
                formatter,
                "occupancy map with {cells} cells has no addressable encoded length"
            ),
        }
    }
}

impl std::error::Error for OccupancyMapDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Geometry(source) => Some(source),
            Self::WorldToOccupancy(source) => Some(source),
            Self::HeightRange(source) => Some(source),
            Self::AllocationFailed { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<OccupancyGridGeometryError> for OccupancyMapDecodeError {
    fn from(source: OccupancyGridGeometryError) -> Self {
        Self::Geometry(source)
    }
}

impl From<WorldToOccupancyError> for OccupancyMapDecodeError {
    fn from(source: WorldToOccupancyError) -> Self {
        Self::WorldToOccupancy(source)
    }
}

impl From<HeightRangeError> for OccupancyMapDecodeError {
    fn from(source: HeightRangeError) -> Self {
        Self::HeightRange(source)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyMapSaveOperation {
    CreateTemporary,
    WriteTemporary,
    SyncTemporary,
    PublishRename,
    OpenParentDirectory,
    SyncParentDirectory,
}

impl fmt::Display for OccupancyMapSaveOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::CreateTemporary => "create temporary file",
            Self::WriteTemporary => "write temporary file",
            Self::SyncTemporary => "sync temporary file",
            Self::PublishRename => "publish temporary file",
            Self::OpenParentDirectory => "open parent directory after publication",
            Self::SyncParentDirectory => "sync parent directory after publication",
        };
        formatter.write_str(name)
    }
}

#[derive(Debug)]
pub struct OccupancyMapSaveIoError {
    operation: OccupancyMapSaveOperation,
    destination: PathBuf,
    temporary_path: Option<PathBuf>,
    published: bool,
    source: io::Error,
    cleanup_error: Option<io::Error>,
}

impl OccupancyMapSaveIoError {
    pub fn operation(&self) -> OccupancyMapSaveOperation {
        self.operation
    }

    pub fn destination(&self) -> &Path {
        self.destination.as_path()
    }

    pub fn temporary_path(&self) -> Option<&Path> {
        self.temporary_path.as_deref()
    }

    /// Whether the destination rename completed before the reported error.
    pub fn published(&self) -> bool {
        self.published
    }

    pub fn io_error(&self) -> &io::Error {
        &self.source
    }

    pub fn cleanup_error(&self) -> Option<&io::Error> {
        self.cleanup_error.as_ref()
    }
}

impl fmt::Display for OccupancyMapSaveIoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "failed to {} for occupancy map '{}': {}",
            self.operation,
            self.destination.display(),
            self.source
        )?;
        if let Some(cleanup_error) = &self.cleanup_error {
            write!(
                formatter,
                "; additionally failed to remove temporary file: {cleanup_error}"
            )?;
        }
        if self.published {
            formatter.write_str(" (destination was already published)")?;
        }
        Ok(())
    }
}

impl std::error::Error for OccupancyMapSaveIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Debug)]
pub enum OccupancyMapSaveError {
    Encode(OccupancyMapEncodeError),
    InvalidDestination {
        path: PathBuf,
    },
    TemporaryNameCollisions {
        destination: PathBuf,
        attempts: usize,
    },
    Io(OccupancyMapSaveIoError),
}

impl fmt::Display for OccupancyMapSaveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(source) => write!(formatter, "cannot encode occupancy map: {source}"),
            Self::InvalidDestination { path } => write!(
                formatter,
                "occupancy-map destination '{}' has no file name",
                path.display()
            ),
            Self::TemporaryNameCollisions {
                destination,
                attempts,
            } => write!(
                formatter,
                "could not reserve a temporary file beside '{}' after {attempts} attempts",
                destination.display()
            ),
            Self::Io(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for OccupancyMapSaveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encode(source) => Some(source),
            Self::Io(source) => Some(source),
            _ => None,
        }
    }
}

impl From<OccupancyMapEncodeError> for OccupancyMapSaveError {
    fn from(source: OccupancyMapEncodeError) -> Self {
        Self::Encode(source)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyMapLoadOperation {
    Open,
    ReadMetadata,
    ReadHeader,
    ReadCells,
    ReadChecksum,
    CheckEndOfFile,
}

impl fmt::Display for OccupancyMapLoadOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Open => "open",
            Self::ReadMetadata => "read metadata for",
            Self::ReadHeader => "read header from",
            Self::ReadCells => "read cells from",
            Self::ReadChecksum => "read checksum from",
            Self::CheckEndOfFile => "check end of",
        };
        formatter.write_str(name)
    }
}

#[derive(Debug)]
pub enum OccupancyMapLoadError {
    Io {
        operation: OccupancyMapLoadOperation,
        path: PathBuf,
        source: io::Error,
    },
    NotRegularFile {
        path: PathBuf,
    },
    FileLengthNotAddressable {
        path: PathBuf,
        bytes: u64,
    },
    Format(OccupancyMapDecodeError),
}

impl fmt::Display for OccupancyMapLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "failed to {operation} occupancy map '{}': {source}",
                path.display()
            ),
            Self::NotRegularFile { path } => write!(
                formatter,
                "occupancy-map path '{}' is not a regular file",
                path.display()
            ),
            Self::FileLengthNotAddressable { path, bytes } => write!(
                formatter,
                "occupancy-map file '{}' has {bytes} bytes, which is not addressable on this host",
                path.display()
            ),
            Self::Format(source) => write!(formatter, "invalid occupancy-map file: {source}"),
        }
    }
}

impl std::error::Error for OccupancyMapLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Format(source) => Some(source),
            _ => None,
        }
    }
}

impl From<OccupancyMapDecodeError> for OccupancyMapLoadError {
    fn from(source: OccupancyMapDecodeError) -> Self {
        Self::Format(source)
    }
}

struct EncodedParts<'a> {
    header: [u8; OCCUPANCY_MAP_HEADER_BYTES],
    cells: &'a [u8],
    checksum: u32,
    encoded_len: usize,
}

struct ParsedHeader {
    geometry: OccupancyGridGeometry,
    world_to_occupancy: WorldToOccupancy,
    height_range: HeightRangeMeters,
    revision: u64,
}

/// Capability proving that persisted and replay occupancy are exactly equal
/// for every serialized domain field and cell.
///
/// The type must be visible to the sibling occupancy module, but its private
/// field means only this verifier can construct it. This prevents another
/// dense sibling from attaching a live identity without performing the exact
/// comparison.
pub(super) struct ExactReplayMatchProof(());

/// A decoded occupancy artifact that deliberately has no process-local map
/// identity.
///
/// This type is the warm-start boundary: it cannot be passed to navigation as
/// a live map until [`Self::verify_replay_and_bind`] proves an exact match to
/// final evidence from a retained sparse-map replay.
#[derive(Debug)]
pub struct PersistedOccupancyMap {
    snapshot: OccupancyGridSnapshot,
}

impl PersistedOccupancyMap {
    fn from_decoded(snapshot: OccupancyGridSnapshot) -> Self {
        debug_assert_eq!(snapshot.map_instance_id(), None);
        Self { snapshot }
    }

    /// Inspect the unbound artifact without granting it a live map identity.
    pub fn snapshot(&self) -> &OccupancyGridSnapshot {
        &self.snapshot
    }

    /// Verify every persisted field and cell against the final occupancy
    /// output of a dataset replay, then attach that replay's process-local map
    /// identity without copying the persisted cell buffer.
    ///
    /// A successful result proves that this occupancy artifact exactly matches
    /// the supplied replay occupancy output. It does **not** prove that a
    /// current live camera frame has relocalized in that map; motion admission
    /// must still require fresh tracker localization evidence.
    pub fn verify_replay_and_bind(
        self,
        replay: ReplayOccupancyEvidence,
    ) -> Result<ReplayMatchedOccupancyMap, OccupancyReplayBindError> {
        let proof = verify_exact_replay(&self.snapshot, &replay.occupancy_snapshot)?;
        let snapshot = self
            .snapshot
            .bind_to_exact_replay(&replay.occupancy_snapshot, proof);
        let sparse_map_snapshot = replay.sparse_map_snapshot;
        Ok(ReplayMatchedOccupancyMap {
            snapshot,
            sparse_map_snapshot,
        })
    }
}

/// Sparse and occupancy outputs captured from one dataset replay.
///
/// Construction parses the weak `Option<MapInstanceId>` carried by a generic
/// occupancy snapshot into a value whose sparse/occupancy identity cannot
/// disagree. That identity check cannot itself prove replay quiescence or
/// synchronization of the two revisions: the integration must capture both
/// outputs behind the replay runtime's successful drain barrier.
#[derive(Debug)]
pub struct ReplayOccupancyEvidence {
    sparse_map_snapshot: MapSnapshot,
    occupancy_snapshot: OccupancyGridSnapshot,
}

impl ReplayOccupancyEvidence {
    pub fn try_new(
        sparse_map_snapshot: MapSnapshot,
        occupancy_snapshot: OccupancyGridSnapshot,
    ) -> Result<Self, ReplayOccupancyEvidenceError> {
        let expected = sparse_map_snapshot.instance_id();
        let actual = occupancy_snapshot
            .map_instance_id()
            .ok_or(ReplayOccupancyEvidenceError::UnboundOccupancySnapshot)?;
        if actual != expected {
            return Err(ReplayOccupancyEvidenceError::MapInstanceMismatch { expected, actual });
        }
        Ok(Self {
            sparse_map_snapshot,
            occupancy_snapshot,
        })
    }

    pub fn sparse_map_snapshot(&self) -> MapSnapshot {
        self.sparse_map_snapshot
    }

    pub fn occupancy_snapshot(&self) -> &OccupancyGridSnapshot {
        &self.occupancy_snapshot
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayOccupancyEvidenceError {
    UnboundOccupancySnapshot,
    MapInstanceMismatch {
        expected: MapInstanceId,
        actual: MapInstanceId,
    },
}

impl fmt::Display for ReplayOccupancyEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnboundOccupancySnapshot => formatter.write_str(
                "dataset replay occupancy has no live map identity; persisted occupancy cannot establish one",
            ),
            Self::MapInstanceMismatch { expected, actual } => write!(
                formatter,
                "dataset replay occupancy belongs to map {}, but the retained sparse replay is map {}",
                actual.as_u64(),
                expected.as_u64()
            ),
        }
    }
}

impl std::error::Error for ReplayOccupancyEvidenceError {}

/// A loaded artifact proven byte-for-domain-field equivalent to final replay
/// output and rebound to that replay's process-local map identity.
///
/// The name intentionally says `ReplayMatched`, not `Relocalized`: live visual
/// relocalization is a separate tracker result.
#[derive(Debug)]
pub struct ReplayMatchedOccupancyMap {
    snapshot: OccupancyGridSnapshot,
    sparse_map_snapshot: MapSnapshot,
}

impl ReplayMatchedOccupancyMap {
    pub fn sparse_map_snapshot(&self) -> MapSnapshot {
        self.sparse_map_snapshot
    }

    pub fn map_instance_id(&self) -> MapInstanceId {
        self.sparse_map_snapshot.instance_id()
    }

    pub fn snapshot(&self) -> &OccupancyGridSnapshot {
        &self.snapshot
    }

    pub fn into_snapshot(self) -> OccupancyGridSnapshot {
        self.snapshot
    }
}

/// Exact persisted/replay field names. Floating-point values are compared by
/// representation so a successful match means the artifact can be substituted
/// without silently changing a transform or metric boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyReplayField {
    GridWidthCells,
    GridHeightCells,
    ResolutionMetersPerCell,
    LowerBoundMeters { axis: usize },
    WorldToOccupancyRotation { row: usize, column: usize },
    WorldToOccupancyTranslationMeters { axis: usize },
    MinimumHeightMeters,
    MaximumHeightMeters,
    Revision,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyReplayBindError {
    FieldMismatch {
        field: OccupancyReplayField,
        persisted_bits: u64,
        replayed_bits: u64,
    },
    CellClassMismatch {
        index: usize,
        persisted: u8,
        replayed: u8,
    },
}

impl fmt::Display for OccupancyReplayBindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldMismatch {
                field,
                persisted_bits,
                replayed_bits,
            } => write!(
                formatter,
                "persisted occupancy field {field:?} does not exactly match dataset replay (persisted={persisted_bits:#018x}, replayed={replayed_bits:#018x})"
            ),
            Self::CellClassMismatch {
                index,
                persisted,
                replayed,
            } => write!(
                formatter,
                "persisted occupancy cell {index} has class {persisted} but dataset replay has class {replayed}"
            ),
        }
    }
}

impl std::error::Error for OccupancyReplayBindError {}

/// Encodes one snapshot without platform-dependent padding or byte order.
pub fn encode_occupancy_map(
    snapshot: &OccupancyGridSnapshot,
) -> Result<Vec<u8>, OccupancyMapEncodeError> {
    let parts = encoded_parts(snapshot)?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(parts.encoded_len)
        .map_err(|source| OccupancyMapEncodeError::AllocationFailed {
            requested_bytes: parts.encoded_len,
            source,
        })?;
    encoded.extend_from_slice(&parts.header);
    encoded.extend_from_slice(parts.cells);
    encoded.extend_from_slice(&parts.checksum.to_le_bytes());
    Ok(encoded)
}

/// Parses an untrusted in-memory map once into validated occupancy domain
/// types. `limits` is checked before allocating the returned cell vector.
pub fn decode_occupancy_map(
    bytes: &[u8],
    limits: OccupancyMapLimits,
) -> Result<OccupancyGridSnapshot, OccupancyMapDecodeError> {
    validate_boundary_length(bytes.len(), limits)?;
    let header = &bytes[..OCCUPANCY_MAP_HEADER_BYTES];
    let parsed = parse_header(header, bytes.len(), limits)?;
    let checksum_offset = bytes.len() - OCCUPANCY_MAP_CHECKSUM_BYTES;
    let cells = &bytes[OCCUPANCY_MAP_HEADER_BYTES..checksum_offset];
    let stored_checksum = read_u32(bytes, checksum_offset);
    validate_payload(header, cells, stored_checksum)?;

    let mut owned_cells = Vec::new();
    owned_cells
        .try_reserve_exact(cells.len())
        .map_err(|source| OccupancyMapDecodeError::AllocationFailed {
            cells: cells.len(),
            source,
        })?;
    owned_cells.extend_from_slice(cells);
    Ok(snapshot_from_parsed(parsed, owned_cells))
}

/// Decode an untrusted in-memory artifact directly into the explicit unbound
/// warm-start type. This performs the same single parse as
/// [`decode_occupancy_map`]; no second validation pass or cell copy is added.
pub fn decode_persisted_occupancy_map(
    bytes: &[u8],
    limits: OccupancyMapLimits,
) -> Result<PersistedOccupancyMap, OccupancyMapDecodeError> {
    decode_occupancy_map(bytes, limits).map(PersistedOccupancyMap::from_decoded)
}

/// Writes a map through a same-directory temporary file, synchronizes its
/// contents, atomically renames it over `path`, then synchronizes the parent
/// directory. If an error reports `published() == true`, the rename succeeded
/// but the final directory-durability step did not.
pub fn save_occupancy_map_atomic(
    path: impl AsRef<Path>,
    snapshot: &OccupancyGridSnapshot,
) -> Result<(), OccupancyMapSaveError> {
    let destination = path.as_ref();
    let file_name = destination
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| OccupancyMapSaveError::InvalidDestination {
            path: destination.to_path_buf(),
        })?;
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parts = encoded_parts(snapshot)?;
    let (temporary_path, mut temporary_file) =
        create_temporary_file(parent, file_name, destination)?;

    if let Err(source) = temporary_file
        .write_all(&parts.header)
        .and_then(|()| temporary_file.write_all(parts.cells))
        .and_then(|()| temporary_file.write_all(&parts.checksum.to_le_bytes()))
    {
        drop(temporary_file);
        return Err(save_io_with_cleanup(
            OccupancyMapSaveOperation::WriteTemporary,
            destination,
            &temporary_path,
            source,
        ));
    }
    if let Err(source) = temporary_file.sync_all() {
        drop(temporary_file);
        return Err(save_io_with_cleanup(
            OccupancyMapSaveOperation::SyncTemporary,
            destination,
            &temporary_path,
            source,
        ));
    }
    drop(temporary_file);

    if let Err(source) = std::fs::rename(&temporary_path, destination) {
        return Err(save_io_with_cleanup(
            OccupancyMapSaveOperation::PublishRename,
            destination,
            &temporary_path,
            source,
        ));
    }

    let parent_directory = File::open(parent).map_err(|source| {
        OccupancyMapSaveError::Io(OccupancyMapSaveIoError {
            operation: OccupancyMapSaveOperation::OpenParentDirectory,
            destination: destination.to_path_buf(),
            temporary_path: None,
            published: true,
            source,
            cleanup_error: None,
        })
    })?;
    parent_directory.sync_all().map_err(|source| {
        OccupancyMapSaveError::Io(OccupancyMapSaveIoError {
            operation: OccupancyMapSaveOperation::SyncParentDirectory,
            destination: destination.to_path_buf(),
            temporary_path: None,
            published: true,
            source,
            cleanup_error: None,
        })
    })
}

/// Loads a regular file with a metadata size check before cell allocation.
pub fn load_occupancy_map(
    path: impl AsRef<Path>,
    limits: OccupancyMapLimits,
) -> Result<OccupancyGridSnapshot, OccupancyMapLoadError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|source| OccupancyMapLoadError::Io {
        operation: OccupancyMapLoadOperation::Open,
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = file
        .metadata()
        .map_err(|source| OccupancyMapLoadError::Io {
            operation: OccupancyMapLoadOperation::ReadMetadata,
            path: path.to_path_buf(),
            source,
        })?;
    if !metadata.is_file() {
        return Err(OccupancyMapLoadError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    let encoded_len = usize::try_from(metadata.len()).map_err(|_| {
        OccupancyMapLoadError::FileLengthNotAddressable {
            path: path.to_path_buf(),
            bytes: metadata.len(),
        }
    })?;
    validate_boundary_length(encoded_len, limits)?;

    let mut header = [0_u8; OCCUPANCY_MAP_HEADER_BYTES];
    file.read_exact(&mut header)
        .map_err(|source| OccupancyMapLoadError::Io {
            operation: OccupancyMapLoadOperation::ReadHeader,
            path: path.to_path_buf(),
            source,
        })?;
    let parsed = parse_header(&header, encoded_len, limits)?;
    let cells_len = parsed.geometry.cell_count();
    let mut cells = Vec::new();
    cells.try_reserve_exact(cells_len).map_err(|source| {
        OccupancyMapDecodeError::AllocationFailed {
            cells: cells_len,
            source,
        }
    })?;
    let cell_read_limit = u64::try_from(cells_len)
        .map_err(|_| OccupancyMapDecodeError::EncodedLengthOverflow { cells: cells_len })?;
    let cells_read = (&mut file)
        .take(cell_read_limit)
        .read_to_end(&mut cells)
        .map_err(|source| OccupancyMapLoadError::Io {
            operation: OccupancyMapLoadOperation::ReadCells,
            path: path.to_path_buf(),
            source,
        })?;
    if cells_read != cells_len {
        return Err(OccupancyMapDecodeError::Truncated {
            expected_bytes: encoded_len,
            actual_bytes: OCCUPANCY_MAP_HEADER_BYTES.saturating_add(cells_read),
        }
        .into());
    }
    let mut checksum_bytes = [0_u8; OCCUPANCY_MAP_CHECKSUM_BYTES];
    file.read_exact(&mut checksum_bytes)
        .map_err(|source| OccupancyMapLoadError::Io {
            operation: OccupancyMapLoadOperation::ReadChecksum,
            path: path.to_path_buf(),
            source,
        })?;
    let stored_checksum = u32::from_le_bytes(checksum_bytes);
    let mut trailing = [0_u8; 1];
    let trailing_len = file
        .read(&mut trailing)
        .map_err(|source| OccupancyMapLoadError::Io {
            operation: OccupancyMapLoadOperation::CheckEndOfFile,
            path: path.to_path_buf(),
            source,
        })?;
    if trailing_len != 0 {
        return Err(OccupancyMapDecodeError::TrailingBytes {
            expected_bytes: encoded_len,
            actual_bytes: encoded_len.saturating_add(trailing_len),
        }
        .into());
    }
    validate_payload(&header, &cells, stored_checksum)?;
    Ok(snapshot_from_parsed(parsed, cells))
}

/// Load an untrusted file directly into the explicit unbound warm-start type.
/// The file is opened, bounded, parsed, and checksummed exactly once.
pub fn load_persisted_occupancy_map(
    path: impl AsRef<Path>,
    limits: OccupancyMapLimits,
) -> Result<PersistedOccupancyMap, OccupancyMapLoadError> {
    load_occupancy_map(path, limits).map(PersistedOccupancyMap::from_decoded)
}

fn verify_exact_replay(
    persisted: &OccupancyGridSnapshot,
    replayed: &OccupancyGridSnapshot,
) -> Result<ExactReplayMatchProof, OccupancyReplayBindError> {
    fn require_exact(
        field: OccupancyReplayField,
        persisted_bits: u64,
        replayed_bits: u64,
    ) -> Result<(), OccupancyReplayBindError> {
        if persisted_bits == replayed_bits {
            Ok(())
        } else {
            Err(OccupancyReplayBindError::FieldMismatch {
                field,
                persisted_bits,
                replayed_bits,
            })
        }
    }

    require_exact(
        OccupancyReplayField::GridWidthCells,
        u64::from(persisted.width()),
        u64::from(replayed.width()),
    )?;
    require_exact(
        OccupancyReplayField::GridHeightCells,
        u64::from(persisted.height()),
        u64::from(replayed.height()),
    )?;
    require_exact(
        OccupancyReplayField::ResolutionMetersPerCell,
        persisted.resolution_m().to_bits(),
        replayed.resolution_m().to_bits(),
    )?;
    for axis in 0..2 {
        require_exact(
            OccupancyReplayField::LowerBoundMeters { axis },
            persisted.lower_bound_m()[axis].to_bits(),
            replayed.lower_bound_m()[axis].to_bits(),
        )?;
    }

    let persisted_transform = persisted.world_to_occupancy();
    let replayed_transform = replayed.world_to_occupancy();
    let persisted_rotation = persisted_transform.rotation();
    let replayed_rotation = replayed_transform.rotation();
    for row in 0..3 {
        for column in 0..3 {
            require_exact(
                OccupancyReplayField::WorldToOccupancyRotation { row, column },
                persisted_rotation[row][column].to_bits(),
                replayed_rotation[row][column].to_bits(),
            )?;
        }
    }
    let persisted_translation = persisted_transform.translation_m();
    let replayed_translation = replayed_transform.translation_m();
    for axis in 0..3 {
        require_exact(
            OccupancyReplayField::WorldToOccupancyTranslationMeters { axis },
            persisted_translation[axis].to_bits(),
            replayed_translation[axis].to_bits(),
        )?;
    }

    require_exact(
        OccupancyReplayField::MinimumHeightMeters,
        persisted.height_range().minimum_m().to_bits(),
        replayed.height_range().minimum_m().to_bits(),
    )?;
    require_exact(
        OccupancyReplayField::MaximumHeightMeters,
        persisted.height_range().maximum_m().to_bits(),
        replayed.height_range().maximum_m().to_bits(),
    )?;
    require_exact(
        OccupancyReplayField::Revision,
        persisted.revision(),
        replayed.revision(),
    )?;

    let persisted_cells = persisted.class_ids();
    let replayed_cells = replayed.class_ids();
    // Valid snapshots contain exactly width * height cells, and those exact
    // dimensions were compared above. A separate runtime length error would
    // therefore describe an unrepresentable state.
    debug_assert_eq!(persisted_cells.len(), replayed_cells.len());
    if let Some((index, (&persisted, &replayed))) = persisted_cells
        .iter()
        .zip(replayed_cells)
        .enumerate()
        .find(|(_, (persisted, replayed))| persisted != replayed)
    {
        return Err(OccupancyReplayBindError::CellClassMismatch {
            index,
            persisted,
            replayed,
        });
    }
    Ok(ExactReplayMatchProof(()))
}

fn encoded_parts(
    snapshot: &OccupancyGridSnapshot,
) -> Result<EncodedParts<'_>, OccupancyMapEncodeError> {
    let geometry = snapshot.geometry();
    let cells = snapshot.class_ids();
    let encoded_len = OCCUPANCY_MAP_FIXED_OVERHEAD_BYTES
        .checked_add(cells.len())
        .ok_or(OccupancyMapEncodeError::EncodedLengthOverflow { cells: cells.len() })?;

    let mut header = [0_u8; OCCUPANCY_MAP_HEADER_BYTES];
    header[..MAGIC.len()].copy_from_slice(&MAGIC);
    put_u16(&mut header, 8, OCCUPANCY_MAP_FORMAT_VERSION);
    put_u16(
        &mut header,
        10,
        u16::try_from(OCCUPANCY_MAP_HEADER_BYTES)
            .expect("version 1 occupancy header length fits u16"),
    );
    header[16] = COORDINATE_FRAME_METRIC_XY_HEIGHT;
    header[17] = ROW_ORDER_INCREASING_Y;
    header[18] = CELL_ENCODING_CLASS_U8;
    put_u32(&mut header, 20, geometry.width());
    put_u32(&mut header, 24, geometry.height());
    let cell_count = u64::from(geometry.width()) * u64::from(geometry.height());
    put_u64(&mut header, 32, cell_count);
    put_u64(&mut header, 40, cell_count);
    put_f64(&mut header, 48, geometry.resolution_m());
    let lower_bound_m = geometry.lower_bound_m();
    put_f64(&mut header, 56, lower_bound_m[0]);
    put_f64(&mut header, 64, lower_bound_m[1]);
    let rotation = snapshot.world_to_occupancy().rotation();
    for (index, value) in rotation.into_iter().flatten().enumerate() {
        put_f64(&mut header, 72 + index * 8, value);
    }
    let translation_m = snapshot.world_to_occupancy().translation_m();
    for (axis, value) in translation_m.into_iter().enumerate() {
        put_f64(&mut header, 144 + axis * 8, value);
    }
    let height_range = snapshot.height_range();
    put_f64(&mut header, 168, height_range.minimum_m());
    put_f64(&mut header, 176, height_range.maximum_m());
    put_u64(&mut header, 184, snapshot.revision());

    let checksum = checksum_for(&header, cells);
    Ok(EncodedParts {
        header,
        cells,
        checksum,
        encoded_len,
    })
}

fn validate_boundary_length(
    actual_bytes: usize,
    limits: OccupancyMapLimits,
) -> Result<(), OccupancyMapDecodeError> {
    if actual_bytes > limits.maximum_encoded_bytes() {
        return Err(OccupancyMapDecodeError::InputExceedsLimit {
            actual_bytes,
            maximum_bytes: limits.maximum_encoded_bytes(),
        });
    }
    if actual_bytes < OCCUPANCY_MAP_FIXED_OVERHEAD_BYTES {
        return Err(OccupancyMapDecodeError::Truncated {
            expected_bytes: OCCUPANCY_MAP_FIXED_OVERHEAD_BYTES,
            actual_bytes,
        });
    }
    Ok(())
}

fn parse_header(
    header: &[u8],
    actual_bytes: usize,
    limits: OccupancyMapLimits,
) -> Result<ParsedHeader, OccupancyMapDecodeError> {
    let mut actual_magic = [0_u8; MAGIC.len()];
    actual_magic.copy_from_slice(&header[..MAGIC.len()]);
    if actual_magic != MAGIC {
        return Err(OccupancyMapDecodeError::MagicMismatch {
            actual: actual_magic,
        });
    }
    let version = read_u16(header, 8);
    if version != OCCUPANCY_MAP_FORMAT_VERSION {
        return Err(OccupancyMapDecodeError::UnsupportedVersion { version });
    }
    let header_len = read_u16(header, 10);
    if usize::from(header_len) != OCCUPANCY_MAP_HEADER_BYTES {
        return Err(OccupancyMapDecodeError::InvalidHeaderLength { bytes: header_len });
    }
    validate_reserved_zero(header, 12..16)?;
    if header[16] != COORDINATE_FRAME_METRIC_XY_HEIGHT {
        return Err(OccupancyMapDecodeError::UnsupportedCoordinateFrame { code: header[16] });
    }
    if header[17] != ROW_ORDER_INCREASING_Y {
        return Err(OccupancyMapDecodeError::UnsupportedRowOrder { code: header[17] });
    }
    if header[18] != CELL_ENCODING_CLASS_U8 {
        return Err(OccupancyMapDecodeError::UnsupportedCellEncoding { code: header[18] });
    }
    validate_reserved_zero(header, 19..20)?;
    validate_reserved_zero(header, 28..32)?;

    let width = read_u32(header, 20);
    let height = read_u32(header, 24);
    let resolution_m = read_f64(header, 48);
    let lower_bound_m = [read_f64(header, 56), read_f64(header, 64)];
    let geometry = OccupancyGridGeometry::try_new(
        resolution_m,
        lower_bound_m,
        width,
        height,
        limits.maximum_cells(),
    )?;
    let computed_cell_count = u64::from(width) * u64::from(height);
    let declared_cell_count = read_u64(header, 32);
    if declared_cell_count != computed_cell_count {
        return Err(OccupancyMapDecodeError::DeclaredCellCountMismatch {
            declared: declared_cell_count,
            computed: computed_cell_count,
        });
    }
    let payload_length = read_u64(header, 40);
    if payload_length != computed_cell_count {
        return Err(OccupancyMapDecodeError::PayloadLengthMismatch {
            declared: payload_length,
            computed: computed_cell_count,
        });
    }
    let expected_bytes = OCCUPANCY_MAP_FIXED_OVERHEAD_BYTES
        .checked_add(geometry.cell_count())
        .ok_or(OccupancyMapDecodeError::EncodedLengthOverflow {
            cells: geometry.cell_count(),
        })?;
    if actual_bytes < expected_bytes {
        return Err(OccupancyMapDecodeError::Truncated {
            expected_bytes,
            actual_bytes,
        });
    }
    if actual_bytes > expected_bytes {
        return Err(OccupancyMapDecodeError::TrailingBytes {
            expected_bytes,
            actual_bytes,
        });
    }

    let mut rotation = [[0.0_f64; 3]; 3];
    for (index, value) in rotation.iter_mut().flatten().enumerate() {
        *value = read_f64(header, 72 + index * 8);
    }
    let translation_m = [
        read_f64(header, 144),
        read_f64(header, 152),
        read_f64(header, 160),
    ];
    let world_to_occupancy = WorldToOccupancy::try_new(rotation, translation_m)?;
    let height_range = HeightRangeMeters::try_new(read_f64(header, 168), read_f64(header, 176))?;
    Ok(ParsedHeader {
        geometry,
        world_to_occupancy,
        height_range,
        revision: read_u64(header, 184),
    })
}

fn snapshot_from_parsed(parsed: ParsedHeader, cells: Vec<u8>) -> OccupancyGridSnapshot {
    OccupancyGridSnapshot::from_validated_persistent_parts(
        parsed.geometry,
        parsed.world_to_occupancy,
        parsed.height_range,
        parsed.revision,
        cells,
    )
}

fn validate_reserved_zero(
    header: &[u8],
    offsets: std::ops::Range<usize>,
) -> Result<(), OccupancyMapDecodeError> {
    if let Some(offset) = offsets.clone().find(|&offset| header[offset] != 0) {
        return Err(OccupancyMapDecodeError::NonZeroReservedByte {
            offset,
            value: header[offset],
        });
    }
    Ok(())
}

fn validate_payload(
    header: &[u8],
    cells: &[u8],
    stored: u32,
) -> Result<(), OccupancyMapDecodeError> {
    let mut checksum = Crc32::new();
    checksum.update(header);
    let mut invalid_cell = None;
    for (index, &class_id) in cells.iter().enumerate() {
        checksum.update_byte(class_id);
        if invalid_cell.is_none() && class_id > 2 {
            invalid_cell = Some((index, class_id));
        }
    }
    let computed = checksum.finish();
    if computed != stored {
        return Err(OccupancyMapDecodeError::ChecksumMismatch { stored, computed });
    }
    if let Some((index, class_id)) = invalid_cell {
        return Err(OccupancyMapDecodeError::UnsupportedCellClass { index, class_id });
    }
    Ok(())
}

fn create_temporary_file(
    parent: &Path,
    file_name: &std::ffi::OsStr,
    destination: &Path,
) -> Result<(PathBuf, File), OccupancyMapSaveError> {
    for _ in 0..TEMP_CREATE_ATTEMPTS {
        let nonce = NEXT_TEMPORARY_FILE_NONCE.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = OsString::from(".");
        temporary_name.push(file_name);
        temporary_name.push(format!(".kiko-map.{}.{}.tmp", std::process::id(), nonce));
        let temporary_path = parent.join(temporary_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((temporary_path, file)),
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(OccupancyMapSaveError::Io(OccupancyMapSaveIoError {
                    operation: OccupancyMapSaveOperation::CreateTemporary,
                    destination: destination.to_path_buf(),
                    temporary_path: Some(temporary_path),
                    published: false,
                    source,
                    cleanup_error: None,
                }));
            }
        }
    }
    Err(OccupancyMapSaveError::TemporaryNameCollisions {
        destination: destination.to_path_buf(),
        attempts: TEMP_CREATE_ATTEMPTS,
    })
}

fn save_io_with_cleanup(
    operation: OccupancyMapSaveOperation,
    destination: &Path,
    temporary_path: &Path,
    source: io::Error,
) -> OccupancyMapSaveError {
    let cleanup_error = match std::fs::remove_file(temporary_path) {
        Ok(()) => None,
        Err(cleanup_error) if cleanup_error.kind() == io::ErrorKind::NotFound => None,
        Err(cleanup_error) => Some(cleanup_error),
    };
    OccupancyMapSaveError::Io(OccupancyMapSaveIoError {
        operation,
        destination: destination.to_path_buf(),
        temporary_path: Some(temporary_path.to_path_buf()),
        published: false,
        source,
        cleanup_error,
    })
}

struct Crc32(u32);

impl Crc32 {
    const fn new() -> Self {
        Self(u32::MAX)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.update_byte(byte);
        }
    }

    fn update_byte(&mut self, byte: u8) {
        let table_index = usize::from((self.0 ^ u32::from(byte)).to_le_bytes()[0]);
        self.0 = (self.0 >> 8) ^ CRC32_TABLE[table_index];
    }

    const fn finish(self) -> u32 {
        !self.0
    }
}

fn checksum_for(header: &[u8], cells: &[u8]) -> u32 {
    let mut checksum = Crc32::new();
    checksum.update(header);
    checksum.update(cells);
    checksum.finish()
}

const CRC32_TABLE: [u32; 256] = build_crc32_table();

const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0_u32; 256];
    let mut index = 0;
    let mut initial_value = 0_u32;
    while index < table.len() {
        let mut value = initial_value;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 == 0 {
                value >> 1
            } else {
                (value >> 1) ^ 0xedb8_8320
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
        initial_value += 1;
    }
    table
}

fn put_u16(output: &mut [u8], offset: usize, value: u16) {
    output[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(output: &mut [u8], offset: usize, value: u64) {
    output[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_f64(output: &mut [u8], offset: usize, value: f64) {
    put_u64(output, offset, value.to_bits());
}

fn read_u16(input: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([input[offset], input[offset + 1]])
}

fn read_u32(input: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        input[offset],
        input[offset + 1],
        input[offset + 2],
        input[offset + 3],
    ])
}

fn read_u64(input: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        input[offset],
        input[offset + 1],
        input[offset + 2],
        input[offset + 3],
        input[offset + 4],
        input[offset + 5],
        input[offset + 6],
        input[offset + 7],
    ])
}

fn read_f64(input: &[u8], offset: usize) -> f64 {
    f64::from_bits(read_u64(input, offset))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::dense::occupancy::{OccupancyCell, OccupancyRowOrder};
    use crate::map::SlamMap;

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn create(label: &str) -> Self {
            let nonce = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kiko-occupancy-persistence-{label}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create isolated occupancy persistence test directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            self.0.as_path()
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.0) {
                assert_eq!(
                    error.kind(),
                    io::ErrorKind::NotFound,
                    "clean occupancy persistence test directory '{}': {error}",
                    self.0.display()
                );
            }
        }
    }

    fn limits(maximum_cells: usize) -> OccupancyMapLimits {
        OccupancyMapLimits::try_new(maximum_cells).expect("positive bounded test map")
    }

    fn fixture(width: u32, height: u32, seed: u64) -> OccupancyGridSnapshot {
        let cell_count = usize::try_from(u64::from(width) * u64::from(height))
            .expect("small test dimensions fit usize");
        let resolutions = [0.01, 0.025, 0.05, 0.125, 1.0];
        let resolution_m = resolutions[(seed as usize) % resolutions.len()];
        let lower_bound_m = [
            -2.0 - f64::from((seed % 7) as u32) * 0.25,
            f64::from((seed % 11) as u32) * 0.125,
        ];
        let geometry =
            OccupancyGridGeometry::try_new(resolution_m, lower_bound_m, width, height, cell_count)
                .expect("valid test geometry");
        let rotations = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        ];
        let rotation = rotations[(seed as usize) % rotations.len()];
        let translation_m = [
            f64::from((seed % 5) as u32) * 0.1,
            -f64::from((seed % 13) as u32) * 0.05,
            0.4 + f64::from((seed % 3) as u32) * 0.2,
        ];
        let world_to_occupancy = WorldToOccupancy::try_new(rotation, translation_m)
            .expect("valid test occupancy transform");
        let height_range = HeightRangeMeters::try_new(-0.2, 1.8 + seed as f64 * 0.001)
            .expect("valid test height range");
        let mut state = seed ^ 0xa076_1d64_78bd_642f;
        let class_ids = (0..cell_count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 32) % 3) as u8
            })
            .collect();
        OccupancyGridSnapshot::from_validated_persistent_parts(
            geometry,
            world_to_occupancy,
            height_range,
            seed.rotate_left(17),
            class_ids,
        )
    }

    fn assert_exact_snapshot(expected: &OccupancyGridSnapshot, actual: &OccupancyGridSnapshot) {
        assert_eq!(actual.width(), expected.width());
        assert_eq!(actual.height(), expected.height());
        assert_eq!(
            actual.resolution_m().to_bits(),
            expected.resolution_m().to_bits()
        );
        for axis in 0..2 {
            assert_eq!(
                actual.lower_bound_m()[axis].to_bits(),
                expected.lower_bound_m()[axis].to_bits()
            );
        }
        for row in 0..3 {
            for column in 0..3 {
                assert_eq!(
                    actual.world_to_occupancy().rotation()[row][column].to_bits(),
                    expected.world_to_occupancy().rotation()[row][column].to_bits()
                );
            }
            assert_eq!(
                actual.world_to_occupancy().translation_m()[row].to_bits(),
                expected.world_to_occupancy().translation_m()[row].to_bits()
            );
        }
        assert_eq!(
            actual.height_range().minimum_m().to_bits(),
            expected.height_range().minimum_m().to_bits()
        );
        assert_eq!(
            actual.height_range().maximum_m().to_bits(),
            expected.height_range().maximum_m().to_bits()
        );
        assert_eq!(actual.row_order(), OccupancyRowOrder::IncreasingOccupancyY);
        assert_eq!(actual.revision(), expected.revision());
        assert_eq!(actual.class_ids(), expected.class_ids());
    }

    fn rewrite_checksum(encoded: &mut [u8]) {
        let checksum_offset = encoded.len() - OCCUPANCY_MAP_CHECKSUM_BYTES;
        let checksum = checksum_for(
            &encoded[..OCCUPANCY_MAP_HEADER_BYTES],
            &encoded[OCCUPANCY_MAP_HEADER_BYTES..checksum_offset],
        );
        put_u32(encoded, checksum_offset, checksum);
    }

    fn assert_no_temporary_files(directory: &Path) {
        for entry in fs::read_dir(directory).expect("list test directory") {
            let name = entry.expect("read directory entry").file_name();
            assert!(
                !name.to_string_lossy().contains(".kiko-map."),
                "temporary occupancy map was not cleaned up: {name:?}"
            );
        }
    }

    fn replay_evidence(snapshot: &OccupancyGridSnapshot) -> ReplayOccupancyEvidence {
        let sparse_map_snapshot = SlamMap::new().snapshot();
        let occupancy_snapshot = snapshot
            .try_duplicate()
            .expect("duplicate small replay fixture")
            .with_test_map_instance_id(sparse_map_snapshot.instance_id());
        ReplayOccupancyEvidence::try_new(sparse_map_snapshot, occupancy_snapshot)
            .expect("matching replay evidence")
    }

    fn assert_replay_field_mismatch(
        persisted_encoding: &[u8],
        maximum_cells: usize,
        mutate_replay: impl FnOnce(&mut [u8]),
        expected_field: OccupancyReplayField,
    ) {
        let mut replay_encoding = persisted_encoding.to_vec();
        mutate_replay(&mut replay_encoding);
        rewrite_checksum(&mut replay_encoding);
        let replayed = decode_occupancy_map(&replay_encoding, limits(maximum_cells))
            .expect("mutated replay fixture remains a valid occupancy snapshot");
        let persisted = decode_persisted_occupancy_map(persisted_encoding, limits(maximum_cells))
            .expect("decode typed persisted fixture");
        match persisted
            .verify_replay_and_bind(replay_evidence(&replayed))
            .expect_err("changed replay field must reject persisted occupancy")
        {
            OccupancyReplayBindError::FieldMismatch {
                field,
                persisted_bits,
                replayed_bits,
            } => {
                assert_eq!(field, expected_field);
                assert_ne!(persisted_bits, replayed_bits);
            }
            other => panic!("unexpected replay bind error: {other}"),
        }
    }

    #[test]
    fn crc32_matches_standard_check_vector_across_updates() {
        let mut checksum = Crc32::new();
        checksum.update(b"1234");
        checksum.update(b"56789");
        assert_eq!(checksum.finish(), 0xcbf4_3926);
    }

    #[test]
    fn limits_reject_zero_and_address_overflow() {
        assert!(matches!(
            OccupancyMapLimits::try_new(0),
            Err(OccupancyMapLimitsError::ZeroMaximumCells)
        ));
        assert!(matches!(
            OccupancyMapLimits::try_new(usize::MAX),
            Err(OccupancyMapLimitsError::EncodedLengthOverflow { .. })
        ));
    }

    #[test]
    fn property_style_roundtrips_preserve_every_persisted_bit() {
        for seed in 0_u64..128 {
            let width = u32::try_from(seed % 17 + 1).expect("bounded width");
            let height = u32::try_from(seed.wrapping_mul(7) % 13 + 1).expect("bounded height");
            let original = fixture(width, height, seed);
            let encoded = encode_occupancy_map(&original).expect("encode generated snapshot");
            let decoded = decode_occupancy_map(&encoded, limits(original.geometry().cell_count()))
                .expect("decode generated snapshot");
            assert_exact_snapshot(&original, &decoded);
            assert_eq!(decoded.map_instance_id(), None);
            assert_eq!(
                encode_occupancy_map(&decoded).expect("re-encode decoded snapshot"),
                encoded
            );
        }
    }

    #[test]
    fn encoding_is_deterministic_and_header_fields_are_little_endian() {
        let snapshot = fixture(3, 2, 0x1234_5678);
        let first = encode_occupancy_map(&snapshot).expect("encode fixture");
        let second = encode_occupancy_map(&snapshot).expect("encode same fixture again");
        assert_eq!(first, second);
        assert_eq!(&first[..8], &MAGIC);
        assert_eq!(read_u16(&first, 8), OCCUPANCY_MAP_FORMAT_VERSION);
        assert_eq!(
            usize::from(read_u16(&first, 10)),
            OCCUPANCY_MAP_HEADER_BYTES
        );
        assert_eq!(first[16], COORDINATE_FRAME_METRIC_XY_HEIGHT);
        assert_eq!(first[17], ROW_ORDER_INCREASING_Y);
        assert_eq!(first[18], CELL_ENCODING_CLASS_U8);
        assert_eq!(read_u32(&first, 20), 3);
        assert_eq!(read_u32(&first, 24), 2);
        assert_eq!(read_u64(&first, 32), 6);
        assert_eq!(read_u64(&first, 40), 6);
        assert_eq!(first.len(), OCCUPANCY_MAP_FIXED_OVERHEAD_BYTES + 6);
    }

    #[test]
    fn loading_drops_process_local_map_instance_identity() {
        let geometry =
            OccupancyGridGeometry::try_new(0.1, [-1.0, -1.0], 2, 2, 4).expect("valid geometry");
        let map_instance_id = SlamMap::new().snapshot().instance_id();
        let original = OccupancyGridSnapshot::from_test_cells(
            geometry,
            &[
                OccupancyCell::Unknown,
                OccupancyCell::Free,
                OccupancyCell::Occupied,
                OccupancyCell::Unknown,
            ],
            map_instance_id,
            9,
        );
        assert_eq!(original.map_instance_id(), Some(map_instance_id));
        let encoded = encode_occupancy_map(&original).expect("encode live snapshot");
        let decoded = decode_occupancy_map(&encoded, limits(4)).expect("decode persisted map");
        assert_eq!(decoded.map_instance_id(), None);
        assert_exact_snapshot(&original, &decoded);
    }

    #[test]
    fn replay_evidence_requires_a_bound_snapshot_from_the_retained_sparse_map() {
        let sparse_map_snapshot = SlamMap::new().snapshot();
        assert!(matches!(
            ReplayOccupancyEvidence::try_new(sparse_map_snapshot, fixture(2, 2, 1)),
            Err(ReplayOccupancyEvidenceError::UnboundOccupancySnapshot)
        ));

        let other_map_snapshot = SlamMap::new().snapshot();
        let wrong_map_occupancy =
            fixture(2, 2, 1).with_test_map_instance_id(other_map_snapshot.instance_id());
        assert!(matches!(
            ReplayOccupancyEvidence::try_new(sparse_map_snapshot, wrong_map_occupancy),
            Err(ReplayOccupancyEvidenceError::MapInstanceMismatch {
                expected,
                actual,
            }) if expected == sparse_map_snapshot.instance_id()
                && actual == other_map_snapshot.instance_id()
        ));
    }

    #[test]
    fn exact_replay_match_rebinds_the_loaded_buffer_without_copying_it() {
        let original = fixture(8, 6, 0x9a);
        let encoded = encode_occupancy_map(&original).expect("encode persisted fixture");
        let persisted = decode_persisted_occupancy_map(&encoded, limits(48))
            .expect("decode typed persisted fixture");
        assert_eq!(persisted.snapshot().map_instance_id(), None);
        assert_eq!(persisted.snapshot().geometry().max_cells(), 48);
        let persisted_cells = persisted.snapshot().class_ids().as_ptr();

        let sparse_map_snapshot = SlamMap::new().snapshot();
        let replayed = decode_occupancy_map(&encoded, limits(96))
            .expect("decode replay fixture with its runtime bound")
            .with_test_map_instance_id(sparse_map_snapshot.instance_id());
        assert_eq!(replayed.geometry().max_cells(), 96);
        let replay = ReplayOccupancyEvidence::try_new(sparse_map_snapshot, replayed)
            .expect("matching replay evidence");
        let expected_sparse_snapshot = replay.sparse_map_snapshot();
        let matched = persisted
            .verify_replay_and_bind(replay)
            .expect("exact final replay must bind");

        assert_eq!(matched.sparse_map_snapshot(), expected_sparse_snapshot);
        assert_eq!(
            matched.map_instance_id(),
            expected_sparse_snapshot.instance_id()
        );
        assert_eq!(
            matched.snapshot().map_instance_id(),
            Some(expected_sparse_snapshot.instance_id())
        );
        assert_eq!(matched.snapshot().geometry().max_cells(), 96);
        assert_eq!(matched.snapshot().class_ids().as_ptr(), persisted_cells);
        assert_exact_snapshot(&original, matched.snapshot());
    }

    #[test]
    fn replay_binding_rejects_dimension_and_cell_differences() {
        let original = fixture(4, 3, 0x42);
        let encoded = encode_occupancy_map(&original).expect("encode persisted fixture");

        let different_geometry = fixture(3, 4, 0x42);
        let persisted = decode_persisted_occupancy_map(&encoded, limits(12))
            .expect("decode typed persisted fixture");
        assert!(matches!(
            persisted.verify_replay_and_bind(replay_evidence(&different_geometry)),
            Err(OccupancyReplayBindError::FieldMismatch {
                field: OccupancyReplayField::GridWidthCells,
                ..
            })
        ));

        let different_height = fixture(4, 4, 0x42);
        let persisted = decode_persisted_occupancy_map(&encoded, limits(12))
            .expect("decode typed persisted fixture");
        assert!(matches!(
            persisted.verify_replay_and_bind(replay_evidence(&different_height)),
            Err(OccupancyReplayBindError::FieldMismatch {
                field: OccupancyReplayField::GridHeightCells,
                ..
            })
        ));

        let mut different_cell_bytes = encoded.clone();
        let cell_offset = OCCUPANCY_MAP_HEADER_BYTES + 5;
        different_cell_bytes[cell_offset] = (different_cell_bytes[cell_offset] + 1) % 3;
        rewrite_checksum(&mut different_cell_bytes);
        let different_cell = decode_occupancy_map(&different_cell_bytes, limits(12))
            .expect("decode valid different cell");
        let persisted = decode_persisted_occupancy_map(&encoded, limits(12))
            .expect("decode typed persisted fixture");
        assert!(matches!(
            persisted.verify_replay_and_bind(replay_evidence(&different_cell)),
            Err(OccupancyReplayBindError::CellClassMismatch { index: 5, .. })
        ));
    }

    #[test]
    fn replay_binding_compares_every_persisted_metadata_category_by_bits() {
        let original = fixture(4, 3, 0x42);
        let encoded = encode_occupancy_map(&original).expect("encode persisted fixture");

        assert_replay_field_mismatch(
            &encoded,
            12,
            |bytes| put_f64(bytes, 48, original.resolution_m() * 2.0),
            OccupancyReplayField::ResolutionMetersPerCell,
        );
        for axis in 0..2 {
            assert_replay_field_mismatch(
                &encoded,
                12,
                |bytes| {
                    put_f64(bytes, 56 + axis * 8, original.lower_bound_m()[axis] + 0.25);
                },
                OccupancyReplayField::LowerBoundMeters { axis },
            );
        }

        assert_replay_field_mismatch(
            &encoded,
            12,
            |bytes| {
                let rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
                for (index, value) in rotation.into_iter().flatten().enumerate() {
                    put_f64(bytes, 72 + index * 8, value);
                }
            },
            OccupancyReplayField::WorldToOccupancyRotation { row: 0, column: 0 },
        );
        for axis in 0..3 {
            assert_replay_field_mismatch(
                &encoded,
                12,
                |bytes| {
                    put_f64(
                        bytes,
                        144 + axis * 8,
                        original.world_to_occupancy().translation_m()[axis] + 0.25,
                    );
                },
                OccupancyReplayField::WorldToOccupancyTranslationMeters { axis },
            );
        }

        assert_replay_field_mismatch(
            &encoded,
            12,
            |bytes| put_f64(bytes, 168, original.height_range().minimum_m() - 0.25),
            OccupancyReplayField::MinimumHeightMeters,
        );
        assert_replay_field_mismatch(
            &encoded,
            12,
            |bytes| put_f64(bytes, 176, original.height_range().maximum_m() + 0.25),
            OccupancyReplayField::MaximumHeightMeters,
        );
        assert_replay_field_mismatch(
            &encoded,
            12,
            |bytes| put_u64(bytes, 184, original.revision().wrapping_add(1)),
            OccupancyReplayField::Revision,
        );
    }

    #[test]
    fn every_single_byte_bit_corruption_is_rejected() {
        let snapshot = fixture(9, 7, 42);
        let encoded = encode_occupancy_map(&snapshot).expect("encode fixture");
        let decode_limits = limits(snapshot.geometry().cell_count() + 8);
        for index in 0..encoded.len() {
            let mut corrupt = encoded.clone();
            corrupt[index] ^= 1_u8 << (index % 8);
            assert!(
                decode_occupancy_map(&corrupt, decode_limits).is_err(),
                "single-bit corruption at byte {index} was accepted"
            );
        }
    }

    #[test]
    fn every_strict_prefix_and_every_suffix_is_rejected() {
        let snapshot = fixture(5, 4, 17);
        let encoded = encode_occupancy_map(&snapshot).expect("encode fixture");
        let decode_limits = limits(snapshot.geometry().cell_count() + 32);
        for prefix_len in 0..encoded.len() {
            assert!(
                decode_occupancy_map(&encoded[..prefix_len], decode_limits).is_err(),
                "strict prefix of {prefix_len} bytes was accepted"
            );
        }
        for suffix_len in 1..=16 {
            let mut with_suffix = encoded.clone();
            with_suffix.extend(std::iter::repeat_n(0x5a, suffix_len));
            assert!(matches!(
                decode_occupancy_map(&with_suffix, decode_limits),
                Err(OccupancyMapDecodeError::TrailingBytes { .. })
            ));
        }
    }

    #[test]
    fn valid_checksum_does_not_bypass_finite_domain_metadata() {
        let snapshot = fixture(4, 3, 8);
        let encoded = encode_occupancy_map(&snapshot).expect("encode fixture");
        let decode_limits = limits(snapshot.geometry().cell_count());

        for offset in [48, 56, 64] {
            let mut invalid = encoded.clone();
            put_f64(&mut invalid, offset, f64::NAN);
            rewrite_checksum(&mut invalid);
            assert!(matches!(
                decode_occupancy_map(&invalid, decode_limits),
                Err(OccupancyMapDecodeError::Geometry(_))
            ));
        }
        for offset in [72, 144] {
            let mut invalid = encoded.clone();
            put_f64(&mut invalid, offset, f64::INFINITY);
            rewrite_checksum(&mut invalid);
            assert!(matches!(
                decode_occupancy_map(&invalid, decode_limits),
                Err(OccupancyMapDecodeError::WorldToOccupancy(_))
            ));
        }
        for offset in [168, 176] {
            let mut invalid = encoded.clone();
            put_f64(&mut invalid, offset, f64::NEG_INFINITY);
            rewrite_checksum(&mut invalid);
            assert!(matches!(
                decode_occupancy_map(&invalid, decode_limits),
                Err(OccupancyMapDecodeError::HeightRange(_))
            ));
        }
    }

    #[test]
    fn reserved_and_discriminator_fields_are_exact() {
        let snapshot = fixture(2, 2, 5);
        let encoded = encode_occupancy_map(&snapshot).expect("encode fixture");
        let decode_limits = limits(4);

        for offset in [12, 13, 14, 15, 19, 28, 29, 30, 31] {
            let mut invalid = encoded.clone();
            invalid[offset] = 1;
            rewrite_checksum(&mut invalid);
            assert!(matches!(
                decode_occupancy_map(&invalid, decode_limits),
                Err(OccupancyMapDecodeError::NonZeroReservedByte {
                    offset: actual,
                    value: 1
                }) if actual == offset
            ));
        }

        for (offset, expected) in [(16, "coordinate"), (17, "row"), (18, "cell")] {
            let mut invalid = encoded.clone();
            invalid[offset] = 0xff;
            rewrite_checksum(&mut invalid);
            let error = decode_occupancy_map(&invalid, decode_limits)
                .expect_err("unknown format discriminator must fail");
            assert!(
                matches!(
                    (expected, error),
                    (
                        "coordinate",
                        OccupancyMapDecodeError::UnsupportedCoordinateFrame { code: 0xff },
                    ) | (
                        "row",
                        OccupancyMapDecodeError::UnsupportedRowOrder { code: 0xff }
                    ) | (
                        "cell",
                        OccupancyMapDecodeError::UnsupportedCellEncoding { code: 0xff }
                    )
                ),
                "wrong error for {expected} discriminator"
            );
        }
    }

    #[test]
    fn declarations_classes_version_and_magic_are_strict() {
        let snapshot = fixture(3, 3, 99);
        let encoded = encode_occupancy_map(&snapshot).expect("encode fixture");
        let decode_limits = limits(64);

        let mut wrong_count = encoded.clone();
        put_u64(&mut wrong_count, 32, 8);
        rewrite_checksum(&mut wrong_count);
        assert!(matches!(
            decode_occupancy_map(&wrong_count, decode_limits),
            Err(OccupancyMapDecodeError::DeclaredCellCountMismatch { .. })
        ));

        let mut wrong_length = encoded.clone();
        put_u64(&mut wrong_length, 40, 8);
        rewrite_checksum(&mut wrong_length);
        assert!(matches!(
            decode_occupancy_map(&wrong_length, decode_limits),
            Err(OccupancyMapDecodeError::PayloadLengthMismatch { .. })
        ));

        let mut wrong_class = encoded.clone();
        wrong_class[OCCUPANCY_MAP_HEADER_BYTES + 4] = 3;
        rewrite_checksum(&mut wrong_class);
        assert!(matches!(
            decode_occupancy_map(&wrong_class, decode_limits),
            Err(OccupancyMapDecodeError::UnsupportedCellClass {
                index: 4,
                class_id: 3
            })
        ));

        let mut wrong_version = encoded.clone();
        put_u16(&mut wrong_version, 8, 2);
        rewrite_checksum(&mut wrong_version);
        assert!(matches!(
            decode_occupancy_map(&wrong_version, decode_limits),
            Err(OccupancyMapDecodeError::UnsupportedVersion { version: 2 })
        ));

        let mut wrong_header_length = encoded.clone();
        put_u16(&mut wrong_header_length, 10, 191);
        rewrite_checksum(&mut wrong_header_length);
        assert!(matches!(
            decode_occupancy_map(&wrong_header_length, decode_limits),
            Err(OccupancyMapDecodeError::InvalidHeaderLength { bytes: 191 })
        ));

        let mut wrong_magic = encoded;
        wrong_magic[0] = b'X';
        rewrite_checksum(&mut wrong_magic);
        assert!(matches!(
            decode_occupancy_map(&wrong_magic, decode_limits),
            Err(OccupancyMapDecodeError::MagicMismatch { .. })
        ));
    }

    #[test]
    fn payload_or_checksum_mutation_reports_checksum_mismatch() {
        let snapshot = fixture(2, 2, 1);
        let encoded = encode_occupancy_map(&snapshot).expect("encode fixture");
        for offset in [OCCUPANCY_MAP_HEADER_BYTES, encoded.len() - 1] {
            let mut corrupt = encoded.clone();
            corrupt[offset] ^= 0x80;
            assert!(matches!(
                decode_occupancy_map(&corrupt, limits(4)),
                Err(OccupancyMapDecodeError::ChecksumMismatch { .. })
            ));
        }
    }

    #[test]
    fn allocation_bombs_are_rejected_from_header_before_payload_allocation() {
        let snapshot = fixture(1, 1, 3);
        let mut bomb = encode_occupancy_map(&snapshot).expect("encode fixture");
        put_u32(&mut bomb, 20, u32::MAX);
        put_u32(&mut bomb, 24, u32::MAX);
        let computed = u64::from(u32::MAX) * u64::from(u32::MAX);
        put_u64(&mut bomb, 32, computed);
        put_u64(&mut bomb, 40, computed);
        rewrite_checksum(&mut bomb);
        assert!(matches!(
            decode_occupancy_map(&bomb, limits(1_024)),
            Err(OccupancyMapDecodeError::Geometry(
                OccupancyGridGeometryError::TooManyCells { .. }
                    | OccupancyGridGeometryError::CellCountNotAddressable { .. }
            ))
        ));

        let too_large = vec![0_u8; limits(4).maximum_encoded_bytes() + 1];
        assert!(matches!(
            decode_occupancy_map(&too_large, limits(4)),
            Err(OccupancyMapDecodeError::InputExceedsLimit { .. })
        ));
    }

    #[test]
    fn atomic_save_replaces_and_loads_without_temporary_residue() {
        let directory = TestDirectory::create("atomic");
        let destination = directory.path().join("room.kiko2d");
        fs::write(&destination, b"obsolete").expect("seed destination");

        let first = fixture(8, 6, 11);
        save_occupancy_map_atomic(&destination, &first).expect("atomically save first map");
        assert_eq!(
            fs::read(&destination).expect("read published map"),
            encode_occupancy_map(&first).expect("encode expected first map")
        );
        let loaded = load_persisted_occupancy_map(&destination, limits(48))
            .expect("load typed published map");
        assert_eq!(loaded.snapshot().map_instance_id(), None);
        assert_exact_snapshot(&first, loaded.snapshot());
        assert_no_temporary_files(directory.path());

        let replacement = fixture(4, 3, 12);
        save_occupancy_map_atomic(&destination, &replacement)
            .expect("atomically replace published map");
        let loaded = load_occupancy_map(&destination, limits(48)).expect("load replacement map");
        assert_exact_snapshot(&replacement, &loaded);
        assert_no_temporary_files(directory.path());
    }

    #[test]
    fn failed_rename_reports_unpublished_and_cleans_temporary_file() {
        let directory = TestDirectory::create("rename-failure");
        let destination = directory.path().join("existing-directory");
        fs::create_dir(&destination).expect("create conflicting directory");
        let snapshot = fixture(2, 2, 7);
        let error = save_occupancy_map_atomic(&destination, &snapshot)
            .expect_err("renaming a file over a directory must fail");
        match error {
            OccupancyMapSaveError::Io(source) => {
                assert_eq!(source.operation(), OccupancyMapSaveOperation::PublishRename);
                assert!(!source.published());
                assert!(source.temporary_path().is_some());
                assert!(source.cleanup_error().is_none());
            }
            other => panic!("unexpected save error: {other}"),
        }
        assert!(destination.is_dir());
        assert_no_temporary_files(directory.path());
    }

    #[test]
    fn load_rejects_non_files_and_oversized_files_before_parsing() {
        let directory = TestDirectory::create("load-boundaries");
        assert!(matches!(
            load_occupancy_map(directory.path(), limits(4)),
            Err(OccupancyMapLoadError::NotRegularFile { .. })
        ));

        let oversized = directory.path().join("oversized.kiko2d");
        fs::write(
            &oversized,
            vec![0_u8; limits(4).maximum_encoded_bytes() + 1],
        )
        .expect("write oversized fixture");
        assert!(matches!(
            load_occupancy_map(&oversized, limits(4)),
            Err(OccupancyMapLoadError::Format(
                OccupancyMapDecodeError::InputExceedsLimit { .. }
            ))
        ));

        let missing = directory.path().join("missing.kiko2d");
        assert!(matches!(
            load_occupancy_map(&missing, limits(4)),
            Err(OccupancyMapLoadError::Io {
                operation: OccupancyMapLoadOperation::Open,
                ..
            })
        ));
    }
}
