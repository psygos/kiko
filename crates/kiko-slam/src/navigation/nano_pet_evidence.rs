//! Strict record/replay boundary for Fable-compatible pet episode evidence.
//!
//! The historical Python owner wrote one compact JSON object per line with
//! top-level `wall` and `episode` fields. V1 retains those fields for existing
//! analysis tools and adds exact integer wall and monotonic times. Parsing
//! accepts only the historical shape or V1; partial cross-version records are
//! rejected rather than repaired.

use core::fmt;
use core::num::{NonZeroU16, NonZeroU64};
use core::time::Duration;

use kiko_expression_runtime::{CharacterPetEpisode, CharacterPetReaction};
use kiko_head_runtime::compliant_hold::CompliantPetEpisodeSummary;
use serde::{Deserialize, Serialize};

pub const FABLE_PET_LOG_SCHEMA_LEGACY: u32 = 0;
pub const NANO_PET_EVIDENCE_SCHEMA_V1: u32 = 1;
pub const MAX_NANO_PET_EVIDENCE_RECORD_BYTES: usize = 4 * 1_024;

const JOINT_COUNT: usize = 4;
const NS_PER_CENTISECOND: u64 = 10_000_000;
const MS_PER_CENTISECOND: u64 = 10;
const MAX_EXACT_F64_INTEGER_U64: u64 = 9_007_199_254_740_992;
const MAX_EXACT_F64_INTEGER: f64 = MAX_EXACT_F64_INTEGER_U64 as f64;
const DECIMAL_INTEGER_TOLERANCE: f64 = 0.000_1;
const MAX_COMPATIBILITY_TIME_ERROR_NS: u64 = 4_096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoPetEvidenceFormat {
    FableLegacy,
    NanoV1,
}

/// Parsed episode facts shared by Fable's field logs and the Rust owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoPetEpisodeEvidence {
    format: NanoPetEvidenceFormat,
    wall_unix_centiseconds: u64,
    exact_wall_unix_ms: Option<u64>,
    started_monotonic_ns: Option<u64>,
    completed_monotonic_ns: Option<u64>,
    duration_ns: NonZeroU64,
    yield_entries: NonZeroU16,
    samples: u64,
    peak_residual_ticks: [u16; JOINT_COUNT],
    accumulated_max_delta_ticks: u64,
    delta_samples: u64,
    reached_rest: bool,
    reached_comfy: bool,
    tap: bool,
    mean_delta_hundredths: u64,
}

impl NanoPetEpisodeEvidence {
    /// Parse one complete NDJSON line. The final newline is part of the
    /// durable record identity and is mandatory. CRLF and trailing documents
    /// are rejected.
    pub fn parse_ndjson_line(line: &[u8]) -> Result<Self, NanoPetEvidenceDecodeError> {
        if line.len() > MAX_NANO_PET_EVIDENCE_RECORD_BYTES {
            return Err(NanoPetEvidenceDecodeError::RecordTooLarge {
                actual_bytes: line.len(),
                maximum_bytes: MAX_NANO_PET_EVIDENCE_RECORD_BYTES,
            });
        }
        let json = line
            .strip_suffix(b"\n")
            .ok_or(NanoPetEvidenceDecodeError::MissingNewline)?;
        if json.is_empty() || json.ends_with(b"\r") {
            return Err(NanoPetEvidenceDecodeError::InvalidLineEnding);
        }
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = PetRecordDto::deserialize(&mut deserializer)
            .map_err(NanoPetEvidenceDecodeError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoPetEvidenceDecodeError::JsonTrailingData)?;
        Self::from_dto(dto)
    }

    /// Build one V1 record from a committed controller summary and a wall
    /// clock observation captured at completion. This function performs no I/O.
    pub fn from_completed_summary(
        summary: CompliantPetEpisodeSummary,
        completed_wall_unix_ms: u64,
    ) -> Result<Self, NanoPetEvidenceEncodeError> {
        let started_monotonic_ns = duration_to_u64_ns(summary.started_at().duration_since_origin())
            .ok_or(NanoPetEvidenceEncodeError::MonotonicTimeOutOfRange {
                field: "started_monotonic_ns",
            })?;
        let completed_monotonic_ns = duration_to_u64_ns(
            summary.completed_at().duration_since_origin(),
        )
        .ok_or(NanoPetEvidenceEncodeError::MonotonicTimeOutOfRange {
            field: "completed_monotonic_ns",
        })?;
        let duration_ns = NonZeroU64::new(
            completed_monotonic_ns
                .checked_sub(started_monotonic_ns)
                .ok_or(NanoPetEvidenceEncodeError::MonotonicTimeRegressed)?,
        )
        .ok_or(NanoPetEvidenceEncodeError::ZeroDuration)?;
        let yield_entries = NonZeroU16::new(summary.yield_entries())
            .ok_or(NanoPetEvidenceEncodeError::ZeroYieldEntries)?;
        let character = CharacterPetEpisode::try_new(
            Duration::from_nanos(duration_ns.get()),
            summary.accumulated_max_delta_ticks(),
            summary.delta_samples(),
            summary.reached_comfy(),
            summary.was_tap(),
        )
        .map_err(NanoPetEvidenceEncodeError::CharacterEpisode)?;
        let evidence = Self {
            format: NanoPetEvidenceFormat::NanoV1,
            wall_unix_centiseconds: round_half_even_u64(completed_wall_unix_ms, MS_PER_CENTISECOND),
            exact_wall_unix_ms: Some(completed_wall_unix_ms),
            started_monotonic_ns: Some(started_monotonic_ns),
            completed_monotonic_ns: Some(completed_monotonic_ns),
            duration_ns,
            yield_entries,
            samples: summary.samples(),
            peak_residual_ticks: summary.peak_residual_ticks(),
            accumulated_max_delta_ticks: summary.accumulated_max_delta_ticks(),
            delta_samples: summary.delta_samples(),
            reached_rest: summary.reached_rest(),
            reached_comfy: summary.reached_comfy(),
            tap: summary.was_tap(),
            mean_delta_hundredths: character.mean_delta_hundredths(),
        };
        evidence
            .require_cross_field_invariants()
            .map_err(NanoPetEvidenceEncodeError::Invariant)?;
        Ok(evidence)
    }

    /// Encode V1 while retaining Fable's historical keys. Exact integer
    /// fields are authoritative; decimal compatibility fields are derived.
    pub fn encode_ndjson_line(self) -> Result<Vec<u8>, NanoPetEvidenceEncodeError> {
        if self.format != NanoPetEvidenceFormat::NanoV1 {
            return Err(NanoPetEvidenceEncodeError::LegacyRecordCannotBeEncodedAsV1);
        }
        let exact_wall_unix_ms =
            self.exact_wall_unix_ms
                .ok_or(NanoPetEvidenceEncodeError::MissingExactV1Field {
                    field: "wall_unix_ms",
                })?;
        let started_monotonic_ns =
            self.started_monotonic_ns
                .ok_or(NanoPetEvidenceEncodeError::MissingExactV1Field {
                    field: "started_monotonic_ns",
                })?;
        let completed_monotonic_ns =
            self.completed_monotonic_ns
                .ok_or(NanoPetEvidenceEncodeError::MissingExactV1Field {
                    field: "completed_monotonic_ns",
                })?;
        if self.wall_unix_centiseconds > MAX_EXACT_F64_INTEGER_U64 {
            return Err(NanoPetEvidenceEncodeError::CompatibilityDecimalOutOfRange {
                field: "wall",
            });
        }
        if self.mean_delta_hundredths > MAX_EXACT_F64_INTEGER_U64 {
            return Err(NanoPetEvidenceEncodeError::CompatibilityDecimalOutOfRange {
                field: "episode.mean_delta",
            });
        }
        let record = PetRecordV1 {
            schema_version: NANO_PET_EVIDENCE_SCHEMA_V1,
            wall: self.wall_unix_centiseconds as f64 / 100.0,
            wall_unix_ms: exact_wall_unix_ms,
            completed_monotonic_ns,
            episode: PetEpisodeV1 {
                started_at: started_monotonic_ns as f64 / 1_000_000_000.0,
                started_monotonic_ns,
                completed_monotonic_ns,
                yield_entries: self.yield_entries.get(),
                samples: self.samples,
                peak_residual: self.peak_residual_ticks,
                delta_accum: self.accumulated_max_delta_ticks,
                delta_samples: self.delta_samples,
                reached_rest: self.reached_rest,
                reached_comfy: self.reached_comfy,
                tap: self.tap,
                duration_s: round_duration_centiseconds(self.duration_ns.get()) as f64 / 100.0,
                duration_ns: self.duration_ns.get(),
                mean_delta: self.mean_delta_hundredths as f64 / 100.0,
            },
        };
        let mut bytes = serde_json::to_vec(&record).map_err(NanoPetEvidenceEncodeError::Json)?;
        bytes.push(b'\n');
        if bytes.len() > MAX_NANO_PET_EVIDENCE_RECORD_BYTES {
            return Err(NanoPetEvidenceEncodeError::RecordTooLarge {
                actual_bytes: bytes.len(),
                maximum_bytes: MAX_NANO_PET_EVIDENCE_RECORD_BYTES,
            });
        }
        Ok(bytes)
    }

    pub const fn format(self) -> NanoPetEvidenceFormat {
        self.format
    }

    pub const fn wall_unix_centiseconds(self) -> u64 {
        self.wall_unix_centiseconds
    }

    pub const fn exact_wall_unix_ms(self) -> Option<u64> {
        self.exact_wall_unix_ms
    }

    /// Exact for V1; rounded from Fable's compatibility seconds for legacy
    /// records.
    pub const fn started_monotonic_ns(self) -> Option<u64> {
        self.started_monotonic_ns
    }

    /// Exact for V1; derived as legacy start plus recorded duration for Fable
    /// records.
    pub const fn completed_monotonic_ns(self) -> Option<u64> {
        self.completed_monotonic_ns
    }

    pub const fn duration_ns(self) -> NonZeroU64 {
        self.duration_ns
    }

    pub const fn yield_entries(self) -> NonZeroU16 {
        self.yield_entries
    }

    pub const fn samples(self) -> u64 {
        self.samples
    }

    pub const fn peak_residual_ticks(self) -> [u16; JOINT_COUNT] {
        self.peak_residual_ticks
    }

    pub const fn accumulated_max_delta_ticks(self) -> u64 {
        self.accumulated_max_delta_ticks
    }

    pub const fn delta_samples(self) -> u64 {
        self.delta_samples
    }

    pub const fn reached_rest(self) -> bool {
        self.reached_rest
    }

    pub const fn reached_comfy(self) -> bool {
        self.reached_comfy
    }

    pub const fn was_tap(self) -> bool {
        self.tap
    }

    pub const fn mean_delta_hundredths(self) -> u64 {
        self.mean_delta_hundredths
    }

    pub fn replay_comparison(self) -> Result<NanoPetReplayComparison, NanoPetEvidenceDecodeError> {
        let episode = CharacterPetEpisode::try_new(
            Duration::from_nanos(self.duration_ns.get()),
            self.accumulated_max_delta_ticks,
            self.delta_samples,
            self.reached_comfy,
            self.tap,
        )
        .map_err(NanoPetEvidenceDecodeError::CharacterEpisode)?;
        let expected = if self.tap {
            CharacterPetReaction::Boop
        } else if self.mean_delta_hundredths >= 600 && !self.reached_comfy {
            CharacterPetReaction::Play
        } else {
            CharacterPetReaction::Affection
        };
        Ok(NanoPetReplayComparison {
            expected_fable_reaction: expected,
            actual_rust_reaction: episode.reaction(),
        })
    }

    fn from_dto(dto: PetRecordDto) -> Result<Self, NanoPetEvidenceDecodeError> {
        let format = match dto.schema_version {
            None => NanoPetEvidenceFormat::FableLegacy,
            Some(NANO_PET_EVIDENCE_SCHEMA_V1) => NanoPetEvidenceFormat::NanoV1,
            Some(actual) => {
                return Err(NanoPetEvidenceDecodeError::UnsupportedSchema {
                    actual,
                    supported: NANO_PET_EVIDENCE_SCHEMA_V1,
                });
            }
        };
        let wall_unix_centiseconds = parse_decimal_hundredths("wall", dto.wall)?;
        let duration_centiseconds =
            parse_decimal_hundredths("episode.duration_s", dto.episode.duration_s)?;
        let duration_ns_from_compatibility = duration_centiseconds
            .checked_mul(NS_PER_CENTISECOND)
            .and_then(NonZeroU64::new)
            .ok_or(NanoPetEvidenceDecodeError::InvalidDuration)?;
        let mean_delta_hundredths =
            parse_decimal_hundredths("episode.mean_delta", dto.episode.mean_delta)?;
        let accumulated_max_delta_ticks =
            parse_integral_number("episode.delta_accum", &dto.episode.delta_accum)?;
        let peak_residual_ticks = parse_peak_residuals(dto.episode.peak_residual)?;
        let yield_entries = NonZeroU16::new(dto.episode.yield_entries)
            .ok_or(NanoPetEvidenceDecodeError::ZeroYieldEntries)?;

        let (exact_wall_unix_ms, started_monotonic_ns, completed_monotonic_ns, duration_ns) =
            match format {
                NanoPetEvidenceFormat::FableLegacy => {
                    if dto.wall_unix_ms.is_some()
                        || dto.completed_monotonic_ns.is_some()
                        || dto.episode.started_monotonic_ns.is_some()
                        || dto.episode.completed_monotonic_ns.is_some()
                        || dto.episode.duration_ns.is_some()
                    {
                        return Err(NanoPetEvidenceDecodeError::LegacyContainsV1Fields);
                    }
                    let started =
                        parse_seconds_to_ns("episode.started_at", dto.episode.started_at)?;
                    let completed = started
                        .checked_add(duration_ns_from_compatibility.get())
                        .ok_or(NanoPetEvidenceDecodeError::NumericOutOfRange {
                            field: "episode.completed_at",
                        })?;
                    (
                        None,
                        Some(started),
                        Some(completed),
                        duration_ns_from_compatibility,
                    )
                }
                NanoPetEvidenceFormat::NanoV1 => {
                    let wall_unix_ms =
                        dto.wall_unix_ms
                            .ok_or(NanoPetEvidenceDecodeError::MissingV1Field {
                                field: "wall_unix_ms",
                            })?;
                    if round_half_even_u64(wall_unix_ms, MS_PER_CENTISECOND)
                        != wall_unix_centiseconds
                    {
                        return Err(NanoPetEvidenceDecodeError::WallTimeMismatch);
                    }
                    let top_completed = dto.completed_monotonic_ns.ok_or(
                        NanoPetEvidenceDecodeError::MissingV1Field {
                            field: "completed_monotonic_ns",
                        },
                    )?;
                    let started = dto.episode.started_monotonic_ns.ok_or(
                        NanoPetEvidenceDecodeError::MissingV1Field {
                            field: "episode.started_monotonic_ns",
                        },
                    )?;
                    let completed = dto.episode.completed_monotonic_ns.ok_or(
                        NanoPetEvidenceDecodeError::MissingV1Field {
                            field: "episode.completed_monotonic_ns",
                        },
                    )?;
                    if top_completed != completed {
                        return Err(NanoPetEvidenceDecodeError::CompletedTimeMismatch);
                    }
                    let duration_ns = NonZeroU64::new(dto.episode.duration_ns.ok_or(
                        NanoPetEvidenceDecodeError::MissingV1Field {
                            field: "episode.duration_ns",
                        },
                    )?)
                    .ok_or(NanoPetEvidenceDecodeError::InvalidDuration)?;
                    if completed.checked_sub(started) != Some(duration_ns.get()) {
                        return Err(NanoPetEvidenceDecodeError::DurationTimeMismatch);
                    }
                    if round_duration_centiseconds(duration_ns.get()) != duration_centiseconds {
                        return Err(NanoPetEvidenceDecodeError::DurationDecimalMismatch);
                    }
                    let compatibility_started_ns =
                        parse_seconds_to_ns("episode.started_at", dto.episode.started_at)?;
                    if started.max(compatibility_started_ns) - started.min(compatibility_started_ns)
                        > MAX_COMPATIBILITY_TIME_ERROR_NS
                    {
                        return Err(NanoPetEvidenceDecodeError::StartedTimeMismatch);
                    }
                    (
                        Some(wall_unix_ms),
                        Some(started),
                        Some(completed),
                        duration_ns,
                    )
                }
            };

        let evidence = Self {
            format,
            wall_unix_centiseconds,
            exact_wall_unix_ms,
            started_monotonic_ns,
            completed_monotonic_ns,
            duration_ns,
            yield_entries,
            samples: dto.episode.samples,
            peak_residual_ticks,
            accumulated_max_delta_ticks,
            delta_samples: dto.episode.delta_samples,
            reached_rest: dto.episode.reached_rest,
            reached_comfy: dto.episode.reached_comfy,
            tap: dto.episode.tap,
            mean_delta_hundredths,
        };
        evidence.require_cross_field_invariants()?;
        Ok(evidence)
    }

    fn require_cross_field_invariants(self) -> Result<(), NanoPetEvidenceDecodeError> {
        if self.delta_samples > self.samples {
            return Err(NanoPetEvidenceDecodeError::DeltaSamplesAboveSamples {
                delta_samples: self.delta_samples,
                samples: self.samples,
            });
        }
        if self.delta_samples == 0 && self.accumulated_max_delta_ticks != 0 {
            return Err(NanoPetEvidenceDecodeError::DeltaWithoutSamples);
        }
        let episode = CharacterPetEpisode::try_new(
            Duration::from_nanos(self.duration_ns.get()),
            self.accumulated_max_delta_ticks,
            self.delta_samples,
            self.reached_comfy,
            self.tap,
        )
        .map_err(NanoPetEvidenceDecodeError::CharacterEpisode)?;
        if episode.mean_delta_hundredths() != self.mean_delta_hundredths {
            return Err(NanoPetEvidenceDecodeError::MeanDeltaMismatch {
                recorded_hundredths: self.mean_delta_hundredths,
                derived_hundredths: episode.mean_delta_hundredths(),
            });
        }
        if self.reached_comfy && !self.reached_rest {
            return Err(NanoPetEvidenceDecodeError::ComfyWithoutRest);
        }
        if self.tap && (self.yield_entries.get() != 1 || self.reached_rest) {
            return Err(NanoPetEvidenceDecodeError::InvalidTapEpisode);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoPetReplayComparison {
    expected_fable_reaction: CharacterPetReaction,
    actual_rust_reaction: CharacterPetReaction,
}

impl NanoPetReplayComparison {
    pub const fn expected_fable_reaction(self) -> CharacterPetReaction {
        self.expected_fable_reaction
    }

    pub const fn actual_rust_reaction(self) -> CharacterPetReaction {
        self.actual_rust_reaction
    }

    pub fn matches(self) -> bool {
        self.expected_fable_reaction == self.actual_rust_reaction
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PetRecordDto {
    #[serde(default)]
    schema_version: Option<u32>,
    wall: f64,
    #[serde(default)]
    wall_unix_ms: Option<u64>,
    #[serde(default)]
    completed_monotonic_ns: Option<u64>,
    episode: PetEpisodeDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PetEpisodeDto {
    started_at: f64,
    #[serde(default)]
    started_monotonic_ns: Option<u64>,
    #[serde(default)]
    completed_monotonic_ns: Option<u64>,
    yield_entries: u16,
    samples: u64,
    peak_residual: [u64; JOINT_COUNT],
    delta_accum: serde_json::Number,
    delta_samples: u64,
    reached_rest: bool,
    reached_comfy: bool,
    tap: bool,
    duration_s: f64,
    #[serde(default)]
    duration_ns: Option<u64>,
    mean_delta: f64,
}

#[derive(Serialize)]
struct PetRecordV1 {
    schema_version: u32,
    wall: f64,
    wall_unix_ms: u64,
    completed_monotonic_ns: u64,
    episode: PetEpisodeV1,
}

#[derive(Serialize)]
struct PetEpisodeV1 {
    started_at: f64,
    started_monotonic_ns: u64,
    completed_monotonic_ns: u64,
    yield_entries: u16,
    samples: u64,
    peak_residual: [u16; JOINT_COUNT],
    delta_accum: u64,
    delta_samples: u64,
    reached_rest: bool,
    reached_comfy: bool,
    tap: bool,
    duration_s: f64,
    duration_ns: u64,
    mean_delta: f64,
}

#[derive(Debug)]
pub enum NanoPetEvidenceDecodeError {
    RecordTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    MissingNewline,
    InvalidLineEnding,
    JsonDecode(serde_json::Error),
    JsonTrailingData(serde_json::Error),
    UnsupportedSchema {
        actual: u32,
        supported: u32,
    },
    LegacyContainsV1Fields,
    MissingV1Field {
        field: &'static str,
    },
    NonFiniteOrNegative {
        field: &'static str,
    },
    DecimalNotHundredths {
        field: &'static str,
    },
    IntegralValueRequired {
        field: &'static str,
    },
    NumericOutOfRange {
        field: &'static str,
    },
    InvalidDuration,
    ZeroYieldEntries,
    PeakResidualOutOfRange {
        value: u64,
    },
    WallTimeMismatch,
    CompletedTimeMismatch,
    DurationTimeMismatch,
    DurationDecimalMismatch,
    StartedTimeMismatch,
    DeltaSamplesAboveSamples {
        delta_samples: u64,
        samples: u64,
    },
    DeltaWithoutSamples,
    MeanDeltaMismatch {
        recorded_hundredths: u64,
        derived_hundredths: u64,
    },
    ComfyWithoutRest,
    InvalidTapEpisode,
    CharacterEpisode(kiko_expression_runtime::CharacterPetEpisodeError),
}

impl fmt::Display for NanoPetEvidenceDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Nano/Fable pet evidence record: {self:?}"
        )
    }
}

impl std::error::Error for NanoPetEvidenceDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::CharacterEpisode(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum NanoPetEvidenceEncodeError {
    MonotonicTimeOutOfRange {
        field: &'static str,
    },
    MonotonicTimeRegressed,
    ZeroDuration,
    ZeroYieldEntries,
    CharacterEpisode(kiko_expression_runtime::CharacterPetEpisodeError),
    Invariant(NanoPetEvidenceDecodeError),
    LegacyRecordCannotBeEncodedAsV1,
    MissingExactV1Field {
        field: &'static str,
    },
    CompatibilityDecimalOutOfRange {
        field: &'static str,
    },
    Json(serde_json::Error),
    RecordTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
}

impl fmt::Display for NanoPetEvidenceEncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot encode Nano pet evidence record: {self:?}"
        )
    }
}

impl std::error::Error for NanoPetEvidenceEncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CharacterEpisode(source) => Some(source),
            Self::Invariant(source) => Some(source),
            Self::Json(source) => Some(source),
            _ => None,
        }
    }
}

fn parse_peak_residuals(
    values: [u64; JOINT_COUNT],
) -> Result<[u16; JOINT_COUNT], NanoPetEvidenceDecodeError> {
    let mut parsed = [0_u16; JOINT_COUNT];
    for (index, value) in values.into_iter().enumerate() {
        parsed[index] = u16::try_from(value)
            .map_err(|_| NanoPetEvidenceDecodeError::PeakResidualOutOfRange { value })?;
    }
    Ok(parsed)
}

fn parse_nonnegative_finite(
    field: &'static str,
    value: f64,
) -> Result<f64, NanoPetEvidenceDecodeError> {
    if !value.is_finite() || value < 0.0 {
        return Err(NanoPetEvidenceDecodeError::NonFiniteOrNegative { field });
    }
    Ok(value)
}

fn parse_decimal_hundredths(
    field: &'static str,
    value: f64,
) -> Result<u64, NanoPetEvidenceDecodeError> {
    let value = parse_nonnegative_finite(field, value)?;
    let scaled = value * 100.0;
    if scaled > MAX_EXACT_F64_INTEGER {
        return Err(NanoPetEvidenceDecodeError::NumericOutOfRange { field });
    }
    let rounded = scaled.round();
    if (scaled - rounded).abs() > DECIMAL_INTEGER_TOLERANCE {
        return Err(NanoPetEvidenceDecodeError::DecimalNotHundredths { field });
    }
    Ok(rounded as u64)
}

fn parse_integral_number(
    field: &'static str,
    number: &serde_json::Number,
) -> Result<u64, NanoPetEvidenceDecodeError> {
    if let Some(value) = number.as_u64() {
        return Ok(value);
    }
    let value = number
        .as_f64()
        .ok_or(NanoPetEvidenceDecodeError::NumericOutOfRange { field })?;
    let value = parse_nonnegative_finite(field, value)?;
    if value > MAX_EXACT_F64_INTEGER {
        return Err(NanoPetEvidenceDecodeError::NumericOutOfRange { field });
    }
    let rounded = value.round();
    if value != rounded {
        return Err(NanoPetEvidenceDecodeError::IntegralValueRequired { field });
    }
    Ok(rounded as u64)
}

fn parse_seconds_to_ns(
    field: &'static str,
    seconds: f64,
) -> Result<u64, NanoPetEvidenceDecodeError> {
    let seconds = parse_nonnegative_finite(field, seconds)?;
    let nanoseconds = seconds * 1_000_000_000.0;
    if !nanoseconds.is_finite() || nanoseconds >= u64::MAX as f64 {
        return Err(NanoPetEvidenceDecodeError::NumericOutOfRange { field });
    }
    Ok(nanoseconds.round() as u64)
}

const fn round_half_even_u64(numerator: u64, divisor: u64) -> u64 {
    let quotient = numerator / divisor;
    let remainder = numerator % divisor;
    let complement = divisor - remainder;
    quotient
        + if remainder > complement || (remainder == complement && quotient % 2 == 1) {
            1
        } else {
            0
        }
}

const fn round_duration_centiseconds(duration_ns: u64) -> u64 {
    round_half_even_u64(duration_ns, NS_PER_CENTISECOND)
}

fn duration_to_u64_ns(duration: Duration) -> Option<u64> {
    u64::try_from(duration.as_nanos()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FABLE_AFFECTION: &[u8] = concat!(
        r#"{"wall":1787759000.25,"episode":{"started_at":1234.5,"yield_entries":1,"samples":41,"peak_residual":[12,4,3,2],"delta_accum":20.0,"delta_samples":40,"reached_rest":true,"reached_comfy":true,"tap":false,"duration_s":4.2,"mean_delta":0.5}}"#,
        "\n"
    )
    .as_bytes();

    #[test]
    fn historical_fable_line_parses_once_and_replays_the_same_reaction() {
        let record = NanoPetEpisodeEvidence::parse_ndjson_line(FABLE_AFFECTION)
            .expect("historical Fable record");
        assert_eq!(record.format(), NanoPetEvidenceFormat::FableLegacy);
        assert_eq!(record.wall_unix_centiseconds(), 178_775_900_025);
        assert_eq!(record.exact_wall_unix_ms(), None);
        assert_eq!(record.started_monotonic_ns(), Some(1_234_500_000_000));
        assert_eq!(record.completed_monotonic_ns(), Some(1_238_700_000_000));
        assert_eq!(record.duration_ns().get(), 4_200_000_000);
        assert_eq!(record.peak_residual_ticks(), [12, 4, 3, 2]);
        assert_eq!(record.mean_delta_hundredths(), 50);
        let comparison = record.replay_comparison().expect("typed replay");
        assert!(comparison.matches());
        assert_eq!(
            comparison.actual_rust_reaction(),
            CharacterPetReaction::Affection
        );
    }

    #[test]
    fn nano_v1_round_trips_exact_fields_and_retains_fable_compatibility_fields() {
        let original = NanoPetEpisodeEvidence {
            format: NanoPetEvidenceFormat::NanoV1,
            wall_unix_centiseconds: 178_775_900_025,
            exact_wall_unix_ms: Some(1_787_759_000_250),
            started_monotonic_ns: Some(10_000_000_000),
            completed_monotonic_ns: Some(14_200_000_000),
            duration_ns: NonZeroU64::new(4_200_000_000).unwrap(),
            yield_entries: NonZeroU16::new(2).unwrap(),
            samples: 41,
            peak_residual_ticks: [12, 4, 3, 2],
            accumulated_max_delta_ticks: 240,
            delta_samples: 40,
            reached_rest: true,
            reached_comfy: false,
            tap: false,
            mean_delta_hundredths: 600,
        };

        let line = original.encode_ndjson_line().expect("V1 record");
        let parsed = NanoPetEpisodeEvidence::parse_ndjson_line(&line).expect("round trip");
        assert_eq!(parsed, original);
        let comparison = parsed.replay_comparison().expect("typed replay");
        assert!(comparison.matches());
        assert_eq!(
            comparison.actual_rust_reaction(),
            CharacterPetReaction::Play
        );

        let json: serde_json::Value =
            serde_json::from_slice(&line[..line.len() - 1]).expect("compatibility JSON");
        assert_eq!(json["wall"].as_f64(), Some(1_787_759_000.25));
        assert_eq!(json["episode"]["duration_s"].as_f64(), Some(4.2));
        assert_eq!(json["episode"]["mean_delta"].as_f64(), Some(6.0));
    }

    #[test]
    fn fable_rounding_boundary_matches_the_rust_character_decision() {
        let line = concat!(
            r#"{"wall":1787759000.25,"episode":{"started_at":1234.5,"yield_entries":1,"samples":201,"peak_residual":[12,4,3,2],"delta_accum":1199.0,"delta_samples":200,"reached_rest":false,"reached_comfy":false,"tap":false,"duration_s":2.0,"mean_delta":6.0}}"#,
            "\n"
        );
        let record =
            NanoPetEpisodeEvidence::parse_ndjson_line(line.as_bytes()).expect("boundary record");
        let comparison = record.replay_comparison().expect("typed replay");
        assert!(comparison.matches());
        assert_eq!(
            comparison.actual_rust_reaction(),
            CharacterPetReaction::Play
        );
    }

    #[test]
    fn unknown_fields_inconsistent_means_and_cross_version_hybrids_fail_closed() {
        let mut unknown: serde_json::Value =
            serde_json::from_slice(&FABLE_AFFECTION[..FABLE_AFFECTION.len() - 1]).unwrap();
        unknown["surprise"] = serde_json::json!(true);
        let mut bytes = serde_json::to_vec(&unknown).unwrap();
        bytes.push(b'\n');
        assert!(matches!(
            NanoPetEpisodeEvidence::parse_ndjson_line(&bytes),
            Err(NanoPetEvidenceDecodeError::JsonDecode(_))
        ));

        let mut mismatch: serde_json::Value =
            serde_json::from_slice(&FABLE_AFFECTION[..FABLE_AFFECTION.len() - 1]).unwrap();
        mismatch["episode"]["mean_delta"] = serde_json::json!(0.51);
        let mut bytes = serde_json::to_vec(&mismatch).unwrap();
        bytes.push(b'\n');
        assert!(matches!(
            NanoPetEpisodeEvidence::parse_ndjson_line(&bytes),
            Err(NanoPetEvidenceDecodeError::MeanDeltaMismatch { .. })
        ));

        let mut hybrid: serde_json::Value =
            serde_json::from_slice(&FABLE_AFFECTION[..FABLE_AFFECTION.len() - 1]).unwrap();
        hybrid["wall_unix_ms"] = serde_json::json!(1_787_759_000_250_u64);
        let mut bytes = serde_json::to_vec(&hybrid).unwrap();
        bytes.push(b'\n');
        assert!(matches!(
            NanoPetEpisodeEvidence::parse_ndjson_line(&bytes),
            Err(NanoPetEvidenceDecodeError::LegacyContainsV1Fields)
        ));
    }

    #[test]
    fn malformed_line_boundaries_are_not_repaired() {
        assert!(matches!(
            NanoPetEpisodeEvidence::parse_ndjson_line(
                &FABLE_AFFECTION[..FABLE_AFFECTION.len() - 1]
            ),
            Err(NanoPetEvidenceDecodeError::MissingNewline)
        ));
        let mut crlf = FABLE_AFFECTION.to_vec();
        crlf.insert(crlf.len() - 1, b'\r');
        assert!(matches!(
            NanoPetEpisodeEvidence::parse_ndjson_line(&crlf),
            Err(NanoPetEvidenceDecodeError::InvalidLineEnding)
        ));
    }
}
