use crate::{ArtifactId, InventoryParseError, Sha256Id, TextField};
use serde::Deserialize;

pub const MAX_CALIBRATION_ARTIFACTS: usize = 8;
pub const MAX_PLANT_ARTIFACTS: usize = 4;
pub const MAX_ARTIFACTS: usize = MAX_CALIBRATION_ARTIFACTS + MAX_PLANT_ARTIFACTS;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactDigestDto {
    pub artifact_id: String,
    pub sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArtifactKind {
    Calibration,
    Plant,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactDigest {
    kind: ArtifactKind,
    artifact_id: ArtifactId,
    sha256: Sha256Id,
}

impl ArtifactDigest {
    pub fn kind(&self) -> ArtifactKind {
        self.kind
    }

    pub fn artifact_id(&self) -> &ArtifactId {
        &self.artifact_id
    }

    pub fn sha256(&self) -> &Sha256Id {
        &self.sha256
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactSet {
    entries: [Option<ArtifactDigest>; MAX_ARTIFACTS],
    len: u8,
}

impl ArtifactSet {
    pub(crate) fn parse_expected(
        calibration: Vec<ArtifactDigestDto>,
        plant: Vec<ArtifactDigestDto>,
    ) -> Result<Self, InventoryParseError> {
        if calibration.is_empty() {
            return Err(InventoryParseError::MissingRequiredArtifactKind {
                kind: ArtifactKind::Calibration,
            });
        }
        if plant.is_empty() {
            return Err(InventoryParseError::MissingRequiredArtifactKind {
                kind: ArtifactKind::Plant,
            });
        }
        Self::parse(calibration, plant)
    }

    pub(crate) fn parse_observed(
        calibration: Vec<ArtifactDigestDto>,
        plant: Vec<ArtifactDigestDto>,
    ) -> Result<Self, InventoryParseError> {
        Self::parse(calibration, plant)
    }

    fn parse(
        calibration: Vec<ArtifactDigestDto>,
        plant: Vec<ArtifactDigestDto>,
    ) -> Result<Self, InventoryParseError> {
        require_count(
            ArtifactKind::Calibration,
            calibration.len(),
            MAX_CALIBRATION_ARTIFACTS,
        )?;
        require_count(ArtifactKind::Plant, plant.len(), MAX_PLANT_ARTIFACTS)?;

        let mut output = Self {
            entries: core::array::from_fn(|_| None),
            len: 0,
        };
        for (kind, values) in [
            (ArtifactKind::Calibration, calibration),
            (ArtifactKind::Plant, plant),
        ] {
            for (index, dto) in values.into_iter().enumerate() {
                let artifact_id = ArtifactId::parse(dto.artifact_id).map_err(|source| {
                    InventoryParseError::InvalidText {
                        field: TextField::ArtifactId { kind, index },
                        source,
                    }
                })?;
                let sha256 = Sha256Id::try_new(dto.sha256).ok_or(
                    InventoryParseError::ZeroArtifactDigest {
                        kind,
                        index,
                        artifact_id,
                    },
                )?;
                if output
                    .iter()
                    .any(|existing| existing.artifact_id == artifact_id)
                {
                    return Err(InventoryParseError::DuplicateArtifactId { artifact_id });
                }
                if output.iter().any(|existing| existing.sha256 == sha256) {
                    return Err(InventoryParseError::DuplicateArtifactDigest { sha256 });
                }
                output.entries[usize::from(output.len)] = Some(ArtifactDigest {
                    kind,
                    artifact_id,
                    sha256,
                });
                output.len += 1;
            }
        }
        let output_len = output.len();
        output.entries[..output_len].sort_unstable();
        Ok(output)
    }

    pub fn len(&self) -> usize {
        usize::from(self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &ArtifactDigest> + '_ {
        self.entries[..self.len()].iter().map(|entry| {
            entry
                .as_ref()
                .expect("parsed artifact prefix is fully initialized")
        })
    }

    pub fn find(&self, kind: ArtifactKind, artifact_id: &ArtifactId) -> Option<&ArtifactDigest> {
        let initialized = &self.entries[..self.len()];
        initialized
            .binary_search_by(|entry| {
                let artifact = entry
                    .as_ref()
                    .expect("parsed artifact prefix is fully initialized");
                artifact
                    .kind
                    .cmp(&kind)
                    .then_with(|| artifact.artifact_id.cmp(artifact_id))
            })
            .ok()
            .and_then(|index| initialized[index].as_ref())
    }
}

fn require_count(
    kind: ArtifactKind,
    actual: usize,
    maximum: usize,
) -> Result<(), InventoryParseError> {
    if actual > maximum {
        Err(InventoryParseError::TooManyArtifacts {
            kind,
            actual,
            maximum,
        })
    } else {
        Ok(())
    }
}
