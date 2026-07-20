//! Text boundary for a map-frame navigation goal.
//!
//! Viewer, CLI, and replay adapters should parse their weak input once and
//! hand the resulting [`NavigationGoalArg`] to the navigation coordinator.

use std::num::ParseFloatError;
use std::str::FromStr;

use super::{MapPoint, PlanarPointError, RecordedMapEpochId, RecordedMapEpochIdError};

/// A finite map-frame goal parsed from the exact text form `X_M,Y_M`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavigationGoalArg(MapPoint);

impl NavigationGoalArg {
    pub fn point(self) -> MapPoint {
        self.0
    }
}

/// Weak boundary object for a viewer or control API map click.
///
/// Both map fields are mandatory: a coordinate alone is ambiguous after a map
/// reset or while a newer occupancy revision is displayed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapPointGoalSelectionDto {
    pub map_epoch_id: u64,
    pub displayed_revision: u64,
    pub x_m: f64,
    pub y_m: f64,
}

/// One finite map point bound to the exact map view from which it was chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapPointGoalSelection {
    map_epoch_id: RecordedMapEpochId,
    displayed_revision: u64,
    point: MapPoint,
}

impl MapPointGoalSelection {
    pub fn parse(dto: MapPointGoalSelectionDto) -> Result<Self, MapPointGoalSelectionParseError> {
        let map_epoch_id = RecordedMapEpochId::try_new(dto.map_epoch_id)
            .map_err(MapPointGoalSelectionParseError::MapEpochId)?;
        let point =
            MapPoint::try_new(dto.x_m, dto.y_m).map_err(MapPointGoalSelectionParseError::Point)?;
        Ok(Self {
            map_epoch_id,
            displayed_revision: dto.displayed_revision,
            point,
        })
    }

    pub fn map_epoch_id(self) -> RecordedMapEpochId {
        self.map_epoch_id
    }

    pub fn displayed_revision(self) -> u64 {
        self.displayed_revision
    }

    pub fn point(self) -> MapPoint {
        self.point
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapPointGoalSelectionParseError {
    MapEpochId(RecordedMapEpochIdError),
    Point(PlanarPointError),
}

impl std::fmt::Display for MapPointGoalSelectionParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid map-point goal selection: {self:?}")
    }
}

impl std::error::Error for MapPointGoalSelectionParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MapEpochId(source) => Some(source),
            Self::Point(source) => Some(source),
        }
    }
}

impl FromStr for NavigationGoalArg {
    type Err = NavigationGoalArgError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        if raw.is_empty() {
            return Err(NavigationGoalArgError::Empty);
        }
        let mut fields = raw.split(',');
        let x = fields.next().expect("nonempty split has a first field");
        let Some(y) = fields.next() else {
            return Err(NavigationGoalArgError::FieldCount);
        };
        if fields.next().is_some() {
            return Err(NavigationGoalArgError::FieldCount);
        }
        if x.is_empty() {
            return Err(NavigationGoalArgError::EmptyCoordinate {
                axis: NavigationGoalAxis::X,
            });
        }
        if y.is_empty() {
            return Err(NavigationGoalArgError::EmptyCoordinate {
                axis: NavigationGoalAxis::Y,
            });
        }
        let x_m = x
            .parse::<f64>()
            .map_err(|source| NavigationGoalArgError::InvalidNumber {
                axis: NavigationGoalAxis::X,
                source,
            })?;
        let y_m = y
            .parse::<f64>()
            .map_err(|source| NavigationGoalArgError::InvalidNumber {
                axis: NavigationGoalAxis::Y,
                source,
            })?;
        MapPoint::try_new(x_m, y_m)
            .map(Self)
            .map_err(NavigationGoalArgError::Domain)
    }
}

impl std::fmt::Display for NavigationGoalArg {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{},{}", self.0.x_m(), self.0.y_m())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavigationGoalAxis {
    X,
    Y,
}

#[derive(Debug)]
pub enum NavigationGoalArgError {
    Empty,
    FieldCount,
    EmptyCoordinate {
        axis: NavigationGoalAxis,
    },
    InvalidNumber {
        axis: NavigationGoalAxis,
        source: ParseFloatError,
    },
    Domain(PlanarPointError),
}

impl std::fmt::Display for NavigationGoalArgError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => formatter.write_str("navigation goal must be X_M,Y_M"),
            Self::FieldCount => formatter
                .write_str("navigation goal must contain exactly two comma-separated values"),
            Self::EmptyCoordinate { axis } => {
                write!(formatter, "navigation goal {axis:?} coordinate is empty")
            }
            Self::InvalidNumber { axis, source } => {
                write!(
                    formatter,
                    "navigation goal {axis:?} coordinate is not a bare number: {source}"
                )
            }
            Self::Domain(source) => write!(formatter, "invalid navigation goal: {source}"),
        }
    }
}

impl std::error::Error for NavigationGoalArgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidNumber { source, .. } => Some(source),
            Self::Domain(source) => Some(source),
            Self::Empty | Self::FieldCount | Self::EmptyCoordinate { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exact_finite_si_coordinates_once() {
        let goal = "-1.25,2.5"
            .parse::<NavigationGoalArg>()
            .expect("finite map goal");
        assert_eq!(goal.point().as_array(), [-1.25, 2.5]);
        assert_eq!(goal.to_string(), "-1.25,2.5");
    }

    #[test]
    fn rejects_missing_extra_empty_nonfinite_units_and_whitespace() {
        for raw in [
            "", "1", "1,2,3", ",2", "1,", "NaN,0", "0,inf", "1m,2", "1,2m", " 1,2", "1, 2",
        ] {
            assert!(
                raw.parse::<NavigationGoalArg>().is_err(),
                "weak goal {raw:?} must be rejected"
            );
        }
    }

    #[test]
    fn finite_coordinate_samples_round_trip_through_the_domain() {
        for (x_m, y_m) in [
            (0.0, 0.0),
            (f64::MIN_POSITIVE, -f64::MIN_POSITIVE),
            (1.0e-200, -1.0e-200),
            (1.0e6, -1.0e6),
        ] {
            let raw = format!("{x_m},{y_m}");
            let parsed = raw
                .parse::<NavigationGoalArg>()
                .expect("finite coordinate sample");
            assert_eq!(parsed.point().x_m().to_bits(), x_m.to_bits());
            assert_eq!(parsed.point().y_m().to_bits(), y_m.to_bits());
        }
    }

    #[test]
    fn viewer_selection_parses_epoch_revision_and_si_point_once() {
        let selection = MapPointGoalSelection::parse(MapPointGoalSelectionDto {
            map_epoch_id: 7,
            displayed_revision: 42,
            x_m: -1.25,
            y_m: 2.5,
        })
        .expect("typed viewer selection");
        assert_eq!(selection.map_epoch_id().as_u64(), 7);
        assert_eq!(selection.displayed_revision(), 42);
        assert_eq!(selection.point().as_array(), [-1.25, 2.5]);
    }

    #[test]
    fn viewer_selection_rejects_zero_epoch_and_nonfinite_coordinates() {
        for dto in [
            MapPointGoalSelectionDto {
                map_epoch_id: 0,
                displayed_revision: 1,
                x_m: 0.0,
                y_m: 0.0,
            },
            MapPointGoalSelectionDto {
                map_epoch_id: 1,
                displayed_revision: 1,
                x_m: f64::NAN,
                y_m: 0.0,
            },
            MapPointGoalSelectionDto {
                map_epoch_id: 1,
                displayed_revision: 1,
                x_m: 0.0,
                y_m: f64::INFINITY,
            },
        ] {
            assert!(MapPointGoalSelection::parse(dto).is_err());
        }
    }
}
