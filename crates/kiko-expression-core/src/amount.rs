//! Fixed-point normalized quantities used at the expression boundary.
//!
//! A scale of 10,000 gives deterministic cross-platform mixing without NaN,
//! infinity, or floating-point accumulation. Floating-point constructors are
//! provided only for parsing external values and reject invalid inputs.

use core::fmt;

const SCALE: i32 = 10_000;

/// A finite value in the inclusive interval `[0, 1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnitAmount(u16);

impl UnitAmount {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(SCALE as u16);

    pub fn try_from_f64(value: f64) -> Result<Self, AmountError> {
        if !value.is_finite() {
            return Err(AmountError::NonFinite { value });
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(AmountError::OutsideUnitInterval { value });
        }
        let scaled = value * f64::from(SCALE);
        Ok(Self((scaled + 0.5) as u16))
    }

    pub const fn try_from_basis_points(value: u16) -> Result<Self, AmountError> {
        if value <= SCALE as u16 {
            Ok(Self(value))
        } else {
            Err(AmountError::UnitBasisPointsOutOfRange { value })
        }
    }

    pub const fn basis_points(self) -> u16 {
        self.0
    }

    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / f64::from(SCALE)
    }

    pub(crate) const fn from_basis_points_proven(value: u16) -> Self {
        debug_assert!(value <= SCALE as u16);
        Self(value)
    }
}

/// A finite, non-zero value in the interval `(0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PositiveUnitAmount(UnitAmount);

impl PositiveUnitAmount {
    pub const ONE: Self = Self(UnitAmount::ONE);

    pub fn try_from_f64(value: f64) -> Result<Self, AmountError> {
        let amount = UnitAmount::try_from_f64(value)?;
        Self::try_from_unit(amount)
    }

    pub const fn try_from_basis_points(value: u16) -> Result<Self, AmountError> {
        match UnitAmount::try_from_basis_points(value) {
            Ok(amount) => Self::try_from_unit(amount),
            Err(error) => Err(error),
        }
    }

    pub const fn try_from_unit(value: UnitAmount) -> Result<Self, AmountError> {
        if value.basis_points() == 0 {
            Err(AmountError::ZeroNotPositive)
        } else {
            Ok(Self(value))
        }
    }

    pub const fn basis_points(self) -> u16 {
        self.0.basis_points()
    }

    pub fn as_f64(self) -> f64 {
        self.0.as_f64()
    }

    pub const fn as_unit(self) -> UnitAmount {
        self.0
    }
}

/// A finite value in the inclusive interval `[-1, 1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignedUnitAmount(i16);

impl SignedUnitAmount {
    pub const MIN: Self = Self(-(SCALE as i16));
    pub const ZERO: Self = Self(0);
    pub const MAX: Self = Self(SCALE as i16);

    pub fn try_from_f64(value: f64) -> Result<Self, AmountError> {
        if !value.is_finite() {
            return Err(AmountError::NonFinite { value });
        }
        if !(-1.0..=1.0).contains(&value) {
            return Err(AmountError::OutsideSignedUnitInterval { value });
        }
        let scaled = value * f64::from(SCALE);
        let rounded = if scaled >= 0.0 {
            scaled + 0.5
        } else {
            scaled - 0.5
        };
        Ok(Self(rounded as i16))
    }

    pub const fn try_from_basis_points(value: i16) -> Result<Self, AmountError> {
        if value >= -(SCALE as i16) && value <= SCALE as i16 {
            Ok(Self(value))
        } else {
            Err(AmountError::SignedBasisPointsOutOfRange { value })
        }
    }

    pub const fn basis_points(self) -> i16 {
        self.0
    }

    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / f64::from(SCALE)
    }

    pub(crate) fn scaled_by(self, gain: UnitAmount) -> Self {
        let product = i32::from(self.0) * i32::from(gain.basis_points());
        let rounded = if product >= 0 {
            (product + SCALE / 2) / SCALE
        } else {
            (product - SCALE / 2) / SCALE
        };
        Self(rounded as i16)
    }

    pub(crate) const fn from_basis_points_proven(value: i16) -> Self {
        debug_assert!(value >= -(SCALE as i16) && value <= SCALE as i16);
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AmountError {
    NonFinite { value: f64 },
    OutsideUnitInterval { value: f64 },
    OutsideSignedUnitInterval { value: f64 },
    UnitBasisPointsOutOfRange { value: u16 },
    SignedBasisPointsOutOfRange { value: i16 },
    ZeroNotPositive,
}

impl fmt::Display for AmountError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { value } => write!(formatter, "amount must be finite, got {value}"),
            Self::OutsideUnitInterval { value } => {
                write!(formatter, "amount must be within [0, 1], got {value}")
            }
            Self::OutsideSignedUnitInterval { value } => {
                write!(formatter, "amount must be within [-1, 1], got {value}")
            }
            Self::UnitBasisPointsOutOfRange { value } => write!(
                formatter,
                "unit amount must be at most {SCALE} basis points, got {value}"
            ),
            Self::SignedBasisPointsOutOfRange { value } => write!(
                formatter,
                "signed amount must be within -{SCALE}..={SCALE} basis points, got {value}"
            ),
            Self::ZeroNotPositive => formatter.write_str("positive unit amount must be non-zero"),
        }
    }
}

impl core::error::Error for AmountError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floating_boundaries_are_checked_before_quantization() {
        assert!(UnitAmount::try_from_f64(-f64::EPSILON).is_err());
        assert!(UnitAmount::try_from_f64(1.0 + f64::EPSILON).is_err());
        assert!(SignedUnitAmount::try_from_f64(-1.0 - f64::EPSILON).is_err());
        assert!(SignedUnitAmount::try_from_f64(f64::NAN).is_err());
        assert!(PositiveUnitAmount::try_from_f64(0.0).is_err());
    }

    #[test]
    fn every_admitted_signed_basis_point_round_trips() {
        for raw in -10_000_i16..=10_000_i16 {
            let value = SignedUnitAmount::try_from_basis_points(raw).unwrap();
            assert_eq!(value.basis_points(), raw);
        }
    }

    #[test]
    fn scaling_never_escapes_signed_bounds() {
        for raw in (-10_000_i16..=10_000_i16).step_by(37) {
            let value = SignedUnitAmount::try_from_basis_points(raw).unwrap();
            for gain in (0_u16..=10_000_u16).step_by(43) {
                let gain = UnitAmount::try_from_basis_points(gain).unwrap();
                let scaled = value.scaled_by(gain).basis_points();
                assert!((-10_000..=10_000).contains(&scaled));
                assert!(i32::from(scaled.abs()) <= i32::from(raw.abs()));
            }
        }
    }
}
