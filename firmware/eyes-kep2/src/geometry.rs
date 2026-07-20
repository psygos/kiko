//! Measured 56-LED circular panel geometry retained from the proven RP2350 demo.

/// Fixed-point position units per millimetre.
pub const UNITS_PER_MM: i32 = 16;
pub const DISC_RADIUS: i32 = 45 * UNITS_PER_MM;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub const fn squared_distance(self, other: Self) -> i64 {
        let dx = (self.x - other.x) as i64;
        let dy = (self.y - other.y) as i64;
        dx * dx + dy * dy
    }
}

/// Chain index to physical centre, in 1/16 mm from the panel centre.
///
/// The source was the measured backlight STL: a 90.2 mm disc, six
/// boustrophedon columns, and 56 LED wells. The ordering is hardware-facing and
/// must not be regenerated from an assumed rectangular grid.
pub const EYE_POSITIONS: [Position; 56] = [
    Position::new(-520, -278),
    Position::new(-520, -167),
    Position::new(-520, -56),
    Position::new(-520, 56),
    Position::new(-520, 167),
    Position::new(-520, 278),
    Position::new(-312, 500),
    Position::new(-312, 389),
    Position::new(-312, 278),
    Position::new(-312, 167),
    Position::new(-312, 56),
    Position::new(-312, -56),
    Position::new(-312, -167),
    Position::new(-312, -278),
    Position::new(-312, -389),
    Position::new(-312, -500),
    Position::new(-104, -611),
    Position::new(-104, -500),
    Position::new(-104, -389),
    Position::new(-104, -278),
    Position::new(-104, -167),
    Position::new(-104, -56),
    Position::new(-104, 56),
    Position::new(-104, 167),
    Position::new(-104, 278),
    Position::new(-104, 389),
    Position::new(-104, 500),
    Position::new(-104, 611),
    Position::new(104, 611),
    Position::new(104, 500),
    Position::new(104, 389),
    Position::new(104, 278),
    Position::new(104, 167),
    Position::new(104, 56),
    Position::new(104, -56),
    Position::new(104, -167),
    Position::new(104, -278),
    Position::new(104, -389),
    Position::new(104, -500),
    Position::new(104, -611),
    Position::new(312, -500),
    Position::new(312, -389),
    Position::new(312, -278),
    Position::new(312, -167),
    Position::new(312, -56),
    Position::new(312, 56),
    Position::new(312, 167),
    Position::new(312, 278),
    Position::new(312, 389),
    Position::new(312, 500),
    Position::new(520, 278),
    Position::new(520, 167),
    Position::new(520, 56),
    Position::new(520, -56),
    Position::new(520, -167),
    Position::new(520, -278),
];

const _: () = {
    let radius_squared = (DISC_RADIUS as i64) * (DISC_RADIUS as i64);
    let mut index = 0;
    while index < EYE_POSITIONS.len() {
        let point = EYE_POSITIONS[index];
        let squared = (point.x as i64) * (point.x as i64) + (point.y as i64) * (point.y as i64);
        assert!(
            squared <= radius_squared,
            "LED centre lies outside eye disc"
        );

        let mirror = EYE_POSITIONS[EYE_POSITIONS.len() - 1 - index];
        assert!(
            mirror.x == -point.x && mirror.y == point.y,
            "chain reversal is not an x reflection"
        );
        index += 1;
    }
};
