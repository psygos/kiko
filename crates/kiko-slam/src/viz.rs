use std::num::NonZeroUsize;

use crate::{Detections, Frame, Keypoint, Raw, VizPacket};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VizDecimation(NonZeroUsize);

impl VizDecimation {
    pub fn every_n(n: NonZeroUsize) -> Self {
        Self(n)
    }

    pub fn get(self) -> usize {
        self.0.get()
    }

    fn should_log(self, index: u64) -> bool {
        let n = self.0.get() as u64;
        index % n == 0
    }
}

impl Default for VizDecimation {
    fn default() -> Self {
        Self(NonZeroUsize::new(1).expect("1 is non-zero"))
    }
}

#[derive(Debug)]
pub enum VizDecimationError {
    Zero,
}

impl std::fmt::Display for VizDecimationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizDecimationError::Zero => write!(f, "decimation must be > 0"),
        }
    }
}

impl std::error::Error for VizDecimationError {}

impl TryFrom<usize> for VizDecimation {
    type Error = VizDecimationError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::new(value)
            .map(VizDecimation)
            .ok_or(VizDecimationError::Zero)
    }
}

#[derive(Debug)]
pub enum VizLogError {
    Rerun(rerun::RecordingStreamError),
}

impl std::fmt::Display for VizLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizLogError::Rerun(err) => write!(f, "rerun logging error: {err}"),
        }
    }
}

impl std::error::Error for VizLogError {}

impl From<rerun::RecordingStreamError> for VizLogError {
    fn from(err: rerun::RecordingStreamError) -> Self {
        VizLogError::Rerun(err)
    }
}

#[derive(Debug)]
pub struct RerunSink {
    rec: rerun::RecordingStream,
    decimation: VizDecimation,
    frame_index: u64,
    tracks: TrackState,
}

impl RerunSink {
    pub fn new(rec: rerun::RecordingStream, decimation: VizDecimation) -> Self {
        Self {
            rec,
            decimation,
            frame_index: 0,
            tracks: TrackState::new(),
        }
    }

    pub fn log(&mut self, packet: &VizPacket<Raw>) -> Result<(), VizLogError> {
        let index = self.frame_index;
        self.frame_index = self.frame_index.saturating_add(1);
        if !self.decimation.should_log(index) {
            return Ok(());
        }

        let left = packet.left();
        let right = packet.right();
        let track_ids = self.tracks.assign_tracks(packet.matches().source_a());

        self.rec.set_time(
            "capture_ns",
            rerun::TimeCell::from_duration_nanos(left.timestamp().as_nanos()),
        );

        let left_image = rerun::Image::from_color_model_and_bytes(
            left.data().to_vec(),
            [left.width(), left.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/left", &left_image)?;

        let right_image = rerun::Image::from_color_model_and_bytes(
            right.data().to_vec(),
            [right.width(), right.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/right", &right_image)?;

        let (stitched, width, height) = stitch_luma(left, right);
        let matches_image = rerun::Image::from_color_model_and_bytes(
            stitched,
            [width, height],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/matches", &matches_image)?;

        log_matches(&self.rec, packet, left.width() as f32, &track_ids)?;

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct TrackConfig {
    max_distance_px: f32,
    min_similarity: f32,
}

impl TrackConfig {
    fn load() -> Self {
        let max_distance_px = env_f32("KIKO_TRACK_MAX_DIST").unwrap_or(24.0);
        let min_similarity = env_f32("KIKO_TRACK_MIN_SIM").unwrap_or(0.8);
        Self {
            max_distance_px,
            min_similarity,
        }
    }
}

#[derive(Debug)]
struct TrackState {
    config: TrackConfig,
    prev_left: Option<std::sync::Arc<Detections>>,
    prev_track_ids: Vec<u64>,
    next_track_id: u64,
}

impl TrackState {
    fn new() -> Self {
        Self {
            config: TrackConfig::load(),
            prev_left: None,
            prev_track_ids: Vec::new(),
            next_track_id: 0,
        }
    }

    fn assign_tracks(&mut self, left: &Detections) -> Vec<u64> {
        let count = left.len();
        let mut track_ids = vec![0u64; count];

        let (prev_left, prev_ids) = match self.prev_left.as_ref() {
            Some(prev) if self.prev_track_ids.len() == prev.len() => (Some(prev), &self.prev_track_ids),
            _ => (None, &self.prev_track_ids),
        };

        if let Some(prev) = prev_left {
            let mut used_prev = vec![false; prev.len()];
            let max_dist_sq = self.config.max_distance_px * self.config.max_distance_px;

            for (i, desc) in left.descriptors().iter().enumerate() {
                let kp = left.keypoints()[i];
                let mut best_idx = None;
                let mut best_sim = self.config.min_similarity;

                for (j, prev_desc) in prev.descriptors().iter().enumerate() {
                    if used_prev[j] {
                        continue;
                    }
                    let prev_kp = prev.keypoints()[j];
                    if distance_sq(kp, prev_kp) > max_dist_sq {
                        continue;
                    }
                    let sim = dot(desc.0.as_slice(), prev_desc.0.as_slice());
                    if sim > best_sim {
                        best_sim = sim;
                        best_idx = Some(j);
                    }
                }

                if let Some(j) = best_idx {
                    track_ids[i] = prev_ids[j];
                    used_prev[j] = true;
                } else {
                    track_ids[i] = self.next_track_id;
                    self.next_track_id = self.next_track_id.saturating_add(1);
                }
            }
        } else {
            for id in &mut track_ids {
                *id = self.next_track_id;
                self.next_track_id = self.next_track_id.saturating_add(1);
            }
        }

        self.prev_left = Some(std::sync::Arc::new(left.clone()));
        self.prev_track_ids = track_ids.clone();

        track_ids
    }
}

fn env_f32(key: &str) -> Option<f32> {
    let raw = std::env::var(key).ok()?;
    match raw.parse::<f32>() {
        Ok(value) => Some(value),
        Err(_) => {
            eprintln!("invalid {key}={raw}, ignoring");
            None
        }
    }
}

fn log_matches(
    rec: &rerun::RecordingStream,
    packet: &VizPacket<Raw>,
    x_offset: f32,
    track_ids: &[u64],
) -> Result<(), rerun::RecordingStreamError> {
    let matches = packet.matches();
    if matches.is_empty() {
        return Ok(());
    }

    let keypoints_left = matches.source_a().keypoints();
    let keypoints_right = matches.source_b().keypoints();
    let mut strips = Vec::with_capacity(matches.len());

    for &(idx_left, idx_right) in matches.indices() {
        if let (Some(kp_left), Some(kp_right)) =
            (keypoints_left.get(idx_left), keypoints_right.get(idx_right))
        {
            strips.push(vec![
                [kp_left.x, kp_left.y],
                [kp_right.x + x_offset, kp_right.y],
            ]);
        }
    }

    if strips.is_empty() {
        return Ok(());
    }

    let palette = [
        rerun::Color::from_rgb(230, 57, 70),
        rerun::Color::from_rgb(241, 250, 238),
        rerun::Color::from_rgb(168, 218, 220),
        rerun::Color::from_rgb(69, 123, 157),
        rerun::Color::from_rgb(29, 53, 87),
        rerun::Color::from_rgb(255, 183, 3),
        rerun::Color::from_rgb(251, 86, 7),
        rerun::Color::from_rgb(131, 56, 236),
    ];
    let colors: Vec<rerun::Color> = matches
        .indices()
        .iter()
        .map(|(idx_left, _)| {
            let track_id = track_ids.get(*idx_left).copied().unwrap_or(0);
            palette[(track_id as usize) % palette.len()]
        })
        .collect();

    rec.log(
        "view/matches",
        &rerun::LineStrips2D::new(strips)
            .with_colors(colors)
            .with_radii([rerun::Radius::new_ui_points(1.5)])
            .with_draw_order(10.0),
    )?;

    Ok(())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn distance_sq(a: Keypoint, b: Keypoint) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

fn stitch_luma(left: &Frame, right: &Frame) -> (Vec<u8>, u32, u32) {
    let left_width = left.width();
    let left_height = left.height();
    let right_width = right.width();
    let right_height = right.height();

    let out_width = left_width + right_width;
    let out_height = left_height.max(right_height);

    let out_width_usize = out_width as usize;
    let out_height_usize = out_height as usize;
    let left_width_usize = left_width as usize;
    let left_height_usize = left_height as usize;
    let right_width_usize = right_width as usize;
    let right_height_usize = right_height as usize;

    let mut out = vec![0u8; out_width_usize * out_height_usize];

    let left_data = left.data();
    let right_data = right.data();

    for y in 0..out_height_usize {
        let out_row = y * out_width_usize;
        if y < left_height_usize {
            let left_row = y * left_width_usize;
            out[out_row..out_row + left_width_usize]
                .copy_from_slice(&left_data[left_row..left_row + left_width_usize]);
        }
        if y < right_height_usize {
            let right_row = y * right_width_usize;
            let out_right = out_row + left_width_usize;
            out[out_right..out_right + right_width_usize]
                .copy_from_slice(&right_data[right_row..right_row + right_width_usize]);
        }
    }

    (out, out_width, out_height)
}
