use image::ImageReader;
use kiko_slam::{Detections, Frame, FrameId, LightGlue, SensorId, SuperPoint, Timestamp};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAX_KEYPOINTS: usize = 1024;

fn repo_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
}

fn limit_detections(detections: Detections, max: usize) -> Detections {
    detections.top_k(max)
}

fn stitch_rgb(
    left: &[u8],
    left_width: u32,
    left_height: u32,
    right: &[u8],
    right_width: u32,
    right_height: u32,
) -> (Vec<u8>, u32, u32) {
    let out_width = left_width + right_width;
    let out_height = left_height.max(right_height);
    let out_width_usize = out_width as usize;
    let out_height_usize = out_height as usize;
    let left_width_usize = left_width as usize;
    let left_height_usize = left_height as usize;
    let right_width_usize = right_width as usize;
    let right_height_usize = right_height as usize;

    let mut out = vec![0u8; out_width_usize * out_height_usize * 3];

    for y in 0..out_height_usize {
        let out_row = y * out_width_usize * 3;
        if y < left_height_usize {
            let left_row = y * left_width_usize * 3;
            out[out_row..out_row + left_width_usize * 3]
                .copy_from_slice(&left[left_row..left_row + left_width_usize * 3]);
        }
        if y < right_height_usize {
            let right_row = y * right_width_usize * 3;
            let out_right = out_row + left_width_usize * 3;
            out[out_right..out_right + right_width_usize * 3]
                .copy_from_slice(&right[right_row..right_row + right_width_usize * 3]);
        }
    }

    (out, out_width, out_height)
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let root = repo_root();
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("target"));
    std::fs::create_dir_all(&target_dir)?;
    let output_path = target_dir.join("kiko-slam-matches.rrd");
    let rec = rerun::RecordingStreamBuilder::new("kiko-slam").save(&output_path)?;
    let img_1 = ImageReader::open(root.join("dataset/static/match/frame_1.jpeg"))?
        .decode()?
        .rotate90();
    let img_2 = ImageReader::open(root.join("dataset/static/match/frame_2.jpeg"))?
        .decode()?
        .rotate90();
    let gray_1 = img_1.to_luma8();
    let gray_2 = img_2.to_luma8();

    let (w_1, h_1) = gray_1.dimensions();
    let (w_2, h_2) = gray_2.dimensions();
    let data_1 = gray_1.into_raw();
    let data_2 = gray_2.into_raw();
    let image_data_1 = img_1.to_rgb8().into_raw();
    let image_data_2 = img_2.to_rgb8().into_raw();

    let frame_1 = Frame::new(
        SensorId::StereoLeft,
        FrameId::new(0),
        Timestamp::from_nanos(0),
        w_1,
        h_1,
        data_1,
    )?;
    let frame_2 = Frame::new(
        SensorId::StereoLeft,
        FrameId::new(1),
        Timestamp::from_nanos(0),
        w_2,
        h_2,
        data_2,
    )?;

    let mut sp = SuperPoint::new(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/sp.onnx"),
    )?;
    let detections_1 = Arc::new(limit_detections(sp.detect(&frame_1)?, MAX_KEYPOINTS));
    let detections_2 = Arc::new(limit_detections(sp.detect(&frame_2)?, MAX_KEYPOINTS));

    let mut lg = LightGlue::new(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/lg.onnx"),
    )?;
    let matches = lg.match_these(detections_1.clone(), detections_2.clone())?;

    let (stitched, stitched_w, stitched_h) =
        stitch_rgb(&image_data_1, w_1, h_1, &image_data_2, w_2, h_2);

    rec.set_time_sequence("frame", 0);
    rec.log(
        "view/matches",
        &rerun::Image::new(
            stitched,
            rerun::components::ImageFormat::rgb8([stitched_w, stitched_h]),
        )
        .with_draw_order(0.0),
    )?;

    let offset_x = w_1 as f32;
    let keypoints_1 = detections_1.keypoints();
    let keypoints_2 = detections_2.keypoints();
    let mut strips = Vec::with_capacity(matches.len());

    for &(idx_1, idx_2) in matches.indices() {
        if let (Some(kp_1), Some(kp_2)) = (keypoints_1.get(idx_1), keypoints_2.get(idx_2)) {
            strips.push(vec![[kp_1.x, kp_1.y], [kp_2.x + offset_x, kp_2.y]]);
        }
    }

    if !strips.is_empty() {
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
        let colors: Vec<rerun::Color> = (0..strips.len())
            .map(|idx| palette[idx % palette.len()])
            .collect();

        rec.log(
            "view/matches/lines",
            &rerun::LineStrips2D::new(strips)
                .with_colors(colors)
                .with_radii([rerun::Radius::new_ui_points(1.5)])
                .with_draw_order(10.0),
        )?;
    }

    println!(
        "Detected {} and {} keypoints; {} matches (saved to {})",
        detections_1.len(),
        detections_2.len(),
        matches.len(),
        output_path.display()
    );
    Ok(())
}
