use image::ImageReader;
use kiko_slam::{Frame, SensorId, SuperPoint, Timestamp};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let rec = rerun::RecordingStreamBuilder::new("kiko-slam").spawn()?;

    let img = ImageReader::open("img.jpg")?.decode()?;
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    let data = gray.into_raw();
    let image_data = img.to_rgb8().into_raw();
    let frame = Frame::new(SensorId::StereoLeft, Timestamp::from_nanos(0), w, h, data)?;

    let mut sp = SuperPoint::new("models/sp.onnx")?;
    let detections = sp.detect(&frame)?;

    rec.set_time_sequence("frame", 0);
    rec.log(
        "view/image",
        &rerun::Image::new(
            image_data,
            rerun::components::ImageFormat::rgb8([w, h]),
        ),
    )?;

    let points: Vec<[f32; 2]> = detections
        .keypoints()
        .iter()
        .map(|kp| [kp.x, kp.y])
        .collect();
    rec.log(
        "view/image/keypoints",
        &rerun::Points2D::new(points).with_radii([2.0]),
    )?;

    println!("Detected {} keypoints", detections.len());
    Ok(())
}
