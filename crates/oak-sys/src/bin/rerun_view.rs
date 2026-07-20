use oak_sys::{Device, DeviceConfig, ImageFrame};
use rerun::Image;

fn imageframe_to_rerun(frame: ImageFrame) -> Image {
    let width = frame.width;
    let height = frame.height;
    Image::from_l8(frame.into_pixels(), [width, height])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "rerun_view".to_owned());
    let mxid = args.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("usage: {program} EXACT_MXID"),
        )
    })?;
    if mxid.trim().is_empty() || args.next().is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("usage: {program} EXACT_MXID"),
        )
        .into());
    }
    let mut camera = Device::connect(&mxid, DeviceConfig::all_streams())?;
    let rec = rerun::RecordingStreamBuilder::new("First Trial").spawn()?;

    loop {
        let stereo_l = imageframe_to_rerun(camera.mono_left(100)?);
        let stereo_r = imageframe_to_rerun(camera.mono_right(100)?);

        rec.log("camera/left", &stereo_l)?;
        rec.log("camera/right", &stereo_r)?;
    }
}
