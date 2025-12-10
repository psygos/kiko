use oak_sys::{Device, DeviceConfig, ImageError, ImageFrame, Timestamp};
use rerun::{ChannelDatatype, ColorModel, Image, ImageFormat, RecordingStreamBuilder};

fn imageframe_to_rerun(frame: ImageFrame) -> Image {
    let width = frame.width;
    let height = frame.height;
    Image::from_l8(frame.into_pixels(), [width, height])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut camera = Device::connect("", DeviceConfig::all_streams())?;
    let rec = rerun::RecordingStreamBuilder::new("First Trial").spawn()?;

    loop {
        let stereo_l = imageframe_to_rerun(camera.mono_left(100)?);
        let stereo_r = imageframe_to_rerun(camera.mono_right(100)?);

        rec.log("camera/left", &stereo_l)?;
        rec.log("camera/right", &stereo_r)?;
    }
}
