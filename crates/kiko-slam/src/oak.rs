use crate::{Frame, FrameId, SensorId, Timestamp};
use oak_sys::ImageFrame;

pub fn oak_to_frame(oak_frame: ImageFrame, sensor: SensorId, frame_id: FrameId) -> Frame {
    Frame::new(
        sensor,
        frame_id,
        Timestamp::from_nanos(oak_frame.timestamp.as_nanos()),
        oak_frame.width,
        oak_frame.height,
        oak_frame.into_pixels(),
    )
    .expect("oak frame dimensions should be valid")
}
