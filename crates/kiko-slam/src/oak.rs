use crate::{DepthImage, DepthImageError, Frame, FrameError, FrameId, SensorId, Timestamp};
use oak_sys::{DepthFrame, ImageFrame};

pub fn oak_to_frame(
    oak_frame: ImageFrame,
    sensor: SensorId,
    frame_id: FrameId,
) -> Result<Frame, FrameError> {
    Frame::new(
        sensor,
        frame_id,
        Timestamp::from_nanos(oak_frame.timestamp.as_nanos()),
        oak_frame.width,
        oak_frame.height,
        oak_frame.into_pixels(),
    )
}

pub fn oak_to_depth_image(oak_frame: DepthFrame) -> Result<DepthImage, DepthImageError> {
    let frame_id = FrameId::new(oak_frame.sequence);
    let timestamp = Timestamp::from_nanos(oak_frame.timestamp.as_nanos());
    let width = oak_frame.width;
    let height = oak_frame.height;
    let depth_mm = oak_frame.into_depth_mm();
    DepthImage::from_depth_mm(frame_id, timestamp, width, height, depth_mm)
}
