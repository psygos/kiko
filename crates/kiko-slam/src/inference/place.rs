use crate::loop_closure::GlobalDescriptor;
use crate::Frame;

use super::InferenceError;

pub trait PlaceDescriptorExtractor: Send {
    fn compute_descriptor(&mut self, frame: &Frame) -> Result<GlobalDescriptor, InferenceError>;
}
