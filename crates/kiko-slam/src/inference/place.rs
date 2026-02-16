use crate::Frame;
use crate::loop_closure::GlobalDescriptor;

use super::InferenceError;

pub trait PlaceDescriptorExtractor: Send {
    fn backend_name(&self) -> &'static str;
    fn compute_descriptor(&mut self, frame: &Frame) -> Result<GlobalDescriptor, InferenceError>;
}
