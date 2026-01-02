use super::InferenceError;
use crate::Detections;
use crate::Matches;
use crate::Raw;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Arc;

pub struct LightGlue {
    session: Session,
}

impl LightGlue {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        let session = Session::builder()
            .map_err(|e| InferenceError::LoadFailed {
                path: path.to_path_buf(),
                source: e,
            })?
            .commit_from_file(path)
            .map_err(|e| InferenceError::LoadFailed {
                path: path.to_path_buf(),
                source: e,
            })?;

        Ok(Self { session })
    }

    pub fn match_these(
        &mut self,
        dec_1: Arc<Detections>,
        dec_2: Arc<Detections>,
    ) -> Result<Matches<Raw>, InferenceError> {
        let kpts_0: Vec<f32> = dec_1
            .keypoints()
            .iter()
            .flat_map(|kp| [kp.x, kp.y])
            .collect();

        let kpts_1: Vec<f32> = dec_2
            .keypoints()
            .iter()
            .flat_map(|kp| [kp.x, kp.y])
            .collect();

        let desc_0: Vec<f32> = dec_1.descriptors().iter().flat_map(|desc| desc.0).collect();
        let desc_1: Vec<f32> = dec_2.descriptors().iter().flat_map(|desc| desc.0).collect();

        let kpts_0_tensor = Tensor::from_array(([1, dec_1.len(), 2], kpts_0))?;
        let kpts_1_tensor = Tensor::from_array(([1, dec_2.len(), 2], kpts_1))?;
        let desc_0_tensor = Tensor::from_array(([1, dec_1.len(), 256], desc_0))?;
        let desc_1_tensor = Tensor::from_array(([1, dec_2.len(), 256], desc_1))?;

        let outputs = self.session.run(ort::inputs!["kpts0" => kpts_0_tensor, "kpts1" => kpts_1_tensor, "desc0" => desc_0_tensor, "desc1" => desc_1_tensor])?;
        let matches_raw = outputs["matches0"].try_extract_tensor::<i64>()?;
        let scores_raw = outputs["mscores0"].try_extract_tensor::<f32>()?;
        let matches_data = matches_raw.1;
        let scores_data = scores_raw.1;

        let mut indices = Vec::new();
        let mut scores = Vec::new();

        for (i, &match_idx) in matches_data.iter().enumerate() {
            if match_idx != -1 {
                indices.push((i, match_idx as usize));
                scores.push(scores_data[i]);
            }
        }

        Matches::new(dec_1, dec_2, indices, scores)
            .map_err(|e| InferenceError::Domain(format!("{e:?}")))
    }
}
