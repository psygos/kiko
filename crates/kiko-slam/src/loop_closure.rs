use crate::map::KeyframeId;

#[derive(Clone, Debug)]
pub struct GlobalDescriptor(pub [f32; 512]);

impl GlobalDescriptor {
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;
        for i in 0..512 {
            let a = self.0[i] as f64;
            let b = other.0[i] as f64;
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        if norm_a <= 0.0 || norm_b <= 0.0 {
            return 0.0;
        }
        (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
    }
}

#[derive(Clone, Debug)]
pub struct PlaceMatch {
    pub query: KeyframeId,
    pub candidate: KeyframeId,
    pub similarity: f32,
}

#[derive(Clone, Debug)]
pub struct KeyframeDatabase {
    entries: Vec<(KeyframeId, GlobalDescriptor)>,
    temporal_gap: usize,
}

impl KeyframeDatabase {
    pub fn new(temporal_gap: usize) -> Self {
        Self {
            entries: Vec::new(),
            temporal_gap,
        }
    }

    pub fn temporal_gap(&self) -> usize {
        self.temporal_gap
    }

    pub fn insert(&mut self, id: KeyframeId, descriptor: GlobalDescriptor) {
        self.entries.push((id, descriptor));
    }

    pub fn query(&self, descriptor: &GlobalDescriptor, top_k: usize) -> Vec<PlaceMatch> {
        if top_k == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let query_idx = self.entries.len() - 1;
        let query_id = self.entries[query_idx].0;
        let mut matches = Vec::new();
        for (idx, (candidate, candidate_desc)) in self.entries.iter().enumerate() {
            if idx == query_idx {
                continue;
            }
            if query_idx.saturating_sub(idx) <= self.temporal_gap {
                continue;
            }
            matches.push(PlaceMatch {
                query: query_id,
                candidate: *candidate,
                similarity: descriptor.cosine_similarity(candidate_desc),
            });
        }
        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(top_k);
        matches
    }
}

impl Default for KeyframeDatabase {
    fn default() -> Self {
        Self::new(30)
    }
}

#[cfg(test)]
mod tests {
    use super::{GlobalDescriptor, KeyframeDatabase};
    use crate::map::{ImageSize, KeyframeId, SlamMap};
    use crate::{FrameId, Keypoint, Pose, Timestamp};

    fn descriptor_with_basis(idx: usize) -> GlobalDescriptor {
        let mut d = [0.0_f32; 512];
        d[idx] = 1.0;
        GlobalDescriptor(d)
    }

    fn make_keyframe_ids(n: usize) -> Vec<KeyframeId> {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(320, 240).expect("size");
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let id = map
                .add_keyframe(
                    FrameId::new((i + 1) as u64),
                    Timestamp::from_nanos(i as i64 + 1),
                    Pose::identity(),
                    size,
                    vec![Keypoint { x: 10.0, y: 10.0 }],
                )
                .expect("keyframe");
            ids.push(id);
        }
        ids
    }

    #[test]
    fn global_descriptor_identical_similarity_is_one() {
        let d = descriptor_with_basis(3);
        let sim = d.cosine_similarity(&d);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn keyframe_database_temporal_gap_filters_recent_frames() {
        let ids = make_keyframe_ids(5);
        let mut db = KeyframeDatabase::new(2);
        for (i, id) in ids.iter().enumerate() {
            db.insert(*id, descriptor_with_basis(i));
        }

        let matches = db.query(&descriptor_with_basis(0), 10);
        // Query is the latest keyframe; with gap=2, only the first two are eligible.
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.candidate == ids[0] || m.candidate == ids[1]));
    }

    #[test]
    fn keyframe_database_returns_top_k_by_similarity() {
        let ids = make_keyframe_ids(4);
        let mut db = KeyframeDatabase::new(0);

        let mut q = [0.0_f32; 512];
        q[0] = 1.0;
        q[1] = 1.0;
        let query = GlobalDescriptor(q);

        db.insert(ids[0], descriptor_with_basis(0)); // sim ~= 0.707
        db.insert(ids[1], descriptor_with_basis(1)); // sim ~= 0.707
        db.insert(ids[2], descriptor_with_basis(2)); // sim = 0
        db.insert(ids[3], query.clone()); // query entry

        let matches = db.query(&query, 2);
        assert_eq!(matches.len(), 2);
        assert!(matches[0].similarity >= matches[1].similarity);
        assert!(matches.iter().all(|m| m.candidate != ids[2]));
    }
}
