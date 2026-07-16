//! Reproducible, dependency-free benchmark for the public local descriptor matcher.
//!
//! Run the full sample set with `cargo bench -p kiko-slam --bench descriptor_matching`.
//! Set `KIKO_MATCH_BENCH_SHORT=1` for a one-iteration compile/smoke run.

use std::hint::black_box;
use std::time::Instant;

use kiko_slam::map::{ImageSize, KeyframeId, SlamMap};
use kiko_slam::{
    DESCRIPTOR_DIM, Descriptor, FrameId, Keypoint, Point3, Timestamp, WorldToCamera,
    try_match_descriptors_for_loop,
};

const THRESHOLD: f32 = 0.95;
const DEFAULT_SAMPLES: usize = 7;
const DEFAULT_WARMUP_ROUNDS: usize = 2;
const DEFAULT_LOGICAL_PAIRS_PER_SAMPLE: usize = 131_072;
const SHORT_SAMPLES: usize = 3;
const CHECKSUM_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const CHECKSUM_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy)]
enum Pattern {
    Distinct,
    Tie,
    Reject,
}

struct Case {
    name: &'static str,
    query_count: usize,
    candidate_count: usize,
    pattern: Pattern,
}

struct Fixture {
    query: Vec<Descriptor>,
    map: SlamMap,
    candidate_keyframe: KeyframeId,
    expected: Vec<(usize, usize)>,
}

fn basis_descriptor(index: usize) -> Descriptor {
    let mut values = [0.0; DESCRIPTOR_DIM];
    values[index] = 1.0;
    Descriptor(values)
}

fn expected_matches(case: &Case) -> Vec<(usize, usize)> {
    match case.pattern {
        Pattern::Distinct => (0..case.query_count.min(case.candidate_count))
            .map(|index| (index, index))
            .collect(),
        Pattern::Tie => vec![(case.query_count - 1, case.candidate_count - 1)],
        Pattern::Reject => Vec::new(),
    }
}

fn query_descriptor(pattern: Pattern, index: usize) -> Descriptor {
    match pattern {
        Pattern::Distinct => basis_descriptor(index),
        Pattern::Tie | Pattern::Reject => basis_descriptor(0),
    }
}

fn candidate_descriptor(pattern: Pattern, index: usize) -> Descriptor {
    match pattern {
        Pattern::Distinct => basis_descriptor(index),
        Pattern::Tie => basis_descriptor(0),
        Pattern::Reject => basis_descriptor(1),
    }
}

fn build_fixture(case: &Case) -> Fixture {
    assert!(case.query_count > 0, "benchmark queries must not be empty");
    assert!(
        case.candidate_count > 0,
        "benchmark candidates must not be empty"
    );
    if matches!(case.pattern, Pattern::Distinct) {
        assert!(
            case.query_count.max(case.candidate_count) <= DESCRIPTOR_DIM,
            "distinct basis fixtures cannot exceed the descriptor dimension"
        );
    }

    let query = (0..case.query_count)
        .map(|index| query_descriptor(case.pattern, index))
        .collect();
    let keypoints = (0..case.candidate_count)
        .map(|index| Keypoint {
            x: (index % 32) as f32 + 0.5,
            y: (index / 32) as f32 + 0.5,
        })
        .collect();

    let mut map = SlamMap::new();
    let image_size = ImageSize::try_new(640, 480).expect("fixed benchmark image size is valid");
    let candidate_keyframe = map
        .add_keyframe(
            FrameId::new(1),
            Timestamp::from_nanos(1),
            WorldToCamera::identity(),
            image_size,
            keypoints,
        )
        .expect("benchmark keyframe is valid");
    for index in 0..case.candidate_count {
        let keypoint = map
            .keyframe_keypoint(candidate_keyframe, index)
            .expect("benchmark keypoint exists");
        map.add_map_point(
            Point3 {
                x: index as f32 * 0.001,
                y: 0.0,
                z: 3.0,
            },
            candidate_descriptor(case.pattern, index).quantize(),
            keypoint,
        )
        .expect("benchmark map point is valid");
    }

    Fixture {
        query,
        map,
        candidate_keyframe,
        expected: expected_matches(case),
    }
}

fn checksum(matches: &[(usize, usize)]) -> u64 {
    let mut checksum = CHECKSUM_OFFSET;
    for value in std::iter::once(matches.len()).chain(
        matches
            .iter()
            .flat_map(|&(query, candidate)| [query, candidate]),
    ) {
        let value = u64::try_from(value).expect("match index must fit in the checksum encoding");
        for byte in value.to_le_bytes() {
            checksum ^= u64::from(byte);
            checksum = checksum.wrapping_mul(CHECKSUM_PRIME);
        }
    }
    checksum
}

fn match_once(fixture: &Fixture) -> Vec<(usize, usize)> {
    try_match_descriptors_for_loop(
        black_box(&fixture.query),
        black_box(fixture.candidate_keyframe),
        black_box(&fixture.map),
        black_box(THRESHOLD),
    )
    .expect("benchmark fixture must remain a valid map")
}

fn run_iterations(fixture: &Fixture, iterations: usize) -> usize {
    let mut match_count = 0_usize;
    for _ in 0..iterations {
        let matches = match_once(fixture);
        match_count = match_count
            .checked_add(black_box(matches.len()))
            .expect("benchmark match count cannot overflow");
        black_box(matches);
    }
    match_count
}

fn benchmark_case(case: &Case, short: bool) {
    let fixture = build_fixture(case);
    let actual = match_once(&fixture);
    assert_eq!(
        actual, fixture.expected,
        "{} fixture no longer exercises its intended behavior",
        case.name
    );
    let result_checksum = checksum(&actual);

    let logical_pairs = case
        .query_count
        .checked_mul(case.candidate_count)
        .expect("benchmark logical-pair count cannot overflow");
    let iterations = if short {
        1
    } else {
        DEFAULT_LOGICAL_PAIRS_PER_SAMPLE.div_ceil(logical_pairs)
    };
    let warmup_rounds = if short { 1 } else { DEFAULT_WARMUP_ROUNDS };
    let samples = if short {
        SHORT_SAMPLES
    } else {
        DEFAULT_SAMPLES
    };
    let expected_match_count = fixture
        .expected
        .len()
        .checked_mul(iterations)
        .expect("benchmark expected-match count cannot overflow");

    for _ in 0..warmup_rounds {
        assert_eq!(run_iterations(&fixture, iterations), expected_match_count);
    }

    let mut nanos_per_call = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        let observed_match_count = run_iterations(&fixture, iterations);
        let elapsed = start.elapsed();
        assert_eq!(observed_match_count, expected_match_count);
        nanos_per_call.push(elapsed.as_secs_f64() * 1.0e9 / iterations as f64);
    }
    nanos_per_call.sort_by(f64::total_cmp);
    let median_ns = nanos_per_call[samples / 2];
    assert!(
        median_ns > 0.0 && median_ns.is_finite(),
        "benchmark timer must produce a positive finite duration"
    );
    let logical_pairs_per_second = logical_pairs as f64 * 1.0e9 / median_ns;

    println!(
        "case={} queries={} candidates={} logical_pairs={} expected_matches={} checksum={result_checksum:#018x} warmup_rounds={warmup_rounds} samples={samples} iterations_per_sample={iterations} median_ns_per_call={median_ns:.1} logical_pairs_per_s={logical_pairs_per_second:.1}",
        case.name,
        case.query_count,
        case.candidate_count,
        logical_pairs,
        fixture.expected.len(),
    );
}

fn short_mode() -> bool {
    let Some(value) = std::env::var_os("KIKO_MATCH_BENCH_SHORT") else {
        return false;
    };
    let value = value
        .to_str()
        .expect("KIKO_MATCH_BENCH_SHORT must be valid UTF-8");
    match value {
        "1" | "true" => true,
        "0" | "false" => false,
        _ => panic!("KIKO_MATCH_BENCH_SHORT must be one of: 0, 1, false, true"),
    }
}

fn main() {
    let short = short_mode();
    println!(
        "descriptor matcher benchmark: short={short}; throughput counts query_count*candidate_count logical pairs per call"
    );

    let cases = [
        Case {
            name: "distinct_square",
            query_count: 128,
            candidate_count: 128,
            pattern: Pattern::Distinct,
        },
        Case {
            name: "distinct_candidate_heavy",
            query_count: 48,
            candidate_count: 192,
            pattern: Pattern::Distinct,
        },
        Case {
            name: "distinct_query_heavy",
            query_count: 192,
            candidate_count: 48,
            pattern: Pattern::Distinct,
        },
        Case {
            name: "tie_square",
            query_count: 96,
            candidate_count: 96,
            pattern: Pattern::Tie,
        },
        Case {
            name: "reject_square",
            query_count: 96,
            candidate_count: 96,
            pattern: Pattern::Reject,
        },
    ];

    for case in &cases {
        benchmark_case(case, short);
    }
}
