//! Serde coverage for the config types (`--features serde`): round-trip and
//! partial deserialize backed by the `Default` impls.

#![cfg(feature = "serde")]

use msg_level_select::{DesiredTraversals, LevelMapConfig, LevelMapPolicy};

#[test]
fn level_map_config_round_trips_through_ron() {
    let original = LevelMapConfig {
        layout: vec![1, 2, 3, 1],
        poisson_radius: 55.0,
        stage_buffer: 7,
        aspect_ratio: 4.0 / 3.0,
        node_position_buffer: 0.25,
        allow_extra_traversal: 2,
        desired_traversals: Some(DesiredTraversals {
            average: 3,
            ..DesiredTraversals::default()
        }),
        seed: Some(1234),
        policy: LevelMapPolicy {
            allow_revisit: true,
            allow_teleport: false,
            allow_path_visit: false,
        },
        max_attempts: 12,
    };

    let text = ron::ser::to_string(&original).expect("LevelMapConfig serializes");
    let parsed: LevelMapConfig = ron::from_str(&text).expect("serialized config parses");

    assert_eq!(parsed.layout, vec![1, 2, 3, 1]);
    assert_eq!(parsed.poisson_radius, 55.0);
    assert_eq!(parsed.stage_buffer, 7);
    assert_eq!(parsed.aspect_ratio, 4.0 / 3.0);
    assert_eq!(parsed.node_position_buffer, 0.25);
    assert_eq!(parsed.allow_extra_traversal, 2);
    assert_eq!(
        parsed.desired_traversals.as_ref().map(|d| d.average),
        Some(3)
    );
    assert_eq!(parsed.seed, Some(1234));
    assert!(parsed.policy.allow_revisit);
    assert!(!parsed.policy.allow_teleport);
    assert!(!parsed.policy.allow_path_visit);
    assert_eq!(parsed.max_attempts, 12);
}

#[test]
fn partial_config_falls_back_to_defaults() {
    let parsed: LevelMapConfig =
        ron::from_str("(poisson_radius: 60.0, policy: (allow_teleport: false))")
            .expect("partial config parses");
    let default = LevelMapConfig::default();

    assert_eq!(parsed.poisson_radius, 60.0);
    assert!(!parsed.policy.allow_teleport);

    assert_eq!(parsed.layout, default.layout);
    assert_eq!(parsed.stage_buffer, default.stage_buffer);
    assert_eq!(parsed.max_attempts, default.max_attempts);
    assert_eq!(parsed.policy.allow_revisit, default.policy.allow_revisit);
    assert_eq!(parsed.seed, None);
}

#[test]
fn empty_config_is_the_default() {
    let parsed: LevelMapConfig = ron::from_str("()").expect("empty config parses");
    let default = LevelMapConfig::default();
    assert_eq!(parsed.layout, default.layout);
    assert_eq!(parsed.poisson_radius, default.poisson_radius);
    assert_eq!(parsed.max_attempts, default.max_attempts);
}
