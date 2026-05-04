# msg_level_select

Procedural FTL-style level map generator for [Bevy](https://bevyengine.org).

Given a layout like `[1, 3, 1, 3, 3, 1]`, it generates a Voronoi-based graph where every
campaign stop connects to at least one stop in the next stage. The crate produces purely logical
ECS entities — nodes, paths, and edges tagged with a traversal FSM — and leaves all rendering to
the consumer.

| Bevy | msg_level_select |
|------|-----------------|
| 0.18 | 0.1             |

## Usage

```toml
[dependencies]
msg_level_select = { git = "https://github.com/MolecularSadism/msg_level_select", tag = "v0.1.0" }
```

## Quick start

```rust
use bevy::prelude::*;
use msg_level_select::{LevelMapCommands, LevelMapConfig, LevelMapRng, LevelSelectPlugin};

fn spawn_map(mut commands: Commands, mut rng: ResMut<LevelMapRng>) {
    let _ = commands.spawn_level_map(&mut rng, LevelMapConfig::default());
}

let mut app = App::new();
app.add_plugins(MinimalPlugins);
app.add_plugins(LevelSelectPlugin { seed: Some(42) });
app.add_systems(Startup, spawn_map);
```

## Core concepts

### Layout

`LevelMapConfig::layout` is a `Vec<u32>` where each entry is the number of campaign stops in
that stage. The first stage is the entry, the last is the exit. Every stop in every stage is
guaranteed to have at least one path to some stop in the next stage.

```
layout: [1, 3, 1, 3, 3, 1]
         ^                ^
         entry            exit
```

### Map entities

The spawn pipeline creates four kinds of entities, all children of a root `LevelMap` entity:

| Entity type | Marker component | Notes |
|-------------|-----------------|-------|
| Every Voronoi cell | `MapNode` | Always has `VoronoiCell` (polygon vertices) and `Transform` at the cell site |
| Campaign stops | `MapNode` + `Site` | `Site { belt, site }` gives the (stage, index) address. Has `LocationState`. |
| Corridor waypoints | `MapNode` + `Waypoint` | Intermediate cells along a multi-hop path. Has `LocationState`. |
| Logical paths | `MapPath` | Connects two `Site` nodes. Has `LocationState` and `PathFrom`/`PathTo`. |
| Voronoi walls | `MapEdge` | The polygon boundary between two adjacent cells. Path-bearing edges have `LocationState` and `EdgePaths`. |

Dead Voronoi cells (no `Site`, no `Waypoint`) have no `LocationState` and are provided so
consumers can render the full mosaic backdrop.

### LocationState FSM

Every `Site`, `Waypoint`, `MapPath`, and path-bearing `MapEdge` carries a `LocationState`
component driven by the `VisitLocation` event:

```
Inactive → Available → Active → Visited
                     ↑                |
                     └────────────────┘  (revisit, gated by LevelMapPolicy)
```

Priority is strict — a higher state is never downgraded by a later write. Shared Voronoi
adjacencies (an edge used by more than one path) are updated correctly regardless of iteration
order.

Fire `VisitLocation` targeting any `MapNode` or `MapPath` to move the player:

```rust
commands.trigger(VisitLocation { target: node_entity });
```

The observer automatically:
1. Marks the previously `Active` node as `Visited`.
2. Marks the connecting path, its edges, and interior waypoints as `Visited`.
3. Promotes the destination to `Active`.
4. Cascades outgoing paths, their edges, and waypoints to `Available`.

### LevelMapPolicy

`LevelMapPolicy` lives on the root `LevelMap` entity and can be mutated at any time without
regenerating the map:

```rust
fn open_cheat(mut q: Query<&mut LevelMapPolicy>) {
    if let Ok(mut policy) = q.single_mut() {
        policy.allow_teleport = true;
        policy.allow_revisit = true;
    }
}
```

| Field | Default | Effect when `false` |
|-------|---------|---------------------|
| `allow_revisit` | `false` | Rejects `VisitLocation` targeting a `Visited` node |
| `allow_teleport` | `true` | Rejects `VisitLocation` when there is no connecting path from the current node |
| `allow_path_visit` | `true` | Rejects `VisitLocation` targeting a `MapPath` directly |

## Configuration reference

`LevelMapConfig` controls both generation and runtime behavior:

```rust
LevelMapConfig {
    // Stage counts — minimum 2 entries.
    layout: vec![1, 3, 1, 3, 3, 1],

    // Poisson-disc radius in world units. Smaller = denser cells.
    poisson_radius: 40.0,

    // Buffer columns outside the visible region to avoid clipping artifacts.
    stage_buffer: 5,

    // Width / height of the visible region.
    aspect_ratio: 16.0 / 9.0,

    // Jitter radius for "point of interest" positions, as a fraction of
    // poisson_radius. 0.0 places stops exactly at the cell site.
    node_position_buffer: 0.10,

    // Maximum intermediate cells BFS may walk through to connect stages.
    // Raise when tight layouts fail to generate.
    allow_extra_traversal: 1,

    // Optional: bias corridor lengths toward a target average.
    desired_traversals: Some(DesiredTraversals {
        average: 2,
        easing: EasingCurve::new(0.0, 1.0, EaseFunction::SmoothStep),
    }),

    // Pin this map to a specific seed for deterministic output.
    // None pulls a fresh sub-seed from LevelMapRng.
    seed: Some(42),

    // Runtime traversal rules.
    policy: LevelMapPolicy::default(),

    // Retry budget when a seed fails to satisfy the layout.
    max_attempts: 8,
}
```

### Seeding

`LevelSelectPlugin` owns a `LevelMapRng` resource. A fixed `seed` on the plugin makes every
run produce the same sequence of maps. Each `LevelMapConfig` may also pin its own per-map seed
independently:

```rust
// Reproducible sequence across runs.
app.add_plugins(LevelSelectPlugin { seed: Some(0) });

// This specific map always generates identically.
commands.spawn_level_map(&mut rng, LevelMapConfig { seed: Some(99), ..default() });
```

## Relationship queries

`MapPath` entities use Bevy's relationship system for their endpoints:

```rust
// Paths leaving a node.
fn outgoing(q: Query<&OutgoingPaths, With<MapNode>>) {
    for paths in &q {
        for path_entity in paths.iter() { /* ... */ }
    }
}

// Paths arriving at a node.
fn incoming(q: Query<&IncomingPaths, With<MapNode>>) { /* ... */ }

// Endpoints of a path.
fn endpoints(q: Query<(&PathFrom, &PathTo), With<MapPath>>) {
    for (from, to) in &q {
        let source_node = from.0;
        let dest_node = to.0;
    }
}
```

The many-to-many edge↔path link uses plain `Vec<Entity>` components because a single Voronoi
adjacency can be shared by multiple paths:

```rust
// Edges that make up a path (in traversal order).
fn path_edges(q: Query<&PathEdges, With<MapPath>>) {
    for edges in &q {
        for edge_entity in edges.iter() { /* ... */ }
    }
}

// Paths that cross a given edge.
fn edge_paths(q: Query<&EdgePaths, With<MapEdge>>) { /* ... */ }
```

## Example

An interactive demo with live config editing, policy toggles, and gizmo visualization:

```sh
cargo run --example basic --features dev
```

The `dev` feature enables `bevy-inspector-egui` which the example requires.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) at your option.
