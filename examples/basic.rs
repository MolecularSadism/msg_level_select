//! Live tweakable demo of `msg_level_select`.
//!
//! Left side panel: edit the `LevelMapConfig` and hit Regenerate.
//! Toggle `LevelMapPolicy` flags live (no regeneration required).
//! Click any node or path in the viewport to fire `VisitLocation`.
//! State changes are visualized via gizmos.
//!
//! Run with: `cargo run --example basic --features dev`

use bevy::color::palettes::css;
use bevy::input::mouse::MouseButton;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use bevy_inspector_egui::bevy_egui::{
    EguiContext, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext,
};
use bevy_inspector_egui::egui;

use msg_level_select::{
    DesiredTraversals, EdgePaths, LevelMap, LevelMapCommands, LevelMapConfig, LevelMapPolicy,
    LevelMapRng, LevelSelectPlugin, LocationState, MapEdge, MapNode, MapPath, OutgoingPaths,
    PathTo, Site, VisitLocation, VoronoiCell, Waypoint,
};

#[derive(Resource, Clone, Debug)]
struct DemoConfig {
    layout_text: String,
    cfg: LevelMapConfig,
    easing_choice: EasingChoice,
    desired_avg: u8,
    desired_enabled: bool,
    edge_style: EdgeStyle,
}

/// How to draw `MapEdge` entities (and the surrounding Voronoi backdrop)
/// in the viewport. All variants use the same underlying entities — the
/// wall geometry comes from [`MapEdge::wall`] and the site geometry from
/// the `from`/`to` cell transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeStyle {
    /// Don't draw any edges or cell outlines. Only nodes (and the map
    /// bounding box) are visible.
    Off,
    /// Draw the shared polygon boundary (the actual Voronoi wall).
    Wall,
    /// Draw a straight line from one cell's site to the other's, with
    /// faint cell-polygon outlines behind it as a backdrop.
    SiteToSite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EasingChoice {
    SmoothStep,
    Linear,
    QuadraticInOut,
}

impl EasingChoice {
    fn build(self) -> EasingCurve<f32> {
        let f = match self {
            EasingChoice::SmoothStep => EaseFunction::SmoothStep,
            EasingChoice::Linear => EaseFunction::Linear,
            EasingChoice::QuadraticInOut => EaseFunction::QuadraticInOut,
        };
        EasingCurve::new(0.0, 1.0, f)
    }
}

impl Default for DemoConfig {
    fn default() -> Self {
        let cfg = LevelMapConfig {
            layout: vec![1, 3, 1, 3, 3, 1],
            poisson_radius: 40.0,
            stage_buffer: 5,
            aspect_ratio: 16.0 / 9.0,
            node_position_buffer: 0.10,
            allow_extra_traversal: 2,
            desired_traversals: Some(DesiredTraversals::default()),
            seed: Some(42),
            policy: LevelMapPolicy::default(),
            max_attempts: 8,
        };
        Self {
            layout_text: layout_to_string(&cfg.layout),
            cfg,
            easing_choice: EasingChoice::SmoothStep,
            desired_avg: 1,
            desired_enabled: true,
            edge_style: EdgeStyle::SiteToSite,
        }
    }
}

fn layout_to_string(layout: &[u32]) -> String {
    layout
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_layout(s: &str) -> Option<Vec<u32>> {
    let parts: Result<Vec<u32>, _> = s.split(',').map(|t| t.trim().parse::<u32>()).collect();
    let parts = parts.ok()?;
    if parts.len() < 2 || parts.contains(&0) {
        return None;
    }
    Some(parts)
}

#[derive(Resource, Default)]
struct CurrentMap(Option<Entity>);

/// Sent by the side panel; consumed by `regenerate_system`.
#[derive(Message)]
struct RegenerateRequested;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin::default())
        .add_plugins(LevelSelectPlugin { seed: Some(0) })
        .init_resource::<DemoConfig>()
        .init_resource::<CurrentMap>()
        .add_message::<RegenerateRequested>()
        .add_systems(Startup, (setup_camera, initial_spawn).chain())
        .add_systems(Update, (regenerate_system, click_to_visit, draw_gizmos))
        .add_systems(EguiPrimaryContextPass, side_panel_ui)
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((Camera2d, Transform::default()));
}

fn initial_spawn(
    mut commands: Commands,
    mut rng: ResMut<LevelMapRng>,
    mut current: ResMut<CurrentMap>,
    demo: Res<DemoConfig>,
) {
    spawn_or_log(&mut commands, &mut rng, &mut current, demo.cfg.clone());
}

fn spawn_or_log(
    commands: &mut Commands,
    rng: &mut LevelMapRng,
    current: &mut CurrentMap,
    cfg: LevelMapConfig,
) {
    if let Some(prev) = current.0.take() {
        commands.entity(prev).despawn();
    }
    match commands.spawn_level_map(rng, cfg) {
        Ok(root) => {
            current.0 = Some(root);
        }
        Err(e) => {
            warn!("level map generation failed: {e:?}");
        }
    }
}

fn regenerate_system(
    mut commands: Commands,
    mut rng: ResMut<LevelMapRng>,
    mut messages: MessageReader<RegenerateRequested>,
    mut current: ResMut<CurrentMap>,
    demo: Res<DemoConfig>,
) {
    if messages.read().next().is_none() {
        return;
    }
    spawn_or_log(&mut commands, &mut rng, &mut current, demo.cfg.clone());
}

#[allow(clippy::too_many_arguments)]
fn side_panel_ui(
    mut egui_ctx_q: Query<&mut EguiContext, With<PrimaryEguiContext>>,
    mut demo: ResMut<DemoConfig>,
    mut regen: MessageWriter<RegenerateRequested>,
    mut q_policy: Query<&mut LevelMapPolicy>,
    q_active_node: Query<(Entity, &LocationState, &Name), (With<MapNode>, With<Site>)>,
    q_outgoing: Query<&OutgoingPaths>,
    q_path_to: Query<(&PathTo, &Name), With<MapPath>>,
    q_node_name: Query<&Name>,
    mut commands: Commands,
) {
    let Ok(mut egui_ctx) = egui_ctx_q.single_mut() else {
        return;
    };
    let ctx = egui_ctx.get_mut();

    egui::SidePanel::left("config_panel")
        .resizable(true)
        .default_width(280.0)
        .show(ctx, |ui| {
            if ui
                .add(egui::Button::new("Regenerate Map").min_size(egui::vec2(200.0, 28.0)))
                .clicked()
            {
                regen.write(RegenerateRequested);
            }

            ui.separator();
            ui.heading("LevelMapConfig");

            ui.horizontal(|ui| {
                ui.label("layout:");
                let response = ui.text_edit_singleline(&mut demo.layout_text);
                if response.changed()
                    && let Some(parsed) = parse_layout(&demo.layout_text)
                {
                    demo.cfg.layout = parsed;
                }
            });

            ui.horizontal(|ui| {
                ui.label("seed:");
                let mut pin = demo.cfg.seed.is_some();
                if ui.checkbox(&mut pin, "pin").changed() {
                    demo.cfg.seed = pin.then_some(0);
                }
                if let Some(seed) = demo.cfg.seed.as_mut() {
                    ui.add(egui::DragValue::new(seed).speed(1.0));
                } else {
                    ui.label("(from plugin RNG)");
                }
            });
            ui.horizontal(|ui| {
                ui.label("poisson_radius:");
                ui.add(
                    egui::DragValue::new(&mut demo.cfg.poisson_radius)
                        .range(5.0..=200.0)
                        .speed(0.5),
                );
            });
            ui.horizontal(|ui| {
                ui.label("stage_buffer:");
                ui.add(egui::DragValue::new(&mut demo.cfg.stage_buffer).range(0..=20));
            });
            ui.horizontal(|ui| {
                ui.label("aspect_ratio:");
                ui.add(
                    egui::DragValue::new(&mut demo.cfg.aspect_ratio)
                        .range(0.3..=4.0)
                        .speed(0.05),
                );
            });
            ui.horizontal(|ui| {
                ui.label("node_position_buffer:");
                ui.add(
                    egui::DragValue::new(&mut demo.cfg.node_position_buffer)
                        .range(0.0..=0.45)
                        .speed(0.01),
                );
            });
            ui.horizontal(|ui| {
                ui.label("allow_extra_traversal:");
                ui.add(egui::DragValue::new(&mut demo.cfg.allow_extra_traversal).range(0..=5));
            });

            ui.separator();
            ui.label("desired_traversals");
            ui.checkbox(&mut demo.desired_enabled, "enabled");
            ui.add_enabled_ui(demo.desired_enabled, |ui| {
                ui.horizontal(|ui| {
                    ui.label("average:");
                    ui.add(egui::DragValue::new(&mut demo.desired_avg).range(0..=8));
                });
                ui.horizontal(|ui| {
                    ui.label("easing:");
                    egui::ComboBox::from_id_salt("easing_combo")
                        .selected_text(format!("{:?}", demo.easing_choice))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut demo.easing_choice,
                                EasingChoice::SmoothStep,
                                "SmoothStep",
                            );
                            ui.selectable_value(
                                &mut demo.easing_choice,
                                EasingChoice::Linear,
                                "Linear",
                            );
                            ui.selectable_value(
                                &mut demo.easing_choice,
                                EasingChoice::QuadraticInOut,
                                "QuadraticInOut",
                            );
                        });
                });
            });

            // Sync derived fields back into cfg.
            demo.cfg.desired_traversals = if demo.desired_enabled {
                Some(DesiredTraversals {
                    average: demo.desired_avg,
                    easing: demo.easing_choice.build(),
                })
            } else {
                None
            };

            ui.separator();
            ui.heading("LevelMapPolicy (live)");
            if let Ok(mut policy) = q_policy.single_mut() {
                ui.checkbox(&mut policy.allow_revisit, "revisit");
                ui.checkbox(&mut policy.allow_teleport, "teleport");
                demo.cfg.policy = *policy;
            } else {
                ui.label("(no map spawned)");
            }

            ui.separator();
            ui.heading("Display");
            ui.horizontal(|ui| {
                ui.label("edge style:");
                egui::ComboBox::from_id_salt("edge_style_combo")
                    .selected_text(format!("{:?}", demo.edge_style))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut demo.edge_style, EdgeStyle::Off, "Off");
                        ui.selectable_value(&mut demo.edge_style, EdgeStyle::Wall, "Wall");
                        ui.selectable_value(
                            &mut demo.edge_style,
                            EdgeStyle::SiteToSite,
                            "SiteToSite",
                        );
                    });
            });

            ui.separator();
            ui.heading("Active node");
            let active = q_active_node
                .iter()
                .find(|(_, s, _)| **s == LocationState::Active);
            if let Some((node, _, name)) = active {
                ui.label(format!("{name}"));
                ui.label("Available paths:");
                if let Ok(out) = q_outgoing.get(node) {
                    for path in out.iter() {
                        let label = q_path_to
                            .get(path)
                            .ok()
                            .map(|(to, n)| {
                                let dest_name = q_node_name.get(to.0).ok().map(|n| n.to_string());
                                (n.to_string(), dest_name.unwrap_or_default())
                            })
                            .map(|(p, d)| format!("{p}  ->  {d}"))
                            .unwrap_or_else(|| "<unknown>".to_string());
                        if ui.button(label).clicked() {
                            commands.trigger(VisitLocation { target: path });
                        }
                    }
                }
            } else {
                ui.label("(none)");
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn click_to_visit(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    q_nodes: Query<(Entity, &Transform), With<MapNode>>,
    q_edges: Query<(&MapEdge, &EdgePaths)>,
    q_node_pos: Query<&Transform, With<MapNode>>,
    demo: Res<DemoConfig>,
    mut commands: Commands,
    mut egui_ctx_q: Query<&mut EguiContext, With<PrimaryEguiContext>>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    // Block clicks landing on the egui panel.
    if let Ok(mut ctx) = egui_ctx_q.single_mut()
        && ctx.get_mut().wants_pointer_input()
    {
        return;
    }

    let Ok(window) = windows.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((cam, cam_xf)) = cameras.single() else {
        return;
    };
    let Ok(world) = cam.viewport_to_world_2d(cam_xf, cursor) else {
        return;
    };

    // Closest node within a small radius wins; otherwise check edges.
    let click_radius = 18.0;
    let mut best_node: Option<(Entity, f32)> = None;
    for (e, t) in &q_nodes {
        let p = t.translation.truncate();
        let d = p.distance(world);
        if d <= click_radius && best_node.is_none_or(|(_, bd)| d < bd) {
            best_node = Some((e, d));
        }
    }
    if let Some((e, _)) = best_node {
        commands.trigger(VisitLocation { target: e });
        return;
    }

    // Check edges: only path-bearing edges are clickable (dead Voronoi
    // adjacencies have no `EdgePaths`). Hit-test against whichever
    // segment is currently visible so clicks register where the user
    // sees the line. With EdgeStyle::Off there is nothing to click.
    // Shared edges belong to multiple paths — arbitrarily target the
    // first one; the observer handles it the same way regardless.
    let mut best_path: Option<(Entity, f32)> = None;
    for (edge, edge_paths) in &q_edges {
        let Some(path) = edge_paths.iter().next() as Option<Entity> else {
            continue;
        };
        let (a, b) = match demo.edge_style {
            EdgeStyle::Off => continue,
            EdgeStyle::Wall => (edge.wall[0], edge.wall[1]),
            EdgeStyle::SiteToSite => {
                let Ok(from) = q_node_pos.get(edge.from) else {
                    continue;
                };
                let Ok(to) = q_node_pos.get(edge.to) else {
                    continue;
                };
                (from.translation.truncate(), to.translation.truncate())
            }
        };
        let d = distance_to_segment(world, a, b);
        if d <= click_radius && best_path.is_none_or(|(_, bd)| d < bd) {
            best_path = Some((path, d));
        }
    }
    if let Some((path, _)) = best_path {
        commands.trigger(VisitLocation { target: path });
    }
}

fn distance_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 < 1e-6 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    let closest = a + ab * t;
    p.distance(closest)
}

#[allow(clippy::too_many_arguments)]
fn draw_gizmos(
    mut gizmos: Gizmos,
    demo: Res<DemoConfig>,
    q_root: Query<&LevelMap>,
    q_level_nodes: Query<
        (&Transform, &LocationState, Option<&Site>, Option<&Waypoint>),
        With<MapNode>,
    >,
    q_cells: Query<&VoronoiCell, With<MapNode>>,
    q_edges: Query<(&MapEdge, Option<&EdgePaths>, Option<&LocationState>)>,
    q_node_pos: Query<&Transform, With<MapNode>>,
) {
    // Bounding box of the map (visual reference).
    if let Ok(map) = q_root.single() {
        gizmos.rect_2d(Vec2::ZERO, map.size, css::DARK_GRAY);
    }

    // EdgeStyle::Off — bail out entirely, leaving only the bbox + nodes.
    if demo.edge_style == EdgeStyle::Off {
        draw_nodes(&mut gizmos, &q_level_nodes);
        return;
    }

    // In SiteToSite mode the polygon walls aren't carrying the "this is
    // a Voronoi diagram" message, so draw cell outlines behind the
    // site-to-site lines as a faint backdrop.
    if demo.edge_style == EdgeStyle::SiteToSite {
        let cell_outline = Color::srgba(0.18, 0.18, 0.22, 1.0);
        for cell in &q_cells {
            if cell.vertices.len() < 2 {
                continue;
            }
            for window in cell.vertices.windows(2) {
                gizmos.line_2d(window[0], window[1], cell_outline);
            }
            if let (Some(first), Some(last)) = (cell.vertices.first(), cell.vertices.last()) {
                gizmos.line_2d(*last, *first, cell_outline);
            }
        }
    }

    // Edges: path-bearing edges are tinted by their own LocationState,
    // which the observer keeps in sync with whichever path is currently
    // highest-priority (shared edges cover all their paths at once).
    // "Dead" adjacencies (no EdgePaths) are drawn as a faint mesh.
    let dead_edge_color = Color::srgba(0.22, 0.22, 0.26, 1.0);
    for (edge, edge_paths, edge_state) in &q_edges {
        let color = match (edge_paths, edge_state) {
            (Some(_), Some(state)) => state_color(*state),
            _ => dead_edge_color,
        };
        let (a, b) = match demo.edge_style {
            EdgeStyle::Wall => (edge.wall[0], edge.wall[1]),
            EdgeStyle::SiteToSite => {
                let Ok(from) = q_node_pos.get(edge.from) else {
                    continue;
                };
                let Ok(to) = q_node_pos.get(edge.to) else {
                    continue;
                };
                (from.translation.truncate(), to.translation.truncate())
            }
            EdgeStyle::Off => unreachable!("Off branch returned earlier"),
        };
        gizmos.line_2d(a, b, color);
    }

    draw_nodes(&mut gizmos, &q_level_nodes);
}

fn draw_nodes(
    gizmos: &mut Gizmos,
    q_level_nodes: &Query<
        (&Transform, &LocationState, Option<&Site>, Option<&Waypoint>),
        With<MapNode>,
    >,
) {
    // Nodes — only Levels and Waypoints carry LocationState, so this
    // filter naturally skips dead cells.
    for (xf, state, level, waypoint) in q_level_nodes.iter() {
        let color = state_color(*state);
        let radius = if level.is_some() {
            12.0
        } else if waypoint.is_some() {
            5.0
        } else {
            7.0
        };
        gizmos.circle_2d(xf.translation.truncate(), radius, color);
    }
}

fn state_color(state: LocationState) -> Color {
    match state {
        LocationState::Inactive => Color::srgb(0.30, 0.30, 0.32),
        LocationState::Available => Color::srgb(0.95, 0.85, 0.20),
        LocationState::Active => Color::srgb(0.95, 0.95, 0.95),
        LocationState::Visited => Color::srgb(0.30, 0.55, 0.95),
    }
}
