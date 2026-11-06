//! Quillip, one page: a sentence about the story, then the story itself.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Mutex, MutexGuard};
use std::thread;

use bevy::asset::RenderAssetUsages;
use bevy::audio::{AddAudioSource, PlaybackSettings, Volume};
use bevy::image::{Image, ImageAddressMode, ImageSampler, ImageSamplerDescriptor};
use bevy::input_focus::AutoFocus;
use bevy::input_focus::tab_navigation::{TabIndex, TabNavigationPlugin};
use bevy::picking::Pickable;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::text::{
    EditableText, FontWeight, Justify, LineBreak, LineHeight, TextCursorStyle, TextReadWriteMode,
};
use bevy::ui::widget::NodeImageMode;
use bevy::ui::{BackgroundGradient, ColorStop, InteractionDisabled, LinearGradient};
use bevy::ui_widgets::{
    Activate, ActivateOnPress, Button, ScrollArea, Slider, SliderRange, SliderThumb, SliderValue,
    TextInput, slider_self_update,
};
use bevy::window::WindowResolution;

use crate::audio::{self, Speech};
use crate::engine::{self, Command, EngineConfig, Phase, WorkerEvent};
use crate::plan::FinishedStory;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PrimaryOffer {
    Warming,
    Write,
    Stop,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadOffer {
    Read,
    Pause,
    Continue,
}

const NIGHT: Color = Color::srgb(0.07, 0.06, 0.05);
const NIGHT_TOP: Color = Color::srgb(0.16, 0.11, 0.09);
const PAGE: Color = Color::srgb(0.90, 0.84, 0.73);
const PAGE_TOP: Color = Color::srgb(0.98, 0.95, 0.88);
const INK: Color = Color::srgb(0.16, 0.12, 0.09);
const MUTED: Color = Color::srgb(0.45, 0.36, 0.27);
const DRAFT: Color = Color::srgb(0.62, 0.52, 0.40);
const ALERT: Color = Color::srgb(0.55, 0.22, 0.16);
const FIELD: Color = Color::srgb(0.97, 0.94, 0.88);
const RUST: Color = Color::srgb(0.52, 0.22, 0.14);
const RUST_TOP: Color = Color::srgb(0.72, 0.36, 0.22);
const READ_TOP: Color = Color::srgb(0.34, 0.24, 0.18);
const CREAM: Color = Color::srgb(0.98, 0.95, 0.90);
const RULE: Color = Color::srgb(0.72, 0.60, 0.44);
const SPINE: Color = Color::srgb(0.36, 0.14, 0.10);
const SPINE_TOP: Color = Color::srgb(0.62, 0.28, 0.16);
const FIELD_TOP: Color = Color::srgb(0.99, 0.97, 0.93);
const DEFAULT_VOLUME: f32 = 0.85;

pub fn run(config: EngineConfig) {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Quillip".into(),
                    resolution: WindowResolution::new(1100, 820),
                    ..default()
                }),
                ..default()
            }),
            TabNavigationPlugin,
        ))
        .add_audio_source::<Speech>()
        .insert_resource(config)
        .init_resource::<Desk>()
        .init_resource::<Ops>()
        .init_resource::<Playback>()
        .init_resource::<Built>()
        .init_resource::<VolumeLevel>()
        .add_observer(slider_self_update)
        .add_observer(on_activate)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                poll_worker,
                apply_ops,
                rebuild_story,
                paint_draft,
                sync_chrome,
                sync_button,
                sync_read,
                paint_progress,
                read_volume,
                pump_audio,
                apply_volume,
            )
                .chain(),
        )
        .run();
}

#[derive(Resource)]
struct EngineHandle {
    tx: Sender<Command>,
    events: Mutex<Receiver<WorkerEvent>>,
}

#[derive(Resource)]
struct Desk {
    busy: bool,
    loaded: bool,
    status: String,
    detail: String,
    step: u32,
    total: u32,
    story: Option<FinishedStory>,
    clips: Vec<Option<Vec<f32>>>,
    cue_count: usize,
    story_rev: u64,
    draft: String,
    error: String,
}

impl Default for Desk {
    fn default() -> Self {
        Self {
            busy: true,
            loaded: false,
            status: "Opening the book.".into(),
            detail: "Spark is loading.".into(),
            step: 0,
            total: 1,
            story: None,
            clips: Vec::new(),
            cue_count: 0,
            story_rev: 0,
            draft: String::new(),
            error: String::new(),
        }
    }
}

#[derive(Resource, Default)]
struct Ops(Vec<Action>);

#[derive(Resource, Default)]
struct Playback {
    cursor: usize,
    paused: bool,
    autoplay: bool,
    player: Option<Entity>,
}

#[derive(Resource, Default)]
struct Built {
    story: u64,
}

#[derive(Resource)]
struct VolumeLevel(f32);

impl Default for VolumeLevel {
    fn default() -> Self {
        Self(DEFAULT_VOLUME)
    }
}

impl Desk {
    fn apply_phase(&mut self, phase: Phase, detail: String, step: u32, total: u32) {
        self.busy = phase.is_work();
        self.status = phase.headline().into();
        self.detail = detail;
        self.step = step;
        self.total = total.max(1);
    }

    fn fail(&mut self, status: &str, detail: &str, error: String) {
        self.busy = false;
        self.draft.clear();
        self.status = status.into();
        self.detail = detail.into();
        self.error = error;
    }

    fn begin_plan(&mut self) {
        self.story = None;
        self.story_rev += 1;
        self.draft.clear();
        self.error.clear();
        self.clips.clear();
        self.cue_count = 0;
        self.apply_phase(Phase::Planning, String::new(), 0, 1);
    }
}

#[derive(Component)]
struct StatusText;
#[derive(Component)]
struct DetailText;
#[derive(Component)]
struct ErrorText;
#[derive(Component)]
struct DraftText;
#[derive(Component)]
struct Placeholder;
#[derive(Component)]
struct ProgressFill;
#[derive(Component)]
struct VolumeRow;
#[derive(Component)]
struct VolumeSlider;
#[derive(Component)]
struct StoryRoot;
#[derive(Component)]
struct PremiseInput;
#[derive(Component)]
struct PrimaryButton;
#[derive(Component)]
struct PrimaryLabel;
#[derive(Component)]
struct ReadButton;
#[derive(Component)]
struct ReadLabel;
#[derive(Component)]
struct SpeechPlayer;

#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum Action {
    Begin,
    Read,
    Stop,
    Pause,
}

fn setup(mut commands: Commands, config: Res<EngineConfig>, mut images: ResMut<Assets<Image>>) {
    commands.spawn(Camera2d);
    let (tx, rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let config = config.clone();
    thread::Builder::new()
        .name("spark".into())
        .spawn(move || engine::serve(config, rx, event_tx))
        .expect("spark thread");
    commands.insert_resource(EngineHandle {
        tx,
        events: Mutex::new(event_rx),
    });
    let night_grain = images.add(paper_grain(false));
    let page_grain = images.add(paper_grain(true));

    commands.spawn((
        Node {
            width: percent(100.),
            height: percent(100.),
            justify_content: JustifyContent::Center,
            padding: UiRect::all(px(36.)),
            ..default()
        },
        BackgroundColor(NIGHT),
        warm_wash(LinearGradient::TO_BOTTOM, NIGHT_TOP, NIGHT),
        children![
            grain_veil(night_grain),
            (
                Node {
                    width: percent(100.),
                    max_width: px(740.),
                    height: percent(100.),
                    flex_direction: FlexDirection::Row,
                    overflow: Overflow::clip(),
                    border_radius: BorderRadius::all(px(16.)),
                    ..default()
                },
                BackgroundColor(PAGE),
                warm_wash(LinearGradient::TO_BOTTOM, PAGE_TOP, PAGE),
                children![
                    grain_veil(page_grain.clone()),
                    (
                        Node {
                            width: px(14.),
                            height: percent(100.),
                            ..default()
                        },
                        BackgroundColor(SPINE),
                        warm_wash(LinearGradient::TO_BOTTOM, SPINE_TOP, SPINE),
                    ),
                    (
                        Node {
                            flex_grow: 1.0,
                            height: percent(100.),
                            flex_direction: FlexDirection::Column,
                            padding: UiRect::axes(px(36.), px(28.)),
                            row_gap: px(14.),
                            ..default()
                        },
                        children![
                            prose("QUILLIP", 13.0, MUTED),
                            prose("Opening the book.", 28.0, INK).with_status(),
                            prose("Spark is loading.", 15.0, MUTED).with_detail(),
                            prose("", 15.0, ALERT).with_error(),
                            progress_rule(),
                            prose("Describe the story. A sentence is enough.", 16.0, MUTED),
                            premise_field(),
                            (
                                Node {
                                    flex_grow: 1.0,
                                    min_height: px(0.),
                                    width: percent(100.),
                                    overflow: Overflow::scroll_y(),
                                    flex_direction: FlexDirection::Column,
                                    row_gap: px(12.),
                                    ..default()
                                },
                                StoryRoot,
                                ScrollArea,
                                children![(
                                    prose("The story will appear here.", 18.0, MUTED),
                                    Placeholder,
                                )],
                            ),
                            (
                                Node {
                                    width: percent(100.),
                                    column_gap: px(16.),
                                    align_items: AlignItems::Center,
                                    justify_content: JustifyContent::SpaceBetween,
                                    ..default()
                                },
                                children![
                                    volume_row(),
                                    (Node {
                                        flex_grow: 1.0,
                                        ..default()
                                    },),
                                    read_button(),
                                    primary_button(),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    ));
}

trait MarkText {
    fn with_status(self) -> (Self, StatusText)
    where
        Self: Sized;
    fn with_detail(self) -> (Self, DetailText)
    where
        Self: Sized;
    fn with_error(self) -> (Self, ErrorText)
    where
        Self: Sized;
}

impl<T> MarkText for T {
    fn with_status(self) -> (Self, StatusText) {
        (self, StatusText)
    }
    fn with_detail(self) -> (Self, DetailText) {
        (self, DetailText)
    }
    fn with_error(self) -> (Self, ErrorText) {
        (self, ErrorText)
    }
}

fn prose(text: impl Into<String>, size: f32, color: Color) -> impl Bundle {
    (
        Text::new(text),
        TextFont {
            font_size: FontSize::Px(size),
            ..default()
        },
        LineHeight::RelativeToFont(1.45),
        TextLayout::new(Justify::Left, LineBreak::WordBoundary),
        TextColor(color),
        Node {
            width: percent(100.),
            ..default()
        },
    )
}

fn premise_field() -> impl Bundle {
    let mut editable = EditableText::new("");
    editable.allow_newlines = true;
    editable.visible_lines = Some(3.);
    (
        Node {
            width: percent(100.),
            border_radius: BorderRadius::all(px(8.)),
            padding: px(12.).all(),
            ..default()
        },
        TextInput,
        TextReadWriteMode::Editable,
        TabIndex(0),
        AutoFocus,
        TextCursorStyle {
            color: INK,
            selection_color: Color::srgba(0.72, 0.36, 0.22, 0.45),
            unfocused_selection_color: Color::srgba(0.45, 0.36, 0.27, 0.28),
            selected_text_color: None,
            selection_radius: 0.15,
        },
        editable,
        TextFont {
            font_size: FontSize::Px(20.),
            ..default()
        },
        LineHeight::RelativeToFont(1.4),
        TextLayout::new(Justify::Left, LineBreak::WordBoundary),
        TextColor(INK),
        BackgroundColor(FIELD),
        warm_wash(LinearGradient::TO_BOTTOM, FIELD_TOP, FIELD),
        PremiseInput,
    )
}

fn warm_wash(angle: f32, from: Color, to: Color) -> BackgroundGradient {
    BackgroundGradient::from(LinearGradient::new(
        angle,
        vec![ColorStop::percent(from, 0.), ColorStop::percent(to, 100.)],
    ))
}

fn grain_veil(image: Handle<Image>) -> impl Bundle {
    (
        Node {
            position_type: PositionType::Absolute,
            left: px(0.),
            top: px(0.),
            width: percent(100.),
            height: percent(100.),
            ..default()
        },
        ImageNode {
            color: Color::srgba(1.0, 0.96, 0.9, 0.55),
            image,
            image_mode: NodeImageMode::Tiled {
                tile_x: true,
                tile_y: true,
                stretch_value: 1.0,
            },
            ..default()
        },
        Pickable::IGNORE,
    )
}

fn paper_grain(warm: bool) -> Image {
    let size = 96u32;
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let speck = grain_hash(x, y) % 41;
            let (red, green, blue, alpha) = if warm {
                (214, 188, 152, 16 + speck / 4)
            } else {
                (255, 236, 210, 8 + speck / 6)
            };
            data.extend([red, green, blue, alpha]);
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    let mut sampler = ImageSamplerDescriptor::linear();
    sampler.address_mode_u = ImageAddressMode::Repeat;
    sampler.address_mode_v = ImageAddressMode::Repeat;
    image.sampler = ImageSampler::Descriptor(sampler);
    image
}

fn grain_hash(x: u32, y: u32) -> u8 {
    let mut n = x
        .wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263));
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n ^ (n >> 16)) as u8
}

fn progress_rule() -> impl Bundle {
    (
        Node {
            width: percent(100.),
            height: px(3.),
            ..default()
        },
        BackgroundColor(RULE),
        children![(
            ProgressFill,
            Node {
                width: percent(0.),
                height: percent(100.),
                ..default()
            },
            BackgroundColor(RUST),
        )],
    )
}

fn volume_row() -> impl Bundle {
    (
        Node {
            width: px(180.),
            height: px(22.),
            column_gap: px(8.),
            align_items: AlignItems::Center,
            display: Display::None,
            ..default()
        },
        VolumeRow,
        children![
            (
                Text::new("Voice"),
                TextFont {
                    font_size: FontSize::Px(14.),
                    ..default()
                },
                TextColor(MUTED),
            ),
            (
                Node {
                    width: px(120.),
                    height: px(18.),
                    ..default()
                },
                VolumeSlider,
                Hovered::default(),
                Slider::default(),
                SliderValue(DEFAULT_VOLUME),
                SliderRange::new(0.0, 1.0),
                children![
                    (
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(0.),
                            right: px(0.),
                            top: px(7.),
                            height: px(4.),
                            border_radius: BorderRadius::all(px(2.)),
                            ..default()
                        },
                        BackgroundColor(RULE),
                    ),
                    (
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(0.),
                            right: px(10.),
                            top: px(0.),
                            bottom: px(0.),
                            ..default()
                        },
                        children![(
                            SliderThumb,
                            Node {
                                width: px(10.),
                                height: px(10.),
                                position_type: PositionType::Absolute,
                                top: px(4.),
                                left: percent(DEFAULT_VOLUME * 100.),
                                border_radius: BorderRadius::all(px(5.)),
                                ..default()
                            },
                            BackgroundColor(INK),
                        )],
                    ),
                ],
            ),
        ],
    )
}

fn command_button(
    action: Action,
    words: &'static str,
    label: impl Component,
    fill: Color,
    top: Color,
    shown: bool,
) -> impl Bundle {
    (
        Button,
        ActivateOnPress,
        action,
        Node {
            padding: UiRect::axes(px(18.), px(10.)),
            min_width: px(168.),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            border_radius: BorderRadius::all(px(8.)),
            display: if shown { Display::Flex } else { Display::None },
            ..default()
        },
        BackgroundColor(fill),
        warm_wash(LinearGradient::TO_BOTTOM, top, fill),
        children![(
            Text::new(words),
            TextFont {
                font_size: FontSize::Px(16.),
                ..default()
            },
            TextColor(CREAM),
            label,
        )],
    )
}

fn primary_button() -> impl Bundle {
    (
        command_button(
            Action::Begin,
            "Write it",
            PrimaryLabel,
            RUST,
            RUST_TOP,
            true,
        ),
        PrimaryButton,
    )
}

fn read_button() -> impl Bundle {
    (
        command_button(
            Action::Read,
            "Read it aloud",
            ReadLabel,
            INK,
            READ_TOP,
            false,
        ),
        ReadButton,
    )
}

fn on_activate(activate: On<Activate>, actions: Query<&Action>, mut ops: ResMut<Ops>) {
    let Ok(action) = actions.get(activate.entity) else {
        return;
    };
    ops.0.push(*action);
}

fn poll_worker(
    handle: Option<Res<EngineHandle>>,
    mut desk: ResMut<Desk>,
    mut playback: ResMut<Playback>,
    mut commands: Commands,
) {
    let Some(handle) = handle else {
        return;
    };
    let Ok(events) = handle.events.lock() else {
        return;
    };
    while let Some(event) = recv(&events) {
        match event {
            WorkerEvent::Ready { .. } => {
                desk.busy = false;
                desk.loaded = true;
                desk.status = "Describe the story.".into();
                desk.detail = "A sentence is enough.".into();
                desk.step = 0;
                desk.total = 1;
            }
            WorkerEvent::Draft(text) => {
                desk.draft = text;
            }
            WorkerEvent::Status {
                phase,
                detail,
                step,
                total,
            } => desk.apply_phase(phase, detail, step, total),
            WorkerEvent::PlanReady(plan) => {
                desk.draft.clear();
                if plan.beats.is_empty() {
                    desk.fail(
                        "The plan was empty.",
                        "Try the idea in different words.",
                        "The model returned a plan with no sections.".into(),
                    );
                    continue;
                }
                desk.error.clear();
                let beats = plan.beats.len() as u32;
                desk.busy = true;
                desk.status = "Writing.".into();
                desk.detail = format!("Section 1 of {beats}");
                desk.step = 0;
                desk.total = beats;
                let _ = handle.tx.send(Command::Write { plan });
            }
            WorkerEvent::PlanUnparsed { error, .. } => {
                desk.fail("The plan could not be read.", "", error);
            }
            WorkerEvent::StoryReady(story) => {
                let complete = story.complete;
                desk.story = Some(story);
                desk.story_rev += 1;
                desk.draft.clear();
                if complete {
                    desk.busy = false;
                    desk.error.clear();
                }
            }
            WorkerEvent::Script(cues) => {
                desk.clips = vec![None; cues.len()];
                desk.cue_count = cues.len();
                playback.cursor = 0;
                playback.autoplay = true;
                playback.paused = false;
                if let Some(player) = playback.player.take() {
                    commands.entity(player).despawn();
                }
            }
            WorkerEvent::Clip { index, samples } => {
                if index < desk.clips.len() {
                    desk.clips[index] = Some(samples);
                }
            }
            WorkerEvent::Failed(error) => {
                desk.fail("The story could not be finished.", "", error);
            }
        }
    }
}

fn recv(events: &MutexGuard<Receiver<WorkerEvent>>) -> Option<WorkerEvent> {
    match events.try_recv() {
        Ok(event) => Some(event),
        Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
    }
}

fn apply_ops(
    mut ops: ResMut<Ops>,
    mut desk: ResMut<Desk>,
    mut playback: ResMut<Playback>,
    handle: Option<Res<EngineHandle>>,
    mut commands: Commands,
    premise: Query<&EditableText, With<PremiseInput>>,
) {
    let queued: Vec<Action> = std::mem::take(&mut ops.0);
    for op in queued {
        match op {
            Action::Begin => {
                if desk.busy || !desk.loaded {
                    continue;
                }
                let idea = text_of(&premise);
                if idea.trim().is_empty() {
                    desk.status = "Describe the story.".into();
                    desk.detail = "A sentence is enough.".into();
                    continue;
                }
                stop_audio(&mut playback, &mut commands);
                desk.begin_plan();
                if let Some(handle) = handle.as_ref() {
                    let _ = handle.tx.send(Command::Plan { premise: idea });
                }
            }
            Action::Read => {
                if desk.busy {
                    continue;
                }
                let Some(story) = desk.story.clone() else {
                    continue;
                };
                if !story.audible() {
                    continue;
                }
                desk.busy = true;
                desk.error.clear();
                desk.status = "Preparing the voice.".into();
                desk.detail = String::new();
                if let Some(handle) = handle.as_ref() {
                    let _ = handle.tx.send(Command::Narrate { story });
                }
            }
            Action::Stop => {
                if let Some(handle) = handle.as_ref() {
                    let _ = handle.tx.send(Command::Cancel);
                }
                stop_audio(&mut playback, &mut commands);
                desk.busy = false;
                desk.draft.clear();
                desk.status = "Stopped.".into();
                desk.detail = "Completed sections remain on the page.".into();
            }
            Action::Pause => {
                playback.paused = !playback.paused;
            }
        }
    }
}

fn stop_audio(playback: &mut Playback, commands: &mut Commands) {
    playback.autoplay = false;
    playback.paused = false;
    playback.cursor = 0;
    if let Some(player) = playback.player.take() {
        commands.entity(player).despawn();
    }
}

fn text_of(query: &Query<&EditableText, With<PremiseInput>>) -> String {
    let Some(text) = query.iter().next() else {
        return String::new();
    };
    let mut output = String::new();
    for part in text.value() {
        output.push_str(part);
    }
    output
}

fn rebuild_story(
    mut commands: Commands,
    desk: Res<Desk>,
    mut built: ResMut<Built>,
    roots: Query<Entity, With<StoryRoot>>,
) {
    if built.story == desk.story_rev {
        return;
    }
    built.story = desk.story_rev;
    let Ok(root) = roots.single() else {
        return;
    };
    commands.entity(root).despawn_children();
    let Some(story) = desk.story.clone() else {
        commands.entity(root).with_children(|parent| {
            parent.spawn((
                prose("The story will appear here.", 18.0, MUTED),
                Placeholder,
            ));
            parent.spawn(draft_line());
        });
        return;
    };
    commands.entity(root).with_children(|parent| {
        for written in &story.beats {
            parent.spawn(prose(written.beat.display_title(), 13.0, MUTED));
            for paragraph in written.prose.split("\n\n") {
                let paragraph = paragraph.trim();
                if paragraph.is_empty() {
                    continue;
                }
                parent.spawn(prose(paragraph, 18.0, INK));
            }
        }
        parent.spawn(draft_line());
    });
}

fn draft_line() -> impl Bundle {
    (
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(18.),
            weight: FontWeight::LIGHT,
            ..default()
        },
        LineHeight::RelativeToFont(1.45),
        TextLayout::new(Justify::Left, LineBreak::WordBoundary),
        TextColor(DRAFT),
        Node {
            width: percent(100.),
            display: Display::None,
            ..default()
        },
        DraftText,
    )
}

fn paint_draft(
    desk: Res<Desk>,
    mut drafts: Query<(&mut Text, &mut Node), With<DraftText>>,
    mut placeholders: Query<&mut Node, (With<Placeholder>, Without<DraftText>)>,
) {
    let show = !desk.draft.is_empty();
    if let Ok((mut text, mut node)) = drafts.single_mut() {
        if text.0 != desk.draft {
            text.0 = desk.draft.clone();
        }
        set_shown(&mut node, show);
    }
    if let Ok(mut node) = placeholders.single_mut() {
        set_shown(&mut node, desk.story.is_none() && !show);
    }
}

fn set_shown(node: &mut Node, shown: bool) {
    let display = if shown { Display::Flex } else { Display::None };
    if node.display != display {
        node.display = display;
    }
}

fn sync_chrome(
    desk: Res<Desk>,
    mut status: Query<&mut Text, (With<StatusText>, Without<DetailText>, Without<ErrorText>)>,
    mut detail: Query<&mut Text, (With<DetailText>, Without<StatusText>, Without<ErrorText>)>,
    mut errors: Query<
        (&mut Text, &mut Node),
        (
            With<ErrorText>,
            Without<StatusText>,
            Without<DetailText>,
            Without<VolumeRow>,
        ),
    >,
    mut volume: Query<&mut Node, (With<VolumeRow>, Without<ErrorText>)>,
) {
    if desk.is_changed() {
        if let Ok(mut text) = status.single_mut() {
            text.0 = desk.status.clone();
        }
        if let Ok(mut text) = detail.single_mut() {
            text.0 = desk.detail.clone();
        }
        if let Ok((mut text, mut node)) = errors.single_mut() {
            if text.0 != desk.error {
                text.0 = desk.error.clone();
            }
            set_shown(&mut node, !desk.error.is_empty());
        }
    }
    let show_voice = desk.story.as_ref().is_some_and(FinishedStory::audible);
    if let Ok(mut node) = volume.single_mut() {
        set_shown(&mut node, show_voice);
    }
}

fn offer(desk: &Desk) -> PrimaryOffer {
    if !desk.loaded {
        PrimaryOffer::Warming
    } else if desk.busy {
        PrimaryOffer::Stop
    } else {
        PrimaryOffer::Write
    }
}

fn read_offer(desk: &Desk, playback: &Playback) -> Option<ReadOffer> {
    if !desk.loaded || desk.busy {
        return None;
    }
    if playback.autoplay && playback.paused {
        Some(ReadOffer::Continue)
    } else if playback.autoplay {
        Some(ReadOffer::Pause)
    } else if desk.story.as_ref().is_some_and(FinishedStory::audible) {
        Some(ReadOffer::Read)
    } else {
        None
    }
}

fn sync_button(
    desk: Res<Desk>,
    mut labels: Query<&mut Text, With<PrimaryLabel>>,
    mut buttons: Query<
        (
            Entity,
            &mut BackgroundColor,
            &mut BackgroundGradient,
            &mut Action,
            Option<&InteractionDisabled>,
        ),
        With<PrimaryButton>,
    >,
    mut commands: Commands,
) {
    let (words, fill, top, action, enabled) = match offer(&desk) {
        PrimaryOffer::Warming => ("Opening the book", RUST, RUST_TOP, Action::Begin, false),
        PrimaryOffer::Write => ("Write it", RUST, RUST_TOP, Action::Begin, true),
        PrimaryOffer::Stop => ("Stop", INK, READ_TOP, Action::Stop, true),
    };
    if let Ok(mut label) = labels.single_mut()
        && label.0 != words
    {
        label.0 = words.to_string();
    }
    let Ok((entity, mut background, mut gradient, mut current, disabled)) = buttons.single_mut()
    else {
        return;
    };
    if *current != action {
        *current = action;
        *gradient = warm_wash(LinearGradient::TO_BOTTOM, top, fill);
    }
    if background.0 != fill {
        background.0 = fill;
    }
    match (enabled, disabled.is_some()) {
        (true, true) => {
            commands.entity(entity).remove::<InteractionDisabled>();
        }
        (false, false) => {
            commands.entity(entity).insert(InteractionDisabled);
        }
        _ => {}
    }
}

fn sync_read(
    desk: Res<Desk>,
    playback: Res<Playback>,
    mut labels: Query<&mut Text, With<ReadLabel>>,
    mut buttons: Query<(&mut Node, &mut Action), With<ReadButton>>,
) {
    let next = read_offer(&desk, &playback);
    let Ok(mut label) = labels.single_mut() else {
        return;
    };
    let Ok((mut node, mut action)) = buttons.single_mut() else {
        return;
    };
    let (words, shown, next_action) = match next {
        None => ("Read it aloud", false, Action::Read),
        Some(ReadOffer::Read) => ("Read it aloud", true, Action::Read),
        Some(ReadOffer::Pause) => ("Pause", true, Action::Pause),
        Some(ReadOffer::Continue) => ("Continue", true, Action::Pause),
    };
    if label.0 != words {
        label.0 = words.to_string();
    }
    if *action != next_action {
        *action = next_action;
    }
    set_shown(&mut node, shown);
}

fn paint_progress(desk: Res<Desk>, mut fills: Query<&mut Node, With<ProgressFill>>) {
    let fraction = if desk.total == 0 {
        0.0
    } else {
        (desk.step as f32 / desk.total as f32).clamp(0.0, 1.0)
    };
    if let Ok(mut fill) = fills.single_mut() {
        let width = percent(fraction * 100.0);
        if fill.width != width {
            fill.width = width;
        }
    }
}

fn read_volume(
    slider: Query<&SliderValue, (With<VolumeSlider>, Changed<SliderValue>)>,
    mut volume: ResMut<VolumeLevel>,
) {
    if let Ok(value) = slider.single() {
        volume.0 = value.0.clamp(0.0, 1.0);
    }
}

fn pump_audio(
    mut commands: Commands,
    mut playback: ResMut<Playback>,
    desk: Res<Desk>,
    volume: Res<VolumeLevel>,
    mut assets: ResMut<Assets<Speech>>,
    players: Query<Entity, With<SpeechPlayer>>,
) {
    if let Some(player) = playback.player
        && !players.iter().any(|entity| entity == player)
    {
        playback.player = None;
        playback.cursor = playback.cursor.saturating_add(1);
    }
    if playback.paused || !playback.autoplay || playback.player.is_some() {
        return;
    }
    if playback.cursor >= desk.cue_count {
        playback.autoplay = false;
        return;
    }
    let Some(samples) = desk.clips.get(playback.cursor).and_then(Option::as_ref) else {
        return;
    };
    let handle = assets.add(Speech {
        samples: audio::with_gap(samples),
    });
    let mut settings = PlaybackSettings::DESPAWN;
    settings.volume = Volume::Linear(volume.0);
    let entity = commands
        .spawn((SpeechPlayer, AudioPlayer(handle), settings))
        .id();
    playback.player = Some(entity);
}

fn apply_volume(
    volume: Res<VolumeLevel>,
    mut sinks: Query<&mut AudioSink, With<SpeechPlayer>>,
    playback: Res<Playback>,
) {
    if volume.is_changed() {
        for mut sink in &mut sinks {
            sink.set_volume(Volume::Linear(volume.0));
        }
    }
    for sink in &mut sinks {
        if playback.paused && !sink.is_paused() {
            sink.pause();
        } else if !playback.paused && sink.is_paused() {
            sink.play();
        }
    }
}
