//! Load Spark-X2.6-1.7B with the attention LoRA and write a plan, then each beat.
//!
//! Generation stays on this thread so the Bevy window can keep drawing. The UI
//! sends one job at a time and reads progress off the channel between beats.

use std::path::PathBuf;

use bevy::ecs::resource::Resource;
use std::sync::mpsc::{Receiver, RecvError, Sender, TryRecvError};

use burn::module::Module;
use rollout_engine::{
    SamplingConfig, SparkTokenizer, Tokenizer, generate_with, load_story_sampling,
};
use spark_burn::{
    LoraSpec, SparkConfig, SparkModel, accelerator_name, attach_lora, bf16_device,
    load_lora_adapters, load_spark_checkpoint, sidecar_dir,
};

use crate::narrate::{self, Cue};
use crate::plan::{FinishedStory, StoryPlan, WrittenBeat};
use crate::prompts::{self, clean_completion, live_text};
use crate::qa::{self, QaReading};

#[derive(Clone, Debug, Resource)]
pub struct EngineConfig {
    pub checkpoint: PathBuf,
    pub plan_tokens: usize,
    pub beat_tokens: usize,
    pub qa_tokens: usize,
}

pub enum Command {
    Plan { premise: String },
    Write { plan: StoryPlan },
    Narrate { story: FinishedStory },
    Cancel,
}

/// What the worker is doing. The desk renders [`Phase::headline`] from this,
/// so progress copy cannot drift from the label that decides whether the
/// page is busy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Planning,
    PlanReady,
    Writing,
    Checking,
    Complete,
    DownloadingSpark,
    DownloadingVoice,
    LoadingVoice,
    Narrating,
    NarrationReady,
    Stopped,
    PlanUnreadable,
}

impl Phase {
    pub fn headline(self) -> &'static str {
        match self {
            Self::Planning => "Composing the plan.",
            Self::PlanReady | Self::Writing => "Writing.",
            Self::Checking => "Reviewing continuity.",
            Self::Complete => "The story is ready.",
            Self::DownloadingSpark => "Downloading Spark.",
            Self::DownloadingVoice => "Downloading Kokoro.",
            Self::LoadingVoice => "Preparing the voice.",
            Self::Narrating | Self::NarrationReady => "Reading aloud.",
            Self::Stopped => "Stopped.",
            Self::PlanUnreadable => "The plan could not be read.",
        }
    }

    pub fn is_work(self) -> bool {
        matches!(
            self,
            Self::Planning
                | Self::Writing
                | Self::Checking
                | Self::DownloadingSpark
                | Self::DownloadingVoice
                | Self::LoadingVoice
                | Self::Narrating
        )
    }
}

pub enum WorkerEvent {
    Ready {
        device: String,
    },
    Status {
        phase: Phase,
        detail: String,
        step: u32,
        total: u32,
    },
    PlanReady(StoryPlan),
    PlanUnparsed {
        text: String,
        error: String,
    },
    StoryReady(FinishedStory),
    Draft(String),
    Script(Vec<Cue>),
    Clip {
        index: usize,
        samples: Vec<f32>,
    },
    Failed(String),
}

struct Loaded {
    model: SparkModel,
    tokenizer: SparkTokenizer,
    directory: PathBuf,
}

pub fn serve(mut config: EngineConfig, jobs: Receiver<Command>, events: Sender<WorkerEvent>) {
    if let Err(error) = crate::fetch::prepare(&mut config.checkpoint, &events) {
        let _ = events.send(WorkerEvent::Failed(error));
        return;
    }
    let loaded = match load(&config) {
        Ok(loaded) => loaded,
        Err(error) => {
            let _ = events.send(WorkerEvent::Failed(error));
            return;
        }
    };
    let _ = events.send(WorkerEvent::Ready {
        device: accelerator_name().to_string(),
    });
    let mut speech: Option<Speech> = None;
    loop {
        let command = match jobs.recv() {
            Ok(command) => command,
            Err(RecvError) => return,
        };
        if matches!(command, Command::Cancel) {
            continue;
        }
        if let Err(error) = run_command(&config, &loaded, &jobs, &events, &mut speech, command) {
            let _ = events.send(WorkerEvent::Failed(error));
        }
    }
}

fn run_command(
    config: &EngineConfig,
    loaded: &Loaded,
    jobs: &Receiver<Command>,
    events: &Sender<WorkerEvent>,
    speech: &mut Option<Speech>,
    command: Command,
) -> Result<(), String> {
    match command {
        Command::Cancel => Ok(()),
        Command::Plan { premise } => plan(config, loaded, jobs, events, premise),
        Command::Write { plan } => write_story(config, loaded, jobs, events, plan),
        Command::Narrate { story } => narrate(speech, jobs, events, story),
    }
}

fn plan(
    config: &EngineConfig,
    loaded: &Loaded,
    jobs: &Receiver<Command>,
    events: &Sender<WorkerEvent>,
    premise: String,
) -> Result<(), String> {
    status(
        events,
        Phase::Planning,
        "Cast, setting, and the shape of the scenes",
        0,
        1,
    );
    if cancelled(jobs) {
        status(events, Phase::Stopped, "No plan was kept", 0, 1);
        return Ok(());
    }
    let sampling = load_story_sampling(&loaded.directory, config.plan_tokens)?;
    let text = match complete(
        loaded,
        jobs,
        events,
        &sampling,
        &prompts::plan_prompt(&premise),
        1,
    )? {
        Completion::Stopped => {
            status(events, Phase::Stopped, "No plan was kept", 0, 1);
            return Ok(());
        }
        Completion::Text(text) => text,
    };
    match crate::plan::parse_plan(&text) {
        Ok(mut story_plan) => {
            story_plan.premise = premise;
            status(events, Phase::PlanReady, "Writing the sections", 1, 1);
            events
                .send(WorkerEvent::PlanReady(story_plan))
                .map_err(|err| err.to_string())?;
        }
        Err(error) => {
            status(events, Phase::PlanUnreadable, &error, 1, 1);
            let _ = events.send(WorkerEvent::Draft(String::new()));
            events
                .send(WorkerEvent::PlanUnparsed { text, error })
                .map_err(|err| err.to_string())?;
        }
    }
    Ok(())
}

fn write_story(
    config: &EngineConfig,
    loaded: &Loaded,
    jobs: &Receiver<Command>,
    events: &Sender<WorkerEvent>,
    plan: StoryPlan,
) -> Result<(), String> {
    if plan.beats.is_empty() {
        return Err("the plan has no beats to write".into());
    }
    let sampling = load_story_sampling(&loaded.directory, config.beat_tokens)?;
    let total = plan.beats.len() as u32;
    let mut written = Vec::with_capacity(plan.beats.len());
    let mut prior = String::new();
    for (index, beat) in plan.beats.iter().enumerate() {
        let step = index as u32;
        if cancelled(jobs) {
            return stop_writing(events, &plan, &written, step, total);
        }
        status(
            events,
            Phase::Writing,
            &format!("Section {} of {total}. {}", index + 1, beat.display_title()),
            step,
            total,
        );
        let prompt = prompts::beat_prompt(&plan.premise, beat, &prior);
        let prose = match complete(loaded, jobs, events, &sampling, &prompt, (index as u64) + 2)? {
            Completion::Stopped => return stop_writing(events, &plan, &written, step, total),
            Completion::Text(prose) => prose,
        };
        if prose.trim().is_empty() {
            return Err(format!(
                "beat {} ({}) came back empty",
                index + 1,
                beat.kind.label()
            ));
        }
        if !prior.is_empty() {
            prior.push_str("\n\n");
        }
        prior.push_str(&prose);
        written.push(WrittenBeat {
            beat: beat.clone(),
            prose,
        });
        let _ = events.send(WorkerEvent::Draft(String::new()));
        let _ = emit_story(events, &plan, &written, false);
        status(
            events,
            Phase::Writing,
            &format!("Section {} of {total} is on the page", index + 1),
            step + 1,
            total,
        );
    }
    status(
        events,
        Phase::Checking,
        "Checking names, places, and the order of events",
        total,
        total,
    );
    let note = match check_story(config, loaded, jobs, events, &mut written) {
        Ok(Check::Note(note)) => note,
        Ok(Check::Stopped) => return stop_writing(events, &plan, &written, total, total),
        Err(error) => format!("The draft stands. {error}"),
    };
    status(events, Phase::Complete, &note, total, total);
    emit_story(events, &plan, &written, true)
}

fn emit_story(
    events: &Sender<WorkerEvent>,
    plan: &StoryPlan,
    beats: &[WrittenBeat],
    complete: bool,
) -> Result<(), String> {
    events
        .send(WorkerEvent::StoryReady(FinishedStory {
            premise: plan.premise.clone(),
            plan: plan.clone(),
            beats: beats.to_vec(),
            complete,
        }))
        .map_err(|err| err.to_string())
}

fn stop_writing(
    events: &Sender<WorkerEvent>,
    plan: &StoryPlan,
    written: &[WrittenBeat],
    step: u32,
    total: u32,
) -> Result<(), String> {
    status(
        events,
        Phase::Stopped,
        "Completed sections remain on the page",
        step,
        total,
    );
    emit_story(events, plan, written, false)
}

enum Check {
    Note(String),
    Stopped,
}

fn check_story(
    config: &EngineConfig,
    loaded: &Loaded,
    jobs: &Receiver<Command>,
    events: &Sender<WorkerEvent>,
    written: &mut [WrittenBeat],
) -> Result<Check, String> {
    if written.is_empty() {
        return Ok(Check::Note("There was nothing to review.".into()));
    }
    let sampling = load_story_sampling(&loaded.directory, config.qa_tokens)?;
    let draft = written
        .iter()
        .map(|beat| format!("title: {}\n{}", beat.beat.display_title(), beat.prose))
        .collect::<Vec<_>>()
        .join("\n\n");
    let verdict = match complete(
        loaded,
        jobs,
        events,
        &sampling,
        &prompts::qa_prompt(&draft, written.len()),
        written.len() as u64 + 40,
    )? {
        Completion::Stopped => return Ok(Check::Stopped),
        Completion::Text(text) => text,
    };
    let _ = events.send(WorkerEvent::Draft(String::new()));
    match qa::read_qa(&verdict, written.len()) {
        QaReading::Coherent => Ok(Check::Note("Continuity holds.".into())),
        QaReading::Revised(pieces) => {
            for (beat, prose) in written.iter_mut().zip(pieces) {
                beat.prose = prose;
            }
            Ok(Check::Note("Sections were revised for continuity.".into()))
        }
        QaReading::Unreadable => Ok(Check::Note(
            "The review was inconclusive. The draft stands.".into(),
        )),
    }
}

fn narrate(
    speech: &mut Option<Speech>,
    jobs: &Receiver<Command>,
    events: &Sender<WorkerEvent>,
    story: FinishedStory,
) -> Result<(), String> {
    status(events, Phase::LoadingVoice, "Loading the voice model", 0, 1);
    let speech = ready_speech(speech)?;
    if cancelled(jobs) {
        status(events, Phase::Stopped, "No audio was started", 0, 1);
        return Ok(());
    }
    let library = speech.engine.voices();
    let voices = narrate::voices_for_plan(&story.plan, &library)?;
    let cues = narrate::script(&story, &voices);
    if cues.is_empty() {
        return Err("the story has no prose to narrate".into());
    }
    let total = cues.len() as u32;
    events
        .send(WorkerEvent::Script(cues.clone()))
        .map_err(|err| err.to_string())?;
    for (index, cue) in cues.iter().enumerate() {
        if cancelled(jobs) {
            status(
                events,
                Phase::Stopped,
                "Lines already prepared can still be played",
                index as u32,
                total,
            );
            return Ok(());
        }
        status(
            events,
            Phase::Narrating,
            &format!(
                "Line {} of {total}: {} ({})",
                index + 1,
                cue.speaker,
                cue.voice
            ),
            index as u32,
            total,
        );
        let samples =
            speech
                .engine
                .synthesize_with_options(&cue.text, Some(&cue.voice), 1.0, 1.0, None)?;
        events
            .send(WorkerEvent::Clip { index, samples })
            .map_err(|err| err.to_string())?;
    }
    status(
        events,
        Phase::NarrationReady,
        "Nicole reads the narration. Each character has a distinct voice.",
        total,
        total,
    );
    Ok(())
}

struct Speech {
    engine: kokoro_micro::TtsEngine,
}

fn ready_speech(slot: &mut Option<Speech>) -> Result<&Speech, String> {
    if slot.is_none() {
        *slot = Some(Speech::open()?);
    }
    slot.as_ref()
        .ok_or_else(|| "voice model did not open".into())
}

impl Speech {
    fn open() -> Result<Self, String> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| err.to_string())?;
        let engine = runtime.block_on(kokoro_micro::TtsEngine::new())?;
        Ok(Self { engine })
    }
}

fn status(events: &Sender<WorkerEvent>, phase: Phase, detail: &str, step: u32, total: u32) {
    let _ = events.send(WorkerEvent::Status {
        phase,
        detail: detail.to_string(),
        step,
        total,
    });
}

fn cancelled(jobs: &Receiver<Command>) -> bool {
    match jobs.try_recv() {
        // A click that arrives while a job is running stops that job. The next
        // click starts a fresh one, so a queued Plan or Write is not resumed.
        Ok(
            Command::Cancel
            | Command::Plan { .. }
            | Command::Write { .. }
            | Command::Narrate { .. },
        ) => true,
        Err(TryRecvError::Empty) => false,
        Err(TryRecvError::Disconnected) => true,
    }
}

enum Completion {
    Text(String),
    Stopped,
}

fn complete(
    loaded: &Loaded,
    jobs: &Receiver<Command>,
    events: &Sender<WorkerEvent>,
    sampling: &SamplingConfig,
    prompt: &str,
    seed: u64,
) -> Result<Completion, String> {
    let tokens = loaded.tokenizer.encode(prompt);
    if tokens.is_empty() {
        return Err("the tokenizer produced no tokens for the prompt".into());
    }
    let mut fresh = Vec::new();
    let mut stopped = false;
    let generated = generate_with(&loaded.model, &tokens, *sampling, seed, |token| {
        fresh.push(token);
        let visible = live_text(&loaded.tokenizer.decode(&fresh));
        let _ = events.send(WorkerEvent::Draft(visible));
        if cancelled(jobs) {
            stopped = true;
            true
        } else {
            false
        }
    });
    if stopped {
        let _ = events.send(WorkerEvent::Draft(String::new()));
        return Ok(Completion::Stopped);
    }
    if generated.tokens.len() < tokens.len() {
        return Err("generation returned fewer tokens than the prompt".into());
    }
    let prose = loaded.tokenizer.decode(&generated.tokens[tokens.len()..]);
    Ok(Completion::Text(clean_completion(&prose)))
}

fn load(config: &EngineConfig) -> Result<Loaded, String> {
    let lora = lora_file();
    if !lora.is_file() {
        return Err(format!("Put the attention LoRA at {}.", lora.display()));
    }
    let directory = sidecar_dir(&config.checkpoint);
    let tokenizer_path = directory.join("tokenizer.json");
    if !tokenizer_path.is_file() {
        return Err(format!(
            "no tokenizer at {}. Download Spark with spark-story download, or pass --checkpoint.",
            tokenizer_path.display()
        ));
    }
    let spark = SparkConfig::load_sidecar(&config.checkpoint)?;
    let device = bf16_device(false)?;
    let model = SparkModel::new(&spark, &device);
    let model = load_spark_checkpoint(model, &config.checkpoint).map_err(|err| err.to_string())?;
    let model = attach_lora(model, LoraSpec::attention_default())?;
    let model = load_lora_adapters(model, &lora)?;
    let model = model.valid();
    let tokenizer = SparkTokenizer::open(&tokenizer_path)?;
    Ok(Loaded {
        model,
        tokenizer,
        directory,
    })
}

/// The attention adapter lives beside the story-desk sources.
fn lora_file() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lora/attention.safetensors")
}
