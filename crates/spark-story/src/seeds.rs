//! Format seeds for self-supervised story rollouts.
//!
//! Each run tries to write 512 situations and appends the ones that come back
//! with text until reaching 128,000 seeds. The prompt does not plant names or
//! places. Situations that come back empty are left out. When a seed file
//! already exists, the bank is the new situations together with the saved
//! ones. A mini-batch draws up to 64 of them, taking each format equally and
//! walking that format's seeds without repeating one until the format has been
//! used through. Each seed asks for one literary format: a scene, a monologue,
//! a dialogue, a character background, or several people talking. The frozen
//! checkpoint scores the story on a rubric in one pass. The score is the
//! harmonic mean, so a weak criterion holds the result down. Stories of one
//! format are ranked together. The negative is the mean score of the two
//! formats a reader could confuse with this one, and each of those formats
//! counts once: a scene can be good writing and still fail as a dialogue.

use std::collections::BTreeSet;

use rlvr::{CpoConfig, GrpoConfig, group_advantages, group_centered, shape_advantage};

/// Situations written and appended on each training run.
pub const SEED_BANK: usize = 512;

/// Number of accumulated seeds where generation stops. Over time across runs
/// with new randomness sources, the process generates up to this cap of seeds
/// with distinct profiles.
pub const MAX_SEEDS: usize = 128_000;

/// Completion length sampled for each seed. The training window has to hold
/// this plus the seed prompt.
pub const GENERATED_TOKENS: usize = 1024;

/// How long a model-written situation may run before the story prompt.
pub const SEED_NEW_TOKENS: usize = 196;

/// Floor for each rubric component. A zero would make the harmonic mean collapse.
pub const SCORE_EPSILON: f32 = 1e-3;

const PER_KIND: usize = SEED_BANK / 5;
const EXTRA_KINDS: usize = SEED_BANK - PER_KIND * 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SeedKind {
    Scene,
    Dialogue,
    Monologue,
    CharacterBackground,
    GroupConversation,
}

impl SeedKind {
    pub const ALL: [SeedKind; 5] = [
        SeedKind::Scene,
        SeedKind::Dialogue,
        SeedKind::Monologue,
        SeedKind::CharacterBackground,
        SeedKind::GroupConversation,
    ];

    pub fn ordinal(self) -> usize {
        Self::ALL
            .iter()
            .position(|item| *item == self)
            .expect("kind is one of the five formats")
    }

    pub fn label(self) -> &'static str {
        match self {
            SeedKind::Scene => "scene",
            SeedKind::Dialogue => "dialogue",
            SeedKind::Monologue => "monologue",
            SeedKind::CharacterBackground => "character-background",
            SeedKind::GroupConversation => "group-conversation",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.label() == label)
    }

    /// Two formats a reader could confuse with this one.
    ///
    /// Each format is a negative for exactly two others, so every seed type
    /// pulls on the contrast the same number of times. A dialogue is trained
    /// against a scene and a monologue: a description or a single voice can be
    /// good writing and still not be two people talking.
    pub fn negatives(self) -> [SeedKind; 2] {
        match self {
            SeedKind::Scene => [SeedKind::CharacterBackground, SeedKind::Dialogue],
            SeedKind::Dialogue => [SeedKind::Scene, SeedKind::Monologue],
            SeedKind::Monologue => [SeedKind::GroupConversation, SeedKind::CharacterBackground],
            SeedKind::CharacterBackground => [SeedKind::Dialogue, SeedKind::GroupConversation],
            SeedKind::GroupConversation => [SeedKind::Monologue, SeedKind::Scene],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seed {
    pub index: usize,
    pub kind: SeedKind,
    pub prompt: String,
    pub atmosphere: String,
}

pub fn kind_quota(kind: SeedKind) -> usize {
    PER_KIND + usize::from(kind.ordinal() < EXTRA_KINDS)
}

/// Concrete, sensory-rich settings intentionally distributed across diverse domains.
///
/// Designed to prevent model mode-collapse into sombre, solitary offices or dim workshops,
/// providing varied physical contexts, sensory details, and environmental pressures.
pub const INTENTIONAL_SETTINGS: &[&str] = &[
    // Maritime, Waterfront & Coastal
    "a coastal ferry slip during a turbulent spring high tide",
    "the deck of a wooden fishing skiff hauling nets at sunrise",
    "a busy harbor pier with crates being unloaded from an island freighter",
    "a coastal oyster farming trestle exposed at low tide",
    "a canal lock station managing evening barge traffic",
    // Outdoor, Natural & Field
    "an apple orchard at dawn during the first autumn frost",
    "a sun-baked archaeological excavation trench in sandstone hills",
    "a high-altitude weather observation ridge in gusting winds",
    "a deep-woods forestry fire lookout tower overlooking a river valley",
    "a roadside honey apiary during late summer harvesting",
    // Urban, Transit & Public Life
    "a bustling open-air street produce and flower market on Saturday morning",
    "a municipal tram depot during a noisy morning shift change",
    "a central train station baggage and parcel sorting depot",
    "a public park pavilion hosting an energetic community chess tournament",
    "a rooftop community greenhouse during a sudden summer downpour",
    "a municipal lost-and-found repository during holiday travel week",
    // Culinary, Trade & Physical Craft (Active, Dynamic)
    "a commercial bakery kitchen an hour before pre-dawn opening",
    "a crowded seaside restaurant galley during a chaotic dinner rush",
    "an open-air stone masonry yard cutting architectural limestone",
    "a noisy printing pressroom running an urgent evening extra edition",
    "a boatyard drydock prepping a wooden hull for safety inspection",
    "an artisanal pottery yard firing a massive wood-burning kiln",
    "a bicycle repair shop busy on the eve of an annual road race",
    "a farm equipment staging barn preparing harvesters before incoming rain",
    // Civic, Cultural, Performance & Athletics
    "backstage in a community theater minutes before opening curtain",
    "a local radio broadcast booth during an unscripted live phone-in show",
    "an outdoor country auction ring bidding on farm equipment and tools",
    "a municipal natural history museum loading bay uncrating specimens",
    "a community brass band rehearsal hall during dress rehearsal",
    "a riverside rowing club boathouse preparing shells before a regatta",
    // Scientific, Technical & Observational
    "an amateur astronomical observatory dome awaiting a rare meteor cluster",
    "a university botanical conservatory during the overnight bloom of a rare cactus",
    "a field geology survey camp cataloging mineral core samples",
    "a rural electrical substation yard during an emergency ice-storm patrol",
];

/// Intentional tones and atmospheres countering default sombre, melancholy solitude.
pub const INTENTIONAL_ATMOSPHERES: &[&str] = &[
    "energetic and fast-paced, with rapid coordination under time pressure",
    "warm and comedic, marked by good-humored banter and witty misunderstandings",
    "brisk and practical, driven by crisp professional competence and trade pride",
    "vibrant and celebratory, full of festive anticipation and community excitement",
    "curious and inquisitive, focused on unraveling an unexpected practical puzzle",
    "tense and suspenseful, where a delicate balance of power or high stakes hang in the air",
    "playful and competitive, with spirited rivalry and trade gamesmanship",
    "warm and hospitable, centered on shared generosity and mutual relief",
    "gritty and resilient, showing stubborn good humor amidst dirt and mechanical setbacks",
    "absurd and satirical, wrestling with eccentric rules or bureaucratic mix-ups",
    "keen and alert, with breathless precision under close public observation",
    "spirited and defiant, fueled by determined teamwork against difficult odds",
];

fn format_dynamics(kind: SeedKind) -> &'static [&'static str] {
    match kind {
        SeedKind::Scene => &[
            "racing against a fast-approaching weather front or delivery deadline",
            "sorting and inspecting an unexpected shipment before the carrier departs",
            "working through a sudden mechanical failure while bystanders wait",
            "clearing an obstruction to keep an essential transport line open",
            "assembling an intricate physical apparatus under strict time limits",
            "securing vulnerable living stock or fragile goods against a sudden hazard",
            "calibrating sensitive gear while physical vibrations shake the floor",
            "handling a surge of demanding patrons or arrivals with scarce supplies",
        ],
        SeedKind::Dialogue => &[
            "two rival operators negotiating shared custody of a prime working space",
            "a veteran mentor and a headstrong apprentice debating a risky technique",
            "two workers arguing over an improvised substitution during an urgent rush",
            "an inspector and a stubborn manager contesting an ambiguous safety guideline",
            "two travelers realizing an unexpected personal connection between their journeys",
            "a technician explaining why a requested 'quick fix' is structurally impossible",
            "two partners scrambling to conceal an accidental mishap before a supervisor arrives",
            "two longtime peers bantering over an unsettled wager from last season",
        ],
        SeedKind::Monologue => &[
            "a speaker frantically ad-libbing and pacing when their partner is delayed",
            "an operator talking themselves through a precarious, step-by-step physical task",
            "a craftsperson critiquing their own work aloud while making final adjustments",
            "a competitive leader giving an energetic pep talk to their own reflection",
            "a watchkeeper calling out rapid navigation coordinates as conditions shift",
            "a host pitching today's eccentric offerings to an oncoming crowd",
            "a coordinator mentally rehearsing a difficult pitch while pacing the floor",
            "an enthusiast marveling aloud at the unexpected behavior of a specimen",
        ],
        SeedKind::CharacterBackground => &[
            "a former high-wire rigger who became a municipal bridge inspector",
            "a sea captain's daughter who established a master bakery in a mountain town",
            "a postal courier who spent four decades charting unmarked mountain paths",
            "an eccentric clockmaker who transitioned to restoring massive harbor beacons",
            "a retired theater performer who became a quick-witted auctioneer",
            "a deep-well diver who took over an inland family orchard in middle age",
            "a traveling artisan who restored storm-damaged community landmarks",
            "a self-taught mechanic who patented an ingenious municipal transit tool",
        ],
        SeedKind::GroupConversation => &[
            "three workers shouting fast coordination over the roar of heavy machinery",
            "a four-person crew debating an immediate workaround after a snapped cable",
            "stallholders arguing over the arrangement and boundaries of a festival layout",
            "a field team debating whether to pack up or press on as clouds gather",
            "a committee hurriedly reallocating scarce resources before an event starts",
            "a boat crew calling out simultaneous rudder and line positions in crosswinds",
            "a research group excitedly disputing the interpretation of a fresh finding",
            "a performance group arguing over tempo and entrance cues before taking the stage",
        ],
    }
}

pub fn setting_for(kind: SeedKind, index: usize) -> &'static str {
    let offset = (index * 7 + kind.ordinal() * 11) % INTENTIONAL_SETTINGS.len();
    INTENTIONAL_SETTINGS[offset]
}

pub fn atmosphere_for(kind: SeedKind, index: usize) -> &'static str {
    let offset = (index * 13 + kind.ordinal() * 5 + 3) % INTENTIONAL_ATMOSPHERES.len();
    INTENTIONAL_ATMOSPHERES[offset]
}

pub fn dynamic_for(kind: SeedKind, index: usize) -> &'static str {
    let dynamics = format_dynamics(kind);
    let offset = (index * 17 + kind.ordinal() * 3 + 1) % dynamics.len();
    dynamics[offset]
}

/// Prompt steering the frozen model toward an intentional, varied situation.
///
/// Anchors the situation in a concrete, non-monotonous setting, an active atmosphere,
/// and a format-tailored dynamic to avoid mode collapse into sombre solitude or uniform offices.
pub fn seed_request_guided(kind: SeedKind, index: usize) -> String {
    let setting = setting_for(kind, index);
    let atmosphere = atmosphere_for(kind, index);
    let dynamic = dynamic_for(kind, index);
    chat_user(&format!(
        "The piece will be a {label}. {brief}\n\nSetting: {setting}.\nAtmosphere: {atmosphere}.\nFocus: {dynamic}.\n\nInvent the situation yourself around this setting and focus. Do not use a famous character, a stock place, or a plot the reader would already know. Reply with one paragraph and then stop.\n\nSituation:",
        label = kind.label(),
        brief = situation_brief(kind),
    ))
}

/// Assignment text with no people, places, or objects filled in.
///
/// The frozen model writes the situation that follows this request. The
/// checkpoint is a chat model: the user text is wrapped so generation starts
/// at `<|Bot|></think>`, which is the published template with thinking off.
pub fn seed_request(kind: SeedKind) -> String {
    seed_request_guided(kind, 0)
}

/// Published Spark chat prompt with thinking disabled.
///
/// The assistant turn is closed with `</think>` so the completion is the
/// answer, not a reasoning trace. Special tokens are the checkpoint strings:
/// fullwidth bars on the sentence markers, ASCII bars on the role markers.
fn chat_user(body: &str) -> String {
    format!(
        "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>{body}<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>"
    )
}

pub fn seed(index: usize, kind: SeedKind, situation: impl Into<String>) -> Seed {
    let situation = situation.into();
    let atmosphere = atmosphere_for(kind, index).to_string();
    Seed {
        index,
        kind,
        prompt: chat_user(&format!(
            "{}\n\nSituation:\n{situation}\n\nWrite at least a thousand tokens. Stay inside this format for the whole piece.",
            instruction(kind)
        )),
        atmosphere,
    }
}

pub fn seed_with_atmosphere(
    index: usize,
    kind: SeedKind,
    situation: impl Into<String>,
    atmosphere: impl Into<String>,
) -> Seed {
    let situation = situation.into();
    let atmosphere = atmosphere.into();
    Seed {
        index,
        kind,
        prompt: chat_user(&format!(
            "{}\n\nSituation:\n{situation}\n\nWrite at least a thousand tokens. Stay inside this format for the whole piece.",
            instruction(kind)
        )),
        atmosphere,
    }
}

/// The 8 universal literary criteria shared across all story formats.
pub fn shared_criteria() -> &'static [&'static str] {
    &[
        "Concrete particularity, images and objects a reader can picture",
        "Causal sequence, later events following from earlier ones",
        "Consequence, a choice or action that changes what can happen next",
        "A distinct voice in the diction",
        "Urgency, tangible stakes, or interpersonal friction driving the action",
        "Thematic progression across the piece rather than emotional circling or rephrasing the same beat",
        "Controlled syntax, sentences that vary and stay intelligible",
        "An ending where the message, dynamic, or choice transforms rather than restating the opening mood",
    ]
}

/// The single defining criterion for a specific story format.
pub fn single_format_criterion(kind: SeedKind) -> &'static str {
    match kind {
        SeedKind::Scene => {
            "A physical place rendered through sensory action and events rather than dialogue transcript or life history"
        }
        SeedKind::Dialogue => {
            "Exactly two people speaking in alternating conversational turns, carried by speech rather than room description"
        }
        SeedKind::Monologue => {
            "A single interior voice exploring thought and feeling without a second voice answering"
        }
        SeedKind::CharacterBackground => {
            "The circumstances, history, and grounded desires of one person rather than a staged live scene"
        }
        SeedKind::GroupConversation => {
            "Three or more distinct speakers who answer one another rather than a single dominant voice"
        }
    }
}

/// 8 shared literary criteria plus 1 format-specific criterion (9 criteria total).
///
/// The scorer reads one digit, 0 through 5, after each line in a single forward pass.
pub fn criteria(kind: SeedKind) -> Vec<&'static str> {
    let mut lines = shared_criteria().to_vec();
    lines.push(single_format_criterion(kind));
    lines
}

/// Rubric criterion enforcing adherence to the assigned tone and atmosphere.
pub fn atmosphere_criterion(atmosphere: &str) -> String {
    format!(
        "Atmospheric alignment, capturing the assigned tone ({atmosphere}) rather than defaulting to passive melancholy or sombre solitude"
    )
}

/// 8 shared literary criteria, 1 format criterion, and 1 assigned atmosphere criterion.
pub fn criteria_with_atmosphere(kind: SeedKind, atmosphere: &str) -> Vec<String> {
    let mut lines: Vec<String> = shared_criteria().iter().map(|s| s.to_string()).collect();
    lines.push(single_format_criterion(kind).to_string());
    lines.push(atmosphere_criterion(atmosphere));
    lines
}

/// Specific rubric criteria for evaluating progression between an opening and an ending.
pub fn progression_rubric_criteria() -> Vec<&'static str> {
    vec![
        "The ending demonstrates a transformed situation, understanding, or dilemma compared to the opening",
        "The text avoids circular restatement of the initial emotion with varied synonyms",
        "A consequential change in relationship, stakes, or character stance occurs between beginning and end",
    ]
}

/// Slices the completion into opening and ending passages for contrast evaluation.
pub fn opening_and_ending(completion: &str, word_count: usize) -> (String, String) {
    let words: Vec<&str> = completion.split_whitespace().collect();
    if words.is_empty() {
        return (String::new(), String::new());
    }
    let count = (words.len() / 3).min(word_count).max(1);
    let opening = words[..count].join(" ");
    let ending = words[words.len() - count..].join(" ");
    (opening, ending)
}

/// Formats an opening-versus-ending comparison prompt for evaluating progression.
pub fn opening_ending_contrast_prompt(opening: &str, ending: &str) -> String {
    format!(
        "Compare the opening and ending of this scene:\n\n\
         Opening:\n{opening}\n\n\
         Ending:\n{ending}\n\n\
         Rate how decisively the narrative message and situation progressed from the opening to the ending."
    )
}

/// Combines the harmonic rubric score with an intra-story progression metric.
///
/// If a story receives a high rubric score but suffers from low progression
/// (e.g., circling the same beat), the score is penalized so circular stories
/// cannot dominate the advantages.
pub fn apply_progression_penalty(score: f32, progression: f32) -> f32 {
    let factor = (progression / 0.35).min(1.0).max(SCORE_EPSILON);
    (score * (0.65 + 0.35 * factor)).clamp(SCORE_EPSILON, 1.0)
}

pub fn rubric_preamble() -> &'static str {
    "\n\nRate each criterion with one digit from 0 to 5. 0 means absent. 5 means fully achieved."
}

/// Default exponent for the softened rubric power mean.
///
/// A value of `0.5` (the square-root power mean) softens the severe collapse
/// of the `-1.0` harmonic mean when a single criterion is imperfect, while still
/// punishing unbalanced or low-quality stories significantly more than an
/// arithmetic mean would.
pub const DEFAULT_POWER_MEAN_P: f32 = 0.5;

/// Generalized power mean of order `p`.
///
/// When `p = -1.0`, this equals the harmonic mean.
/// When `p = 0.5`, this produces a softened power mean that resists mode-collapse
/// caused by single-criterion over-penalization.
pub fn power_mean(scores: &[f32], p: f32) -> f32 {
    if scores.is_empty() {
        return SCORE_EPSILON;
    }
    if p.abs() < 1e-4 {
        let count = scores.len() as f32;
        let log_sum: f32 = scores.iter().map(|s| s.max(SCORE_EPSILON).ln()).sum();
        return (log_sum / count).exp().clamp(SCORE_EPSILON, 1.0);
    }
    let count = scores.len() as f32;
    let sum: f32 = scores
        .iter()
        .map(|score| score.max(SCORE_EPSILON).powf(p))
        .sum();
    (sum / count).powf(1.0 / p).clamp(SCORE_EPSILON, 1.0)
}

/// Softened rubric mean combining component scores using [`power_mean`] with [`DEFAULT_POWER_MEAN_P`].
pub fn softened_rubric_mean(scores: &[f32]) -> f32 {
    power_mean(scores, DEFAULT_POWER_MEAN_P)
}

/// Harmonic mean. Each component is clamped to [`SCORE_EPSILON`] first, so one
/// missed criterion pulls the score down without dividing by zero.
pub fn harmonic_mean(scores: &[f32]) -> f32 {
    if scores.is_empty() {
        return SCORE_EPSILON;
    }
    let count = scores.len() as f32;
    let reciprocal: f32 = scores
        .iter()
        .map(|score| 1.0 / score.max(SCORE_EPSILON))
        .sum();
    (count / reciprocal).clamp(SCORE_EPSILON, 1.0)
}

/// Draw up to `count` seeds, spread equally across the formats on hand.
///
/// One draw does not remember earlier draws. Training keeps a [`SeedStream`]
/// so later mini-batches continue through each format instead of repeating it.
pub fn sample_seeds(bank: &[Seed], draw: u64, count: usize) -> Vec<Seed> {
    SeedStream::new(bank, draw).next_batch(bank, count)
}

/// Walks the bank format by format.
///
/// Each mini-batch takes the same number of seeds from every format that is
/// present, give or take one when the batch does not divide evenly. The extra
/// slots rotate, so five batches use every format the same number of times.
/// A format is shuffled once and then consumed. It is shuffled again only
/// after its seeds have all been used, and a seed already taken in the current
/// batch is not drawn again until the others have been.
pub struct SeedStream {
    order: Vec<Vec<usize>>,
    cursor: Vec<usize>,
    epoch: Vec<u64>,
    salt: u64,
    batches: usize,
    rng: Rng,
}

impl SeedStream {
    pub fn new(bank: &[Seed], salt: u64) -> Self {
        let mut order = vec![Vec::new(); SeedKind::ALL.len()];
        for (position, seed) in bank.iter().enumerate() {
            order[seed.kind.ordinal()].push(position);
        }
        let mut stream = Self {
            order,
            cursor: vec![0; SeedKind::ALL.len()],
            epoch: vec![0; SeedKind::ALL.len()],
            salt,
            batches: 0,
            rng: Rng(salt),
        };
        for kind in 0..SeedKind::ALL.len() {
            shuffle(&mut stream.order[kind], &mut stream.rng);
        }
        stream
    }

    pub fn next_batch(&mut self, bank: &[Seed], count: usize) -> Vec<Seed> {
        if bank.is_empty() || count == 0 {
            return Vec::new();
        }
        let count = count.min(bank.len());
        let sizes: Vec<usize> = self.order.iter().map(|pool| pool.len()).collect();
        let take = allocate_kinds(&sizes, count, self.batches);
        self.batches = self.batches.wrapping_add(1);
        let mut chosen = Vec::with_capacity(count);
        for kind in 0..SeedKind::ALL.len() {
            for position in self.take_positions(kind, take[kind]) {
                chosen.push(bank[position].clone());
            }
        }
        shuffle_seeds(&mut chosen, &mut self.rng);
        chosen
    }

    fn take_positions(&mut self, kind: usize, need: usize) -> Vec<usize> {
        let pool_len = self.order[kind].len();
        if pool_len == 0 || need == 0 {
            return Vec::new();
        }
        let mut picked = Vec::with_capacity(need);
        while picked.len() < need {
            if self.cursor[kind] >= pool_len {
                self.reshuffle(kind);
                self.cursor[kind] = 0;
                if picked.len() < pool_len {
                    self.defer_picked(kind, &picked);
                }
            }
            let position = self.order[kind][self.cursor[kind]];
            self.cursor[kind] += 1;
            if picked.len() >= pool_len || !picked.contains(&position) {
                picked.push(position);
            }
        }
        picked
    }

    fn reshuffle(&mut self, kind: usize) {
        self.epoch[kind] = self.epoch[kind].wrapping_add(1);
        let mut rng = Rng(self
            .salt
            .wrapping_add(kind as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(self.epoch[kind]));
        shuffle(&mut self.order[kind], &mut rng);
    }

    /// Puts seeds already chosen in this batch at the back of a new pass.
    fn defer_picked(&mut self, kind: usize, picked: &[usize]) {
        let mut fresh = Vec::with_capacity(self.order[kind].len());
        let mut stale = Vec::new();
        for position in self.order[kind].iter().copied() {
            if picked.contains(&position) {
                stale.push(position);
            } else {
                fresh.push(position);
            }
        }
        fresh.append(&mut stale);
        self.order[kind] = fresh;
    }
}

/// How many seeds to take from each format.
///
/// `rotation` decides which formats receive the leftover slots. The draw never
/// asks a format for more seeds than it has; anything still missing is given
/// to formats that have some left.
fn allocate_kinds(pool_sizes: &[usize], count: usize, rotation: usize) -> Vec<usize> {
    let available: Vec<usize> = pool_sizes
        .iter()
        .enumerate()
        .filter_map(|(kind, size)| (*size > 0).then_some(kind))
        .collect();
    let mut take = vec![0_usize; pool_sizes.len()];
    if available.is_empty() || count == 0 {
        return take;
    }
    let count = count.min(pool_sizes.iter().sum());
    let base = count / available.len();
    let mut remainder = count % available.len();
    let start = rotation % available.len();
    for step in 0..available.len() {
        let kind = available[(start + step) % available.len()];
        let share = base + usize::from(remainder > 0);
        if remainder > 0 {
            remainder -= 1;
        }
        take[kind] = share.min(pool_sizes[kind]);
    }
    let mut left = count - take.iter().sum::<usize>();
    while left > 0 {
        let mut progressed = false;
        for step in 0..available.len() {
            let kind = available[(start + step) % available.len()];
            if take[kind] < pool_sizes[kind] {
                take[kind] += 1;
                left -= 1;
                progressed = true;
                if left == 0 {
                    break;
                }
            }
        }
        if !progressed {
            break;
        }
    }
    take
}

/// Story indexes and rubric ordinals the contrast has to score.
///
/// Every story is scored on its own format. Every story of a hard-negative
/// format is scored on this format as well.
pub fn rubric_jobs(kinds: &[SeedKind]) -> Vec<(usize, usize)> {
    let mut jobs = BTreeSet::new();
    for (index, kind) in kinds.iter().enumerate() {
        jobs.insert((index, kind.ordinal()));
        for negative in kind.negatives() {
            for (row, other) in kinds.iter().enumerate() {
                if *other == negative {
                    jobs.insert((row, kind.ordinal()));
                }
            }
        }
    }
    jobs.into_iter().collect()
}

/// Own rubric score, and the out-of-group baseline, for each row.
///
/// The baseline is the unweighted mean of the hard-negative formats' mean
/// scores on this row's rubric. A format with more stories does not outweigh
/// a format with fewer. Stories of the same format stay out of the baseline;
/// they are the in-group ranking in [`contrast_advantages`]. When neither
/// negative format is in the batch, the baseline equals the row's own score
/// and the margin is zero.
pub fn contrast_baselines(
    kinds: &[SeedKind],
    mut score: impl FnMut(usize, SeedKind) -> f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut own = Vec::with_capacity(kinds.len());
    let mut foreign = Vec::with_capacity(kinds.len());
    for (index, kind) in kinds.iter().enumerate() {
        let mine = score(index, *kind);
        let mut acc = 0.0;
        let mut used = 0_usize;
        for negative in kind.negatives() {
            let rows: Vec<usize> = kinds
                .iter()
                .enumerate()
                .filter_map(|(row, other)| (*other == negative).then_some(row))
                .collect();
            if rows.is_empty() {
                continue;
            }
            let mean = rows.iter().map(|row| score(*row, *kind)).sum::<f32>() / rows.len() as f32;
            acc += mean;
            used += 1;
        }
        own.push(mine);
        foreign.push(if used == 0 { mine } else { acc / used as f32 });
    }
    (own, foreign)
}

/// Compute own scores and out-of-group negative format baselines directly from each row's own score.
///
/// For each row, the baseline is the unweighted mean of the hard-negative formats' mean
/// scores on their respective rubrics. This avoids re-scoring stories on other formats.
pub fn contrast_baselines_from_own(kinds: &[SeedKind], own: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(kinds.len(), own.len());
    let mut foreign = Vec::with_capacity(kinds.len());
    for (index, kind) in kinds.iter().enumerate() {
        let mut acc = 0.0;
        let mut used = 0_usize;
        for negative in kind.negatives() {
            let rows: Vec<usize> = kinds
                .iter()
                .enumerate()
                .filter_map(|(row, other)| (*other == negative).then_some(row))
                .collect();
            if rows.is_empty() {
                continue;
            }
            let mean = rows.iter().map(|row| own[*row]).sum::<f32>() / rows.len() as f32;
            acc += mean;
            used += 1;
        }
        foreign.push(if used == 0 {
            own[index]
        } else {
            acc / used as f32
        });
    }
    (own.to_vec(), foreign)
}

/// Per-story advantage.
///
/// In-group stories of one format are ranked with GRPO on the rubric score.
/// The out-of-group margin is applied after that ranking. Folding the margin
/// into the group reward would cancel it, because every story in the format
/// shares the same negative baseline and GRPO subtracts the group mean. When
/// the group is tied, that margin is still a clipped advantage, so the batch
/// teaches the format boundary.
/// Within-format reward minus the format mean, before dividing by the standard deviation.
pub fn contrast_centered(kinds: &[SeedKind], own: &[f32]) -> Vec<f32> {
    assert_eq!(kinds.len(), own.len());
    let mut centered = vec![0.0_f32; kinds.len()];
    for kind in SeedKind::ALL {
        let rows: Vec<usize> = kinds
            .iter()
            .enumerate()
            .filter_map(|(index, row)| (*row == kind).then_some(index))
            .collect();
        if rows.is_empty() {
            continue;
        }
        let rewards: Vec<f32> = rows.iter().map(|index| own[*index]).collect();
        let scores = group_centered(&rewards);
        for (slot, index) in rows.into_iter().enumerate() {
            centered[index] = scores[slot];
        }
    }
    centered
}

pub fn contrast_advantages(kinds: &[SeedKind], own: &[f32], foreign: &[f32]) -> Vec<f32> {
    assert_eq!(kinds.len(), own.len());
    assert_eq!(kinds.len(), foreign.len());
    let config = CpoConfig {
        alpha: 1.0,
        zero_advantage_clip: 0.5,
        adjustment_clip: 0.5,
    };
    let mut advantages = vec![0.0_f32; kinds.len()];
    for kind in SeedKind::ALL {
        let rows: Vec<usize> = kinds
            .iter()
            .enumerate()
            .filter_map(|(index, row)| (*row == kind).then_some(index))
            .collect();
        if rows.is_empty() {
            continue;
        }
        let rewards: Vec<f32> = rows.iter().map(|index| own[*index]).collect();
        let grouped = group_advantages(&rewards, &GrpoConfig::default());
        for (slot, index) in rows.into_iter().enumerate() {
            let margin = own[index] - foreign[index];
            advantages[index] = shape_advantage(grouped[slot], margin, &config);
        }
    }
    advantages
}

/// Chart reward: how far this text's rubric score sits above the other format.
pub fn contrast_reward(own: f32, foreign: f32) -> f32 {
    own - foreign
}

fn situation_brief(kind: SeedKind) -> &'static str {
    match kind {
        SeedKind::Scene => "It stays in one place and shows what can be sensed and what people do.",
        SeedKind::Dialogue => "Exactly two people speak to each other in turns.",
        SeedKind::Monologue => "One person speaks or thinks, and nobody answers.",
        SeedKind::CharacterBackground => {
            "It tells one person's past and circumstances, not a live scene."
        }
        SeedKind::GroupConversation => "At least three people talk and answer one another.",
    }
}

pub(crate) fn instruction(kind: SeedKind) -> &'static str {
    match kind {
        SeedKind::Scene => {
            "Write a scene that stays in one place. Show what can be seen, heard, and touched, and what the people do. Narrate the action. Do not turn the piece into a transcript of dialogue, a speech by one voice, or a biography."
        }
        SeedKind::Dialogue => {
            "Write a dialogue between exactly two people. Give their spoken turns, one speaker then the other. A description of the room is not a dialogue, and a third speaker must not join."
        }
        SeedKind::Monologue => {
            "Write a monologue for one person. They speak or think alone, and the reader hears only that voice. No one answers, and do not cut away to other characters talking."
        }
        SeedKind::CharacterBackground => {
            "Write the background of one character: where they came from, what they have already done, who they have lost or kept, and what they want now. Tell it as history. Do not stage a live scene."
        }
        SeedKind::GroupConversation => {
            "Write a conversation among at least three people. Each person speaks in turn and answers the others. One voice alone is not enough, and a silent description of the room is not the piece."
        }
    }
}

fn shuffle(values: &mut [usize], rng: &mut Rng) {
    for index in (1..values.len()).rev() {
        let swap = rng.index(index + 1);
        values.swap(index, swap);
    }
}

fn shuffle_seeds(values: &mut [Seed], rng: &mut Rng) {
    for index in (1..values.len()).rev() {
        let swap = rng.index(index + 1);
        values.swap(index, swap);
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn index(&mut self, len: usize) -> usize {
        (self.next() as usize) % len
    }
}

fn word_set(text: &str) -> BTreeSet<&str> {
    text.split_whitespace().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_bank() -> Vec<Seed> {
        let mut bank = Vec::with_capacity(SEED_BANK);
        for kind in SeedKind::ALL {
            for number in 0..kind_quota(kind) {
                bank.push(seed(
                    bank.len(),
                    kind,
                    format!("Situation {number} invented for this assignment and no other."),
                ));
            }
        }
        bank
    }

    #[test]
    fn requests_ask_the_model_to_invent_the_situation() {
        assert_eq!(
            SeedKind::ALL
                .iter()
                .map(|kind| kind_quota(*kind))
                .sum::<usize>(),
            SEED_BANK
        );
        assert!(GENERATED_TOKENS >= 1000);
        assert_eq!(criteria(SeedKind::Dialogue).len(), 9);
        assert_eq!(shared_criteria().len(), 8);
        for kind in SeedKind::ALL {
            let request = seed_request(kind);
            assert!(request.contains("Invent the situation yourself"));
            assert!(request.contains("<|Bot|></think>"));
            assert!(request.starts_with("<｜start▁of▁sentence｜><|System|>"));
            assert!(!request.contains("Ilya"));
            assert!(!request.contains("river station"));
            assert!(!request.contains("glasshouse"));
        }
        let scene = seed(0, SeedKind::Scene, "one private conflict");
        let dialogue = seed(
            1,
            SeedKind::Dialogue,
            "two people who disagree about a debt",
        );
        assert!(scene.prompt.contains("Write a scene"));
        assert!(dialogue.prompt.contains("Write a dialogue"));
        assert!(
            word_set(&scene.prompt)
                .symmetric_difference(&word_set(&dialogue.prompt))
                .count()
                > 10
        );
    }

    #[test]
    fn harmonic_mean_is_pulled_down_by_one_weak_criterion() {
        let even = harmonic_mean(&[0.8, 0.8, 0.8]);
        let weak = harmonic_mean(&[0.8, 0.8, 0.05]);
        assert!(even > weak);
        assert!((even - 0.8).abs() < 1e-5);
        let clamped = harmonic_mean(&[0.0, 1.0]);
        assert!(clamped > 0.0 && clamped < 0.01);
    }

    #[test]
    fn power_mean_softens_weak_criterion_collapse() {
        let scores = [0.8, 0.8, 0.05];
        let h = harmonic_mean(&scores);
        let p_soft = power_mean(&scores, 0.5);
        let h_as_power = power_mean(&scores, -1.0);
        assert!((h - h_as_power).abs() < 1e-4);
        assert!(
            p_soft > h,
            "power mean p=0.5 ({p_soft}) should soften harmonic collapse ({h})"
        );
        let even = power_mean(&[0.8, 0.8, 0.8], 0.5);
        assert!((even - 0.8).abs() < 1e-4);
        assert!(even > p_soft);
    }

    #[test]
    fn criteria_with_atmosphere_includes_assigned_tone() {
        let crit = criteria_with_atmosphere(SeedKind::Dialogue, "warm and comedic");
        assert_eq!(crit.len(), 10);
        assert!(crit.iter().any(|c| c.contains("warm and comedic")));
        assert!(crit.iter().any(|c| c.contains("interpersonal friction")));
    }

    #[test]
    fn a_mini_batch_draws_64_seeds_and_keeps_every_format() {
        let bank = synthetic_bank();
        let batch = sample_seeds(&bank, 11, 64);
        assert_eq!(batch.len(), 64);
        let mut seen = BTreeSet::new();
        for seed in &batch {
            assert!(seen.insert(seed.index));
            assert!(seed.index < SEED_BANK);
        }
        let mut counts = [0_usize; SeedKind::ALL.len()];
        for seed in &batch {
            counts[seed.kind.ordinal()] += 1;
        }
        let low = *counts.iter().min().unwrap();
        let high = *counts.iter().max().unwrap();
        assert!(
            high - low <= 1,
            "formats were not drawn equally: {counts:?}"
        );
    }

    #[test]
    fn successive_batches_use_each_format_equally_without_repeating() {
        let bank = synthetic_bank();
        let mut stream = SeedStream::new(&bank, 7);
        let mut totals = [0_usize; SeedKind::ALL.len()];
        let mut seen = BTreeSet::new();
        for _ in 0..5 {
            let batch = stream.next_batch(&bank, 64);
            assert_eq!(batch.len(), 64);
            let mut counts = [0_usize; SeedKind::ALL.len()];
            for seed in &batch {
                counts[seed.kind.ordinal()] += 1;
                assert!(seen.insert(seed.index));
            }
            let low = *counts.iter().min().unwrap();
            let high = *counts.iter().max().unwrap();
            assert!(high - low <= 1);
            for (kind, count) in counts.into_iter().enumerate() {
                totals[kind] += count;
            }
        }
        assert!(totals.iter().all(|count| *count == totals[0]));
    }

    #[test]
    fn a_short_format_is_not_repeated_inside_one_batch() {
        let mut bank = Vec::new();
        for kind in [SeedKind::Scene, SeedKind::Dialogue] {
            for number in 0..5 {
                bank.push(seed(bank.len(), kind, format!("{number}")));
            }
        }
        let mut stream = SeedStream::new(&bank, 3);
        let first = stream.next_batch(&bank, 6);
        let second = stream.next_batch(&bank, 6);
        for batch in [&first, &second] {
            let mut seen = BTreeSet::new();
            assert!(batch.iter().all(|seed| seen.insert(seed.index)));
            for kind in [SeedKind::Scene, SeedKind::Dialogue] {
                assert_eq!(batch.iter().filter(|seed| seed.kind == kind).count(), 3);
            }
        }
    }

    #[test]
    fn a_shorter_bank_is_sampled_as_it_is() {
        let bank = vec![
            seed(9_000, SeedKind::Scene, "one"),
            seed(9_001, SeedKind::Scene, "two"),
            seed(9_002, SeedKind::Dialogue, "three"),
        ];
        let batch = sample_seeds(&bank, 3, 64);
        assert_eq!(batch.len(), 3);
        let mut seen = BTreeSet::new();
        for item in &batch {
            assert!(seen.insert(item.index));
            assert!(item.index >= 9_000);
        }
        assert!(batch.iter().any(|item| item.kind == SeedKind::Scene));
        assert!(batch.iter().any(|item| item.kind == SeedKind::Dialogue));

        let mut partial = synthetic_bank();
        partial.retain(|item| item.kind != SeedKind::Monologue);
        let filled = sample_seeds(&partial, 5, 64);
        assert_eq!(filled.len(), 64);
        assert!(filled.iter().all(|item| item.kind != SeedKind::Monologue));
        assert!(filled.iter().any(|item| item.kind == SeedKind::Scene));

        assert!(sample_seeds(&[], 1, 64).is_empty());
    }

    #[test]
    fn each_format_is_a_negative_for_exactly_two_others() {
        let mut hits = [0_usize; SeedKind::ALL.len()];
        for kind in SeedKind::ALL {
            let negatives = kind.negatives();
            assert_ne!(negatives[0], negatives[1]);
            assert!(negatives.iter().all(|negative| *negative != kind));
            for negative in negatives {
                hits[negative.ordinal()] += 1;
            }
        }
        assert_eq!(hits, [2, 2, 2, 2, 2]);
    }

    #[test]
    fn out_of_group_baselines_weight_each_negative_format_once() {
        let kinds = [
            SeedKind::Dialogue,
            SeedKind::Dialogue,
            SeedKind::Scene,
            SeedKind::Scene,
            SeedKind::Scene,
            SeedKind::Monologue,
        ];
        let jobs: BTreeSet<_> = rubric_jobs(&kinds).into_iter().collect();
        let mut asked = BTreeSet::new();
        // Dialogue's negatives are a scene and a monologue. Three scenes at 0
        // and one monologue at 0.9 must count as two formats, not four stories.
        let (own, foreign) = contrast_baselines(&kinds, |row, kind| {
            asked.insert((row, kind.ordinal()));
            if kind != SeedKind::Dialogue {
                return 0.0;
            }
            match (kinds[row], row) {
                (SeedKind::Dialogue, 0) => 0.9,
                (SeedKind::Dialogue, _) => 0.3,
                (SeedKind::Scene, _) => 0.0,
                (SeedKind::Monologue, _) => 0.9,
                _ => 0.0,
            }
        });
        assert_eq!(asked, jobs);
        assert!((own[0] - 0.9).abs() < 1e-5);
        assert!((own[1] - 0.3).abs() < 1e-5);
        assert!((foreign[0] - 0.45).abs() < 1e-5);
        assert!((foreign[1] - foreign[0]).abs() < 1e-5);
        let advantages = contrast_advantages(&kinds, &own, &foreign);
        assert!(advantages[0] > advantages[1]);
        assert!(advantages[0] > 0.0);

        let own_scores = [0.9, 0.3, 0.0, 0.0, 0.0, 0.9];
        let (own_direct, foreign_direct) = contrast_baselines_from_own(&kinds, &own_scores);
        assert_eq!(own_direct, own_scores);
        assert!((foreign_direct[0] - 0.45).abs() < 1e-5);
        assert!((foreign_direct[1] - 0.45).abs() < 1e-5);

        let alone = [SeedKind::Dialogue, SeedKind::Dialogue];
        let (own, foreign) = contrast_baselines(&alone, |_, _| 0.7);
        assert!((foreign[0] - own[0]).abs() < 1e-6);

        let flat = contrast_advantages(
            &[SeedKind::Dialogue, SeedKind::Dialogue],
            &[1.0, 1.0],
            &[-1.0, -1.0],
        );
        assert!(flat[0] > 0.2 && flat[1] > 0.2);

        let missed = contrast_advantages(
            &[SeedKind::Dialogue, SeedKind::Dialogue],
            &[-1.0, -1.0],
            &[-1.0, -1.0],
        );
        assert!(missed.iter().all(|advantage| advantage.abs() < 1e-4));
        assert!(contrast_reward(0.8, 0.45) > contrast_reward(0.3, 0.45));
    }

    #[test]
    fn opening_and_ending_progression_tools_and_penalties() {
        let text = "One two three four five six seven eight nine ten eleven twelve.";
        let (opening, ending) = opening_and_ending(text, 3);
        assert_eq!(opening, "One two three");
        assert_eq!(ending, "ten eleven twelve.");

        let prompt = opening_ending_contrast_prompt(&opening, &ending);
        assert!(prompt.contains("Opening:\nOne two three"));
        assert!(prompt.contains("Ending:\nten eleven twelve."));
        assert_eq!(progression_rubric_criteria().len(), 3);

        let high_progression_score = apply_progression_penalty(0.8, 0.7);
        let low_progression_score = apply_progression_penalty(0.8, 0.1);
        assert!((high_progression_score - 0.8).abs() < 1e-5);
        assert!(low_progression_score < high_progression_score - 0.15);
    }

    #[test]
    fn intentional_seed_requests_vary_settings_atmospheres_and_dynamics() {
        assert!(INTENTIONAL_SETTINGS.len() >= 30);
        assert!(INTENTIONAL_ATMOSPHERES.len() >= 10);

        // Ensure settings deliberately avoid generic monotonous office or workshop tropes
        for setting in INTENTIONAL_SETTINGS {
            assert!(
                !setting.contains("office room"),
                "setting {setting} contains office room"
            );
            assert!(
                !setting.contains("dimly lit workshop"),
                "setting {setting} contains workshop"
            );
        }

        // Check that different indices produce different requests with distinct settings
        let mut seen_settings = BTreeSet::new();
        let mut seen_atmospheres = BTreeSet::new();
        for i in 0..50 {
            let setting = setting_for(SeedKind::Scene, i);
            let atmosphere = atmosphere_for(SeedKind::Scene, i);
            seen_settings.insert(setting);
            seen_atmospheres.insert(atmosphere);

            let request = seed_request_guided(SeedKind::Scene, i);
            assert!(request.contains(setting));
            assert!(request.contains(atmosphere));
            assert!(request.contains("Setting:"));
            assert!(request.contains("Atmosphere:"));
            assert!(request.contains("Focus:"));
            assert!(request.contains("Invent the situation yourself"));
        }

        // 50 draws must cover high variety across the banks
        assert!(
            seen_settings.len() >= 25,
            "insufficient setting variety: {}",
            seen_settings.len()
        );
        assert!(
            seen_atmospheres.len() >= 10,
            "insufficient atmosphere variety: {}",
            seen_atmospheres.len()
        );
    }
}
