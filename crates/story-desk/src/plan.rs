//! A short-story plan as a cast and an ordered series of literary beats.
//!
//! Character backgrounds arrive when someone is introduced. Later beats are
//! scenes, dialogues, monologues, or group conversations, and each one says
//! which part of a character arc it explores.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeatKind {
    Scene,
    Dialogue,
    Monologue,
    CharacterBackground,
    GroupConversation,
}

impl BeatKind {
    pub const ALL: [BeatKind; 5] = [
        BeatKind::CharacterBackground,
        BeatKind::Scene,
        BeatKind::Dialogue,
        BeatKind::Monologue,
        BeatKind::GroupConversation,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Scene => "Scene description",
            Self::Dialogue => "Dialogue",
            Self::Monologue => "Monologue",
            Self::CharacterBackground => "Character background",
            Self::GroupConversation => "Group conversation",
        }
    }

    pub fn wire(self) -> &'static str {
        match self {
            Self::Scene => "scene",
            Self::Dialogue => "dialogue",
            Self::Monologue => "monologue",
            Self::CharacterBackground => "character-background",
            Self::GroupConversation => "group-conversation",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim().to_ascii_lowercase().replace('_', "-");
        let text = text.replace("scene description", "scene");
        let text = text.replace("character background", "character-background");
        let text = text.replace("group conversation", "group-conversation");
        Self::ALL.into_iter().find(|kind| kind.wire() == text)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CastMember {
    pub name: String,
    pub desire: String,
    pub arc: String,
}

impl CastMember {
    fn blank() -> Self {
        Self {
            name: String::new(),
            desire: String::new(),
            arc: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Beat {
    pub kind: BeatKind,
    pub title: String,
    pub characters: Vec<String>,
    pub arc: String,
    pub brief: String,
}

impl Beat {
    pub fn empty(kind: BeatKind) -> Self {
        Self {
            kind,
            title: String::new(),
            characters: Vec::new(),
            arc: String::new(),
            brief: String::new(),
        }
    }

    /// Title the reader sees. An empty title falls back to the beat's form.
    pub fn display_title(&self) -> &str {
        let title = self.title.trim();
        if title.is_empty() {
            self.kind.label()
        } else {
            title
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoryPlan {
    pub premise: String,
    pub cast: Vec<CastMember>,
    pub beats: Vec<Beat>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenBeat {
    pub beat: Beat,
    pub prose: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishedStory {
    pub premise: String,
    pub plan: StoryPlan,
    pub beats: Vec<WrittenBeat>,
    pub complete: bool,
}

impl FinishedStory {
    pub fn audible(&self) -> bool {
        self.complete && !self.beats.is_empty()
    }
}

/// Names in cast order, then anyone a beat adds later.
pub fn character_order(plan: &StoryPlan) -> Vec<String> {
    let mut names = Vec::new();
    for member in &plan.cast {
        push_unique(&mut names, &member.name);
    }
    for beat in &plan.beats {
        for name in &beat.characters {
            push_unique(&mut names, name);
        }
    }
    names
}

fn push_unique(names: &mut Vec<String>, name: &str) {
    let name = name.trim();
    if name.is_empty() {
        return;
    }
    if names.iter().any(|have| have.eq_ignore_ascii_case(name)) {
        return;
    }
    names.push(name.to_string());
}

pub fn warnings(plan: &StoryPlan) -> Vec<String> {
    let mut notes = Vec::new();
    let mut introduced: Vec<String> = Vec::new();
    if plan.beats.is_empty() {
        notes.push("The plan has no beats yet.".into());
    }
    for member in &plan.cast {
        if member.name.trim().is_empty() {
            notes.push("A cast row is missing a name.".into());
        } else if member.arc.trim().is_empty() {
            notes.push(format!("{} has no character arc.", member.name.trim()));
        }
    }
    for (index, beat) in plan.beats.iter().enumerate() {
        let place = index + 1;
        if beat.brief.trim().is_empty() {
            notes.push(format!("Beat {place} has an empty situation."));
        }
        match beat.kind {
            BeatKind::Dialogue if beat.characters.len() != 2 => {
                notes.push(format!(
                    "Beat {place} is a dialogue and needs exactly two people."
                ));
            }
            BeatKind::GroupConversation if beat.characters.len() < 3 => {
                notes.push(format!(
                    "Beat {place} is a group conversation and needs at least three people."
                ));
            }
            BeatKind::Monologue | BeatKind::CharacterBackground if beat.characters.len() != 1 => {
                notes.push(format!(
                    "Beat {place} ({}) needs exactly one person.",
                    beat.kind.label()
                ));
            }
            _ => {}
        }
        if beat.kind == BeatKind::CharacterBackground {
            for name in &beat.characters {
                push_unique(&mut introduced, name);
            }
        } else {
            for name in &beat.characters {
                if !introduced
                    .iter()
                    .any(|have| have.eq_ignore_ascii_case(name))
                {
                    notes.push(format!(
                        "{name} appears in beat {place} before a character background."
                    ));
                    push_unique(&mut introduced, name);
                }
            }
        }
    }
    notes
}

pub fn parse_plan(text: &str) -> Result<StoryPlan, String> {
    let mut draft = Draft::new();
    for raw in text.lines() {
        draft.take_line(raw);
    }
    draft.finish();
    if draft.beats.is_empty() {
        let snippet: String = text.chars().take(180).collect();
        return Err(format!(
            "the model did not return any BEAT blocks. The reply began: {snippet}"
        ));
    }
    Ok(StoryPlan {
        premise: String::new(),
        cast: draft.cast,
        beats: draft.beats,
    })
}

struct Draft {
    cast: Vec<CastMember>,
    beats: Vec<Beat>,
    mode: Mode,
    beat: Option<Beat>,
    member: Option<CastMember>,
    field: Field,
}

impl Draft {
    fn new() -> Self {
        Self {
            cast: Vec::new(),
            beats: Vec::new(),
            mode: Mode::None,
            beat: None,
            member: None,
            field: Field::None,
        }
    }

    fn finish(&mut self) {
        self.flush_member();
        self.flush_beat();
    }

    fn flush_member(&mut self) {
        if let Some(person) = self.member.take()
            && !person.name.trim().is_empty()
        {
            self.cast.push(person);
        }
    }

    fn flush_beat(&mut self) {
        if let Some(beat) = self.beat.take()
            && (!beat.brief.trim().is_empty() || !beat.title.trim().is_empty())
        {
            self.beats.push(beat);
        }
    }

    fn begin_section(&mut self, mode: Mode) {
        self.flush_beat();
        self.flush_member();
        self.mode = mode;
        self.field = Field::None;
        if mode == Mode::Beat {
            self.beat = Some(Beat::empty(BeatKind::Scene));
        }
    }

    fn take_line(&mut self, raw: &str) {
        let line = raw.trim();
        if line.is_empty() {
            if self.field == Field::Brief
                && let Some(beat) = self.beat.as_mut()
            {
                beat.brief.push('\n');
            }
            return;
        }
        let heading = section_heading(line);
        if heading.eq_ignore_ascii_case("CAST") {
            self.begin_section(Mode::Cast);
            return;
        }
        if heading.eq_ignore_ascii_case("BEAT") || heading.to_ascii_lowercase().starts_with("beat ")
        {
            self.begin_section(Mode::Beat);
            return;
        }
        match self.mode {
            Mode::Cast => self.take_cast_line(line),
            Mode::Beat => self.take_beat_line(line),
            Mode::None => {}
        }
    }

    fn take_cast_line(&mut self, line: &str) {
        if let Some(person) = cast_bar_line(line) {
            self.flush_member();
            self.cast.push(person);
            self.field = Field::None;
            return;
        }
        let Some((key, value)) = split_field(line) else {
            append_field(self.field, &mut self.member, &mut self.beat, line);
            return;
        };
        match key.as_str() {
            "name" => {
                self.flush_member();
                self.member = Some(CastMember {
                    name: value,
                    ..CastMember::blank()
                });
                self.field = Field::Name;
            }
            "desire" | "want" | "wants" => {
                self.member.get_or_insert_with(CastMember::blank).desire = value;
                self.field = Field::Desire;
            }
            "arc" => {
                self.member.get_or_insert_with(CastMember::blank).arc =
                    value.trim_start_matches("arc:").trim().to_string();
                self.field = Field::Arc;
            }
            _ => append_field(self.field, &mut self.member, &mut self.beat, line),
        }
    }

    fn take_beat_line(&mut self, line: &str) {
        let field = self.field;
        let beat = self
            .beat
            .get_or_insert_with(|| Beat::empty(BeatKind::Scene));
        let Some((key, value)) = split_field(line) else {
            append_beat_line(field, beat, line);
            return;
        };
        let next = match key.as_str() {
            "kind" | "type" | "form" => {
                if let Some(kind) = BeatKind::parse(&value) {
                    beat.kind = kind;
                }
                Field::Kind
            }
            "title" => {
                beat.title = value;
                Field::Title
            }
            "characters" | "character" | "people" | "speakers" => {
                beat.characters = split_names(&value);
                Field::Characters
            }
            "arc" => {
                beat.arc = value;
                Field::Arc
            }
            "brief" | "situation" | "focus" => {
                beat.brief = value;
                Field::Brief
            }
            _ => {
                append_beat_line(field, beat, line);
                return;
            }
        };
        self.field = next;
    }
}

fn section_heading(line: &str) -> &str {
    line.trim_start_matches(['#', '*', '-', ' '])
        .trim()
        .trim_end_matches(':')
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    None,
    Cast,
    Beat,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Field {
    None,
    Name,
    Desire,
    Arc,
    Kind,
    Title,
    Characters,
    Brief,
}

fn split_field(line: &str) -> Option<(String, String)> {
    let (key, value) = line.split_once(':')?;
    let key = key.trim().trim_start_matches(['-', '*']).trim();
    if key.is_empty()
        || key.len() > 24
        || key.contains(' ') && !key.eq_ignore_ascii_case("character arc")
    {
        return None;
    }
    let key = if key.eq_ignore_ascii_case("character arc") {
        "arc".to_string()
    } else {
        key.to_ascii_lowercase()
    };
    Some((key, value.trim().to_string()))
}

fn cast_bar_line(line: &str) -> Option<CastMember> {
    let line = line.trim().trim_start_matches(['-', '*']).trim();
    if !line.contains('|') {
        return None;
    }
    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
    let name = parts.first()?.trim();
    if name.is_empty() || name.contains(':') {
        return None;
    }
    let desire = parts.get(1).copied().unwrap_or("").to_string();
    let arc = parts
        .get(2)
        .copied()
        .unwrap_or("")
        .trim_start_matches("arc:")
        .trim()
        .to_string();
    Some(CastMember {
        name: name.to_string(),
        desire,
        arc,
    })
}

fn split_names(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for part in text.split(|ch| matches!(ch, ',' | ';' | '\n')) {
        let part = part.trim().trim_matches(['.', ' ']);
        if part.is_empty() {
            continue;
        }
        push_unique(&mut names, part);
    }
    names
}

fn append_field(
    field: Field,
    member: &mut Option<CastMember>,
    beat: &mut Option<Beat>,
    line: &str,
) {
    let Some(person) = member.as_mut() else {
        if let Some(beat) = beat.as_mut() {
            append_beat_line(field, beat, line);
        }
        return;
    };
    let slot = match field {
        Field::Desire => &mut person.desire,
        Field::Arc => &mut person.arc,
        Field::Name => &mut person.name,
        _ => return,
    };
    if !slot.is_empty() {
        slot.push(' ');
    }
    slot.push_str(line.trim());
}

fn append_beat_line(field: Field, beat: &mut Beat, line: &str) {
    let slot = match field {
        Field::Brief => &mut beat.brief,
        Field::Arc => &mut beat.arc,
        Field::Title => &mut beat.title,
        Field::Characters => {
            for name in split_names(line) {
                push_unique(&mut beat.characters, &name);
            }
            return;
        }
        _ => return,
    };
    if !slot.is_empty() && !slot.ends_with('\n') {
        slot.push(' ');
    }
    slot.push_str(line.trim());
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
CAST
- Mara Voss | keep the ferry on the old chart | arc: certain of the channel -> loses the tide -> trusts a shared reading
- Ellis Quinn | prove the new sounding | arc: eager to be right -> admits a misread mark -> keeps the correction

BEAT
kind: character-background
title: Mara's channel
characters: Mara Voss
arc: establishes her certainty
brief: Mara learned the harbor from her aunt and has never missed a dawn sailing.

BEAT
kind: dialogue
title: The disputed mark
characters: Mara Voss, Ellis Quinn
arc: Ellis's certainty cracks
brief: On the wet pier they argue over a penciled correction before the freight whistle.
";

    #[test]
    fn a_plan_keeps_the_cast_the_beats_and_their_kinds() {
        let plan = parse_plan(SAMPLE).unwrap();
        assert_eq!(plan.cast.len(), 2);
        assert_eq!(plan.cast[0].name, "Mara Voss");
        assert!(plan.cast[1].arc.contains("misread"));
        assert_eq!(plan.beats.len(), 2);
        assert_eq!(plan.beats[0].display_title(), "Mara's channel");
        assert_eq!(plan.beats[0].kind, BeatKind::CharacterBackground);
        assert_eq!(plan.beats[1].kind, BeatKind::Dialogue);
        assert_eq!(
            plan.beats[1].characters,
            vec!["Mara Voss".to_string(), "Ellis Quinn".to_string()]
        );
    }

    #[test]
    fn a_speaker_without_a_background_is_noted() {
        let plan = parse_plan(SAMPLE).unwrap();
        let notes = warnings(&plan);
        assert!(
            notes.iter().any(|note| note.contains("Ellis Quinn")),
            "{notes:?}"
        );
        assert!(!notes.iter().any(|note| note.contains("Mara Voss appears")));
    }
}
