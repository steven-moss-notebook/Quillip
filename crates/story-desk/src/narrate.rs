//! Turn a finished story into narration cues.
//!
//! The narrator is Kokoro's `af_nicole`. Each character receives a different
//! English voice from the library, alternating American and British women and
//! men so two people in a dialogue do not share a timbre.

use kokoro_micro::Lang;

use crate::plan::{BeatKind, FinishedStory, StoryPlan, character_order};

pub const NARRATOR_VOICE: &str = "af_nicole";
pub const NARRATOR_NAME: &str = "Narrator";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
    pub speaker: String,
    pub voice: String,
    pub text: String,
}

/// Pair each character with a distinct English voice, leaving `af_nicole` for the narrator.
pub fn assign_voices(
    characters: &[String],
    library: &[String],
) -> Result<Vec<(String, String)>, String> {
    let mut buckets = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut extra = Vec::new();
    for voice in library {
        if voice == NARRATOR_VOICE {
            continue;
        }
        match Lang::from_voice(voice) {
            Some(Lang::AmericanEnglish | Lang::BritishEnglish) => {}
            _ => continue,
        }
        match voice.split('_').next().unwrap_or("") {
            "af" => buckets[0].push(voice.clone()),
            "am" => buckets[1].push(voice.clone()),
            "bf" => buckets[2].push(voice.clone()),
            "bm" => buckets[3].push(voice.clone()),
            _ => extra.push(voice.clone()),
        }
    }
    for bucket in &mut buckets {
        bucket.sort();
    }
    extra.sort();
    let available = buckets.iter().map(Vec::len).sum::<usize>() + extra.len();
    if characters.len() > available {
        return Err(format!(
            "this story has {} characters and Kokoro has {available} English voices besides {NARRATOR_VOICE}",
            characters.len()
        ));
    }
    if !library.iter().any(|voice| voice == NARRATOR_VOICE) {
        return Err(format!(
            "Kokoro's voice library has no {NARRATOR_VOICE} narrator voice"
        ));
    }

    let mut cursor = [0usize; 4];
    let mut extra_cursor = 0usize;
    let mut turn = 0usize;
    let mut assigned = Vec::with_capacity(characters.len());
    for name in characters {
        let mut voice = None;
        for _ in 0..4 {
            let bucket = turn % 4;
            turn += 1;
            if cursor[bucket] < buckets[bucket].len() {
                voice = Some(buckets[bucket][cursor[bucket]].clone());
                cursor[bucket] += 1;
                break;
            }
        }
        let voice = voice.unwrap_or_else(|| {
            let picked = extra[extra_cursor].clone();
            extra_cursor += 1;
            picked
        });
        assigned.push((name.clone(), voice));
    }
    Ok(assigned)
}

pub fn script(story: &FinishedStory, voices: &[(String, String)]) -> Vec<Cue> {
    let mut cues = Vec::new();
    for written in &story.beats {
        let prose = written.prose.trim();
        if prose.is_empty() {
            continue;
        }
        match written.beat.kind {
            BeatKind::Scene | BeatKind::CharacterBackground => {
                push_cue(&mut cues, NARRATOR_NAME, NARRATOR_VOICE, prose);
            }
            BeatKind::Monologue => {
                let speaker = written.beat.characters.first();
                let (name, voice) = speaker
                    .map(|name| (name.as_str(), voice_for(voices, name)))
                    .unwrap_or((NARRATOR_NAME, NARRATOR_VOICE));
                push_cue(&mut cues, name, voice, prose);
            }
            BeatKind::Dialogue | BeatKind::GroupConversation => {
                let turns = split_turns(prose, &written.beat.characters);
                if turns.iter().all(|(speaker, _)| speaker.is_none()) {
                    push_cue(&mut cues, NARRATOR_NAME, NARRATOR_VOICE, prose);
                    continue;
                }
                for (speaker, line) in turns {
                    match speaker {
                        Some(name) => {
                            push_cue(&mut cues, &name, voice_for(voices, &name), &line);
                        }
                        None => push_cue(&mut cues, NARRATOR_NAME, NARRATOR_VOICE, &line),
                    }
                }
            }
        }
    }
    cues
}

pub fn voices_for_plan(
    plan: &StoryPlan,
    library: &[String],
) -> Result<Vec<(String, String)>, String> {
    assign_voices(&character_order(plan), library)
}

fn voice_for<'a>(voices: &'a [(String, String)], name: &str) -> &'a str {
    voices
        .iter()
        .find(|(have, _)| have.eq_ignore_ascii_case(name))
        .map(|(_, voice)| voice.as_str())
        .unwrap_or(NARRATOR_VOICE)
}

fn push_cue(cues: &mut Vec<Cue>, speaker: &str, voice: &str, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    cues.push(Cue {
        speaker: speaker.to_string(),
        voice: voice.to_string(),
        text: text.to_string(),
    });
}

fn split_turns(text: &str, names: &[String]) -> Vec<(Option<String>, String)> {
    let mut turns: Vec<(Option<String>, String)> = Vec::new();
    let mut current: Option<String> = None;
    let mut buffer = String::new();
    let flush = |turns: &mut Vec<(Option<String>, String)>,
                 current: &Option<String>,
                 buffer: &mut String| {
        if buffer.trim().is_empty() {
            buffer.clear();
            return;
        }
        turns.push((current.clone(), buffer.trim().to_string()));
        buffer.clear();
    };
    for line in text.lines() {
        if let Some((name, rest)) = speaker_line(line, names) {
            flush(&mut turns, &current, &mut buffer);
            current = Some(name);
            buffer = rest;
        } else if buffer.is_empty() {
            buffer = line.to_string();
        } else {
            buffer.push('\n');
            buffer.push_str(line);
        }
    }
    flush(&mut turns, &current, &mut buffer);
    turns
}

fn speaker_line(line: &str, names: &[String]) -> Option<(String, String)> {
    let line = line.trim().trim_start_matches(['-', '*']).trim();
    let (head, rest) = line.split_once(':')?;
    let head = head.trim().trim_matches('*').trim();
    if head.is_empty() || head.len() > 48 || head.contains('.') {
        return None;
    }
    let matched = names.iter().find(|name| name.eq_ignore_ascii_case(head))?;
    Some((matched.clone(), rest.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{
        Beat, BeatKind, CastMember, FinishedStory, StoryPlan, WrittenBeat, character_order,
    };

    fn library() -> Vec<String> {
        [
            "af_nicole",
            "af_bella",
            "am_adam",
            "bf_emma",
            "bm_george",
            "zf_xiaoni",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }

    #[test]
    fn characters_receive_distinct_english_voices_and_nicole_stays_the_narrator() {
        let assigned = assign_voices(
            &["Mara".into(), "Ellis".into(), "Ned".into(), "Ivy".into()],
            &library(),
        )
        .unwrap();
        let voices: Vec<_> = assigned.iter().map(|(_, voice)| voice.as_str()).collect();
        assert_eq!(voices.len(), 4);
        assert!(voices.iter().all(|voice| *voice != NARRATOR_VOICE));
        assert_eq!(
            voices
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4
        );
        assert!(!voices.contains(&"zf_xiaoni"));
        assert_eq!(voices[0], "af_bella");
        assert_eq!(voices[1], "am_adam");
    }

    #[test]
    fn dialogue_lines_use_character_voices_and_scenes_use_nicole() {
        let story = FinishedStory {
            premise: "A ferry and a wrong chart.".into(),
            plan: StoryPlan {
                premise: String::new(),
                cast: vec![
                    CastMember {
                        name: "Mara".into(),
                        desire: String::new(),
                        arc: String::new(),
                    },
                    CastMember {
                        name: "Ellis".into(),
                        desire: String::new(),
                        arc: String::new(),
                    },
                ],
                beats: Vec::new(),
            },
            beats: vec![
                WrittenBeat {
                    beat: Beat {
                        kind: BeatKind::Scene,
                        title: "The pier".into(),
                        characters: vec!["Mara".into()],
                        arc: String::new(),
                        brief: String::new(),
                    },
                    prose: "The pier boards shone with rain.".into(),
                },
                WrittenBeat {
                    beat: Beat {
                        kind: BeatKind::Dialogue,
                        title: "The mark".into(),
                        characters: vec!["Mara".into(), "Ellis".into()],
                        arc: String::new(),
                        brief: String::new(),
                    },
                    prose: "Mara: The old mark still holds.\nEllis: Then why is the keel singing?"
                        .into(),
                },
            ],
            complete: true,
        };
        let voices = assign_voices(&character_order(&story.plan), &library()).unwrap();
        let cues = script(&story, &voices);
        assert_eq!(cues[0].voice, NARRATOR_VOICE);
        assert_eq!(cues[1].speaker, "Mara");
        assert_eq!(cues[1].voice, "af_bella");
        assert_eq!(cues[2].speaker, "Ellis");
        assert_eq!(cues[2].voice, "am_adam");
        assert_ne!(cues[1].voice, cues[2].voice);
    }
}
