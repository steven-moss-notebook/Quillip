//! Prompts for the Spark chat template the attention LoRA was trained on.
//!
//! The format instructions are the ones in `spark-story`'s seed bank. A scene,
//! a dialogue, a monologue, a character background, and a group conversation
//! each keep that wording so the adapter still recognizes the form.

use crate::plan::BeatKind;

/// Published Spark chat prompt with thinking disabled.
///
/// The assistant turn is closed with `</think>` so the completion is the
/// answer. Special tokens match the checkpoint: fullwidth bars on the sentence
/// markers, ASCII bars on the role markers.
pub fn chat_user(body: &str) -> String {
    format!(
        "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>{body}<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>"
    )
}

pub fn instruction(kind: BeatKind) -> &'static str {
    match kind {
        BeatKind::Scene => {
            "Write a scene that stays in one place. Show what can be seen, heard, and touched, and what the people do. Narrate the action. Do not turn the piece into a transcript of dialogue, a speech by one voice, or a biography."
        }
        BeatKind::Dialogue => {
            "Write a dialogue between exactly two people. Give their spoken turns, one speaker then the other. A description of the room is not a dialogue, and a third speaker must not join."
        }
        BeatKind::Monologue => {
            "Write a monologue for one person. They speak or think alone, and the reader hears only that voice. No one answers, and do not cut away to other characters talking."
        }
        BeatKind::CharacterBackground => {
            "Write the background of one character: where they came from, what they have already done, who they have lost or kept, and what they want now. Tell it as history. Do not stage a live scene."
        }
        BeatKind::GroupConversation => {
            "Write a conversation among at least three people. Each person speaks in turn and answers the others. One voice alone is not enough, and a silent description of the room is not the piece."
        }
    }
}

pub fn plan_prompt(premise: &str) -> String {
    chat_user(&format!(
        "Plan a short story. Do not write the story yet.\n\n\
         Premise:\n{premise}\n\n\
         If the premise is only a mood or a kind of story, invent the people, the place, and the trouble yourself.\n\n\
         Split the plan into an ordered series of beats. Use only these kinds:\n\
         - character-background: the circumstances, history, and present desire of one person. Put this beat when that person is introduced, before they speak or enter a scene.\n\
         - scene: a scene description of one place, told through what can be seen, heard, touched, and done.\n\
         - dialogue: exactly two people speaking in alternating turns.\n\
         - monologue: one person speaking or thinking alone.\n\
         - group-conversation: three or more people who answer one another.\n\n\
         Give every person a character arc, from the stance they hold at the start, through the turn that changes them, to where they stand at the end. \
         Each beat says which part of an arc it explores.\n\n\
         Reply in this shape and nothing else. Use six to ten beats.\n\n\
         CAST\n\
         - First Last | what they want now | arc: beginning -> turn -> end\n\
         BEAT\n\
         kind: character-background\n\
         title: a short title\n\
         characters: First Last\n\
         arc: the part of that person's arc this beat establishes\n\
         brief: the situation to dramatize, in one paragraph\n"
    ))
}

pub fn beat_prompt(premise: &str, beat: &crate::plan::Beat, prior: &str) -> String {
    let people = if beat.characters.is_empty() {
        "whoever this beat needs".to_string()
    } else {
        beat.characters.join(", ")
    };
    let turns = match beat.kind {
        BeatKind::Dialogue => {
            "\nBegin every spoken turn with the speaker's name, a colon, and the speech. Keep it to the two people named here."
        }
        BeatKind::GroupConversation => {
            "\nBegin every spoken turn with the speaker's name, a colon, and the speech. At least three people speak, and they answer one another."
        }
        BeatKind::Monologue => "\nStay inside the one voice named here.",
        BeatKind::Scene => {
            "\nNarrate the place and the action. Do not write the beat as a transcript of talk."
        }
        BeatKind::CharacterBackground => {
            "\nTell it as that person's history. Do not stage a live scene."
        }
    };
    let prior = if prior.trim().is_empty() {
        "Nothing has been written yet.".to_string()
    } else {
        tail(prior, 1_600)
    };
    chat_user(&format!(
        "{instruction}{turns}\n\n\
         Premise:\n{premise}\n\n\
         This beat is \"{title}\".\n\
         People: {people}.\n\
         Character arc in this beat: {arc}.\n\n\
         Situation:\n{brief}\n\n\
         Already written, for continuity only. Do not repeat it:\n{prior}\n\n\
         Write this beat only. Stay inside this format for the whole piece.",
        instruction = instruction(beat.kind),
        title = beat.title,
        arc = if beat.arc.trim().is_empty() {
            "carry the arc forward"
        } else {
            beat.arc.as_str()
        },
        brief = beat.brief,
    ))
}

fn tail(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    text.chars().skip(count - max_chars).collect()
}

pub fn qa_prompt(story: &str, sections: usize) -> String {
    chat_user(&format!(
        "Read this finished short story and check that it is coherent.\n\
         Names, places, objects, and the order of events must agree from the first section to the last. \
         A person must not change name, appear in a place they never reached, or know a fact they were not present for.\n\n\
         If the story already agrees with itself, reply with exactly this word and nothing else:\n\
         COHERENT\n\n\
         If it does not, rewrite the sections that must change. Keep all {sections} sections, in the same order, with the same titles. \
         Do not add a new plot. Reply in this shape and nothing else:\n\n\
         REVISED\n\
         title: the section title\n\
         prose:\n\
         the revised section\n\n\
         Story:\n{story}"
    ))
}

/// Text safe to show while tokens are still arriving.
///
/// Special tokens are cut. The last sentence is left unfinished.
pub fn live_text(text: &str) -> String {
    cut_special_tokens(text).trim().to_string()
}

pub fn clean_completion(text: &str) -> String {
    finish_at_sentence(&cut_special_tokens(text))
}

fn cut_special_tokens(text: &str) -> String {
    let markers = [
        "<｜end▁of▁sentence｜>",
        "<|endoftext|>",
        "<|User|>",
        "<|System|>",
        "<｜start▁of▁sentence｜>",
    ];
    let mut end = text.len();
    for marker in markers {
        if let Some(index) = text.find(marker) {
            end = end.min(index);
        }
    }
    text[..end]
        .trim()
        .trim_start_matches("</think>")
        .trim()
        .to_string()
}

fn finish_at_sentence(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() || ends_sentence(trimmed) {
        return trimmed.to_string();
    }
    let end = trimmed
        .char_indices()
        .rev()
        .find_map(|(index, ch)| matches!(ch, '.' | '!' | '?').then_some(index));
    match end {
        Some(index) => trimmed[..=index].trim().to_string(),
        None => trimmed.to_string(),
    }
}

fn ends_sentence(text: &str) -> bool {
    text.trim_end()
        .trim_end_matches(['"', '\'', '\u{201d}', '\u{2019}', ')', ']'])
        .chars()
        .next_back()
        .is_some_and(|ch| matches!(ch, '.' | '!' | '?'))
}
