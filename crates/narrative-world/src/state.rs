use serde::{Deserialize, Serialize};

use crate::characters::Character;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Genre {
    Drama,
    Mystery,
    Fable,
    Tragedy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingMode {
    /// Ask for the scene directly.
    Off,
    /// A short plan, then the scene.
    Low,
    /// A structured plan, then the scene. This is the training format.
    High,
}

impl ThinkingMode {
    pub fn control_tag(self) -> &'static str {
        match self {
            Self::Off => "thinking:off",
            Self::Low => "thinking:low",
            Self::High => "thinking:high",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub proposition: String,
    /// Fact ids that cannot be established together with this one.
    #[serde(default)]
    pub exclusive_with: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NarrativeState {
    pub characters: Vec<Character>,
    pub established_facts: Vec<Fact>,
    pub unresolved_conflicts: Vec<String>,
    pub artifacts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Constraint {
    MustNotKnowInitially { character: String, fact: String },
    ArtifactPersistence { artifact: String },
    MustAffect { entity: String },
    MustReachConsequentialEnding,
    ConflictResolvedOrAcknowledged { conflict: String },
    MinScenes { scenes: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoryChallenge {
    pub id: String,
    pub family: String,
    pub premise: String,
    pub initial_state: NarrativeState,
    pub constraints: Vec<Constraint>,
    pub target_words: usize,
    pub genre: Genre,
    pub held_out: bool,
    pub knower: String,
    pub ignorant: String,
    pub secret_fact: String,
    pub artifact: String,
    pub affected_entity: String,
    pub conflict_id: String,
    pub protagonist: String,
}

impl StoryChallenge {
    pub fn character(&self, id: &str) -> Option<&Character> {
        self.initial_state
            .characters
            .iter()
            .find(|character| character.id == id)
    }

    pub fn fact(&self, id: &str) -> Option<&Fact> {
        self.initial_state
            .established_facts
            .iter()
            .find(|fact| fact.id == id)
    }
}
