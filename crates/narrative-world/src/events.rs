use serde::{Deserialize, Serialize};

/// A state change the model claims the scene performs.
///
/// The verifier applies these in order. It does not treat any effect as the
/// correct literary choice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Effect {
    Reveal {
        fact: String,
        to: String,
    },
    Destroy {
        artifact: String,
    },
    Recover {
        artifact: String,
        explanation: String,
    },
    Affect {
        entity: String,
    },
    Resolve {
        conflict: String,
    },
    Acknowledge {
        conflict: String,
    },
    ChangeBelief {
        character: String,
        belief: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transformation {
    TragicTransformation,
    Redemption,
    IronicReversal,
    FlatArc,
    MoralAmbiguity,
    FailedTransformation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ending {
    pub consequential: bool,
    pub summary: String,
    #[serde(default)]
    pub transformation: Option<Transformation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanEvent {
    pub id: String,
    pub actor: String,
    pub action: String,
    pub scene: u32,
    #[serde(default)]
    pub requires_knowledge: Vec<String>,
    pub effects: Vec<Effect>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NarrativePlan {
    pub events: Vec<PlanEvent>,
    pub ending: Ending,
    #[serde(default)]
    pub unresolved_conflicts: Vec<String>,
    pub prose: String,
}
