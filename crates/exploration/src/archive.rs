use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use narrative_world::VerificationReport;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NarrativeSignature {
    pub features: BTreeSet<String>,
}

pub fn features_from_report(report: &VerificationReport) -> NarrativeSignature {
    let mut features = BTreeSet::new();
    if report.protagonist_belief_changed {
        features.insert("protagonist_changed".into());
    }
    if report.other_belief_changed {
        features.insert("other_changed".into());
    }
    for fact in &report.reveal_order {
        features.insert(format!("reveal:{fact}"));
    }
    for conflict in &report.resolved {
        features.insert(format!("resolve:{conflict}"));
    }
    for conflict in &report.acknowledged {
        features.insert(format!("acknowledge:{conflict}"));
    }
    for entity in &report.affected {
        features.insert(format!("affect:{entity}"));
    }
    NarrativeSignature { features }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NarrativeArchive {
    pub signatures: Vec<NarrativeSignature>,
    pub prose: Vec<BTreeSet<String>>,
}

impl NarrativeArchive {
    pub fn insert(&mut self, signature: NarrativeSignature, prose: BTreeSet<String>) {
        self.signatures.push(signature);
        self.prose.push(prose);
    }

    pub fn len(&self) -> usize {
        self.signatures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }
}
