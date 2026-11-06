//! Deterministic checks on a narrative plan, plus a lexical tier-2 check that
//! the prose actually mentions the transitions the plan claims.
//!
//! Tier 2 can be wrong. When a claim has no content words to look for, the
//! report marks itself uncertain and does not treat that claim as a failure.
//! Literary quality is not scored here.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::events::{Effect, NarrativePlan, Transformation};
use crate::state::{Constraint, StoryChallenge, ThinkingMode};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Violation {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub parsed: bool,
    pub consistency: f32,
    pub causality: f32,
    pub completion: f32,
    pub structure: f32,
    pub tier2_confidence: f32,
    pub tier2_uncertain: bool,
    /// Tier-1 constraints passed. Novelty may use this. It ignores prose quality.
    pub hard_pass: bool,
    /// Tier 1 passed and tier 2 is either confident or honestly uncertain.
    /// Contrastive shaping treats this as the binary correctness bit.
    pub binary_pass: bool,
    pub violations: Vec<Violation>,
    pub contradiction_count: usize,
    pub protagonist_belief_changed: bool,
    pub other_belief_changed: bool,
    pub reveal_order: Vec<String>,
    pub resolved: Vec<String>,
    pub acknowledged: Vec<String>,
    pub affected: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guidance {
    pub text: String,
}

impl VerificationReport {
    pub fn guidance(&self) -> Guidance {
        let status = if self.binary_pass { "pass" } else { "fail" };
        let lines = if self.violations.is_empty() {
            "- none".to_string()
        } else {
            self.violations
                .iter()
                .map(|violation| format!("- {}: {}", violation.code, violation.detail))
                .collect::<Vec<_>>()
                .join("\n")
        };
        Guidance {
            text: format!(
                "Verifier trace ({status}). Binary correctness is {status}.\n\
                 Do not treat this trace as a preferred story. It only names the checked state.\n\
                 {lines}"
            ),
        }
    }
}

pub fn render_prompt(challenge: &StoryChallenge, mode: ThinkingMode) -> String {
    format!(
        "{tag}\n\
         Write one original scene for this challenge. No single ending is correct.\n\
         Premise: {premise}\n\
         Protagonist: {protagonist}. The other character does not initially know `{secret}`.\n\
         Required checks, not required solutions:\n\
         - Respect who knows what at the start.\n\
         - `{artifact}` cannot be destroyed and later recovered without an explanation.\n\
         - `{entity}` must be affected by the decision.\n\
         - Conflict `{conflict}` must be resolved or explicitly left unresolved.\n\
         - The ending must be consequential.\n\
         - The prose must state the same events as the plan.\n\
         Target length: about {words} words of prose.\n\
         Return one JSON object with keys events, ending, unresolved_conflicts, prose.\n\
         events items have id, actor, action, scene, requires_knowledge, effects.\n\
         effect.kind is one of reveal, destroy, recover, affect, resolve, acknowledge, change_belief.\n\
         ending has consequential, summary, and optional transformation.",
        tag = mode.control_tag(),
        premise = challenge.premise,
        protagonist = challenge.protagonist,
        secret = challenge.secret_fact,
        artifact = challenge.artifact,
        entity = challenge.affected_entity,
        conflict = challenge.conflict_id,
        words = challenge.target_words,
    )
}

pub fn parse_model_output(text: &str) -> Result<NarrativePlan, String> {
    let start = text.find('{').ok_or_else(|| "no JSON object".to_string())?;
    let end = text
        .rfind('}')
        .ok_or_else(|| "no JSON object".to_string())?;
    if end < start {
        return Err("JSON object is truncated".into());
    }
    serde_json::from_str(&text[start..=end]).map_err(|err| err.to_string())
}

pub fn verify_output(challenge: &StoryChallenge, text: &str) -> VerificationReport {
    match parse_model_output(text) {
        Ok(plan) => verify_plan(challenge, &plan),
        Err(detail) => failed_parse(detail),
    }
}

pub fn verify_plan(challenge: &StoryChallenge, plan: &NarrativePlan) -> VerificationReport {
    let mut violations = Vec::new();
    let mut knowledge = BTreeMap::<String, BTreeSet<String>>::new();
    let mut belief_changed = BTreeSet::new();
    for character in &challenge.initial_state.characters {
        knowledge.insert(
            character.id.clone(),
            character.knowledge.iter().cloned().collect(),
        );
    }
    for constraint in &challenge.constraints {
        if let Constraint::MustNotKnowInitially { character, fact } = constraint {
            if knowledge
                .get(character)
                .is_some_and(|known| known.contains(fact))
            {
                violations.push(violation(
                    "initial_knowledge",
                    format!("{character} already knows {fact}"),
                ));
            }
        }
    }

    let mut destroyed = BTreeSet::new();
    let mut affected = BTreeSet::new();
    let mut resolved = BTreeSet::new();
    let mut acknowledged = BTreeSet::new();
    let mut reveal_order = Vec::new();
    let mut previous_scene = 0_u32;
    let mut saw_scene = false;
    let mut scenes = BTreeSet::new();

    for event in &plan.events {
        if saw_scene && event.scene < previous_scene {
            violations.push(violation(
                "chronology",
                format!("scene {} follows scene {previous_scene}", event.scene),
            ));
        }
        previous_scene = event.scene;
        saw_scene = true;
        scenes.insert(event.scene);

        if challenge.character(&event.actor).is_none() {
            violations.push(violation(
                "unknown_actor",
                format!("no character {}", event.actor),
            ));
            continue;
        }
        let known = knowledge.get(&event.actor).cloned().unwrap_or_default();
        for fact in &event.requires_knowledge {
            if !known.contains(fact) {
                violations.push(violation(
                    "missing_knowledge",
                    format!("{} acts on {fact} without having learned it", event.actor),
                ));
            }
        }
        for effect in &event.effects {
            apply_effect(
                challenge,
                &event.actor,
                effect,
                &mut knowledge,
                &mut belief_changed,
                &mut destroyed,
                &mut affected,
                &mut resolved,
                &mut acknowledged,
                &mut reveal_order,
                &mut violations,
            );
        }
    }

    let protagonist_belief_changed = belief_changed.contains(&challenge.protagonist);
    let other_belief_changed = belief_changed
        .iter()
        .any(|character| character != &challenge.protagonist);
    if let Some(pattern) = plan.ending.transformation {
        if !transformation_matches(pattern, protagonist_belief_changed, other_belief_changed) {
            violations.push(violation(
                "transformation_mismatch",
                "the declared transformation does not match the belief changes".into(),
            ));
        }
    }

    for constraint in &challenge.constraints {
        match constraint {
            Constraint::MustAffect { entity } => {
                if !affected.contains(entity) {
                    violations.push(violation(
                        "missing_consequence",
                        format!("{entity} is never affected"),
                    ));
                }
            }
            Constraint::MustReachConsequentialEnding => {
                if !plan.ending.consequential {
                    violations.push(violation(
                        "inconsequential_ending",
                        "the ending does not change anything".into(),
                    ));
                }
            }
            Constraint::ConflictResolvedOrAcknowledged { conflict } => {
                if !resolved.contains(conflict) && !acknowledged.contains(conflict) {
                    violations.push(violation(
                        "open_conflict",
                        format!("{conflict} is neither resolved nor acknowledged"),
                    ));
                }
            }
            Constraint::MinScenes { scenes: required } => {
                if scenes.len() < *required as usize {
                    violations.push(violation(
                        "scene_coverage",
                        format!("plan uses {} scenes, needs {required}", scenes.len()),
                    ));
                }
            }
            Constraint::MustNotKnowInitially { .. } | Constraint::ArtifactPersistence { .. } => {}
        }
    }

    let prose_empty = plan.prose.split_whitespace().count() == 0;
    if prose_empty {
        violations.push(violation("empty_prose", "the scene text is empty".into()));
    }
    if keyword_stuffed(&plan.prose) {
        violations.push(violation(
            "keyword_stuffing",
            "a token dominates the prose".into(),
        ));
    }

    let (tier2_confidence, tier2_uncertain, tier2_misses) =
        prose_support(challenge, plan, &knowledge_names(challenge));
    if !tier2_uncertain && tier2_confidence < 0.6 && !prose_empty {
        violations.push(violation(
            "prose_unsupported",
            format!(
                "only {:.0}% of claimed transitions appear in the prose",
                tier2_confidence * 100.0
            ),
        ));
    }
    let _ = tier2_misses;

    let hard_codes = [
        "initial_knowledge",
        "chronology",
        "unknown_actor",
        "missing_knowledge",
        "unknown_fact",
        "unknown_character",
        "unknown_artifact",
        "recover_intact",
        "recover_unexplained",
        "transformation_mismatch",
        "missing_consequence",
        "inconsequential_ending",
        "open_conflict",
        "scene_coverage",
        "empty_prose",
        "keyword_stuffing",
        "exclusive_facts",
    ];
    let hard_hits = violations
        .iter()
        .filter(|violation| hard_codes.contains(&violation.code.as_str()))
        .count();
    let causal_hits = violations
        .iter()
        .filter(|violation| {
            matches!(
                violation.code.as_str(),
                "missing_knowledge" | "recover_intact" | "recover_unexplained"
            )
        })
        .count();
    let completion_hits = violations
        .iter()
        .filter(|violation| {
            matches!(
                violation.code.as_str(),
                "missing_consequence"
                    | "inconsequential_ending"
                    | "open_conflict"
                    | "scene_coverage"
            )
        })
        .count();
    let consistency_hits = violations
        .iter()
        .filter(|violation| {
            matches!(
                violation.code.as_str(),
                "initial_knowledge"
                    | "chronology"
                    | "unknown_actor"
                    | "unknown_fact"
                    | "unknown_character"
                    | "unknown_artifact"
                    | "transformation_mismatch"
                    | "exclusive_facts"
            )
        })
        .count();

    let contradiction_count = violations
        .iter()
        .filter(|violation| {
            matches!(
                violation.code.as_str(),
                "missing_knowledge"
                    | "chronology"
                    | "recover_unexplained"
                    | "transformation_mismatch"
                    | "exclusive_facts"
                    | "initial_knowledge"
            )
        })
        .count();

    let structure = structure_score(plan, challenge.target_words, &violations);
    let hard_pass = hard_hits == 0;
    let binary_pass = hard_pass && (tier2_uncertain || tier2_confidence >= 0.6);

    VerificationReport {
        parsed: true,
        consistency: component(consistency_hits),
        causality: component(causal_hits),
        completion: component(completion_hits),
        structure,
        tier2_confidence,
        tier2_uncertain,
        hard_pass,
        binary_pass,
        violations,
        contradiction_count,
        protagonist_belief_changed,
        other_belief_changed,
        reveal_order,
        resolved: resolved.into_iter().collect(),
        acknowledged: acknowledged.into_iter().collect(),
        affected: affected.into_iter().collect(),
    }
}

fn component(hits: usize) -> f32 {
    (1.0 - hits as f32 * 0.5).max(0.0)
}

fn structure_score(plan: &NarrativePlan, target: usize, violations: &[Violation]) -> f32 {
    if violations
        .iter()
        .any(|violation| violation.code == "empty_prose")
    {
        return 0.0;
    }
    let shape = if plan.events.is_empty() { 0.0 } else { 1.0 };
    let words = plan.prose.split_whitespace().count() as f32;
    let tolerance = (target as f32 * 0.2).max(40.0);
    let distance = (words - target as f32).abs();
    let length = if distance <= tolerance {
        1.0
    } else {
        (1.0 - (distance - tolerance) / target.max(1) as f32).max(0.0)
    };
    let stuffing = if violations
        .iter()
        .any(|violation| violation.code == "keyword_stuffing")
    {
        0.0
    } else {
        1.0
    };
    (0.4 * shape + 0.4 * length + 0.2 * stuffing).clamp(0.0, 1.0)
}

fn transformation_matches(
    pattern: Transformation,
    protagonist_changed: bool,
    other_changed: bool,
) -> bool {
    match pattern {
        Transformation::FailedTransformation => !protagonist_changed,
        Transformation::FlatArc => !protagonist_changed && other_changed,
        Transformation::TragicTransformation
        | Transformation::Redemption
        | Transformation::IronicReversal
        | Transformation::MoralAmbiguity => protagonist_changed,
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_effect(
    challenge: &StoryChallenge,
    actor: &str,
    effect: &Effect,
    knowledge: &mut BTreeMap<String, BTreeSet<String>>,
    belief_changed: &mut BTreeSet<String>,
    destroyed: &mut BTreeSet<String>,
    affected: &mut BTreeSet<String>,
    resolved: &mut BTreeSet<String>,
    acknowledged: &mut BTreeSet<String>,
    reveal_order: &mut Vec<String>,
    violations: &mut Vec<Violation>,
) {
    match effect {
        Effect::Reveal { fact, to } => {
            if challenge.fact(fact).is_none() {
                violations.push(violation("unknown_fact", format!("no fact {fact}")));
                return;
            }
            if challenge.character(to).is_none() {
                violations.push(violation("unknown_character", format!("no character {to}")));
                return;
            }
            if let Some(known) = knowledge.get(actor) {
                if !known.contains(fact) {
                    violations.push(violation(
                        "missing_knowledge",
                        format!("{actor} reveals {fact} without knowing it"),
                    ));
                }
            }
            knowledge
                .entry(to.clone())
                .or_default()
                .insert(fact.clone());
            reveal_order.push(fact.clone());
            if let Some(fact_def) = challenge.fact(fact) {
                if let Some(other) = &fact_def.exclusive_with {
                    let recipient = knowledge.get(to).cloned().unwrap_or_default();
                    if recipient.contains(other) {
                        violations.push(violation(
                            "exclusive_facts",
                            format!("{fact} contradicts {other}"),
                        ));
                    }
                }
            }
        }
        Effect::Destroy { artifact } => {
            if !challenge
                .initial_state
                .artifacts
                .iter()
                .any(|id| id == artifact)
            {
                violations.push(violation(
                    "unknown_artifact",
                    format!("no artifact {artifact}"),
                ));
                return;
            }
            destroyed.insert(artifact.clone());
        }
        Effect::Recover {
            artifact,
            explanation,
        } => {
            if !challenge
                .initial_state
                .artifacts
                .iter()
                .any(|id| id == artifact)
            {
                violations.push(violation(
                    "unknown_artifact",
                    format!("no artifact {artifact}"),
                ));
                return;
            }
            if !destroyed.contains(artifact) {
                violations.push(violation(
                    "recover_intact",
                    format!("{artifact} is recovered without having been lost"),
                ));
            }
            if explanation.split_whitespace().count() < 3 {
                violations.push(violation(
                    "recover_unexplained",
                    format!("{artifact} reappears without an explanation"),
                ));
            }
            destroyed.remove(artifact);
        }
        Effect::Affect { entity } => {
            affected.insert(entity.clone());
        }
        Effect::Resolve { conflict } => {
            if !challenge
                .initial_state
                .unresolved_conflicts
                .iter()
                .any(|id| id == conflict)
            {
                violations.push(violation(
                    "open_conflict",
                    format!("no conflict {conflict} to resolve"),
                ));
            }
            resolved.insert(conflict.clone());
        }
        Effect::Acknowledge { conflict } => {
            acknowledged.insert(conflict.clone());
        }
        Effect::ChangeBelief {
            character,
            belief: _,
        } => {
            if challenge.character(character).is_none() {
                violations.push(violation(
                    "unknown_character",
                    format!("no character {character}"),
                ));
                return;
            }
            belief_changed.insert(character.clone());
        }
    }
}

fn prose_support(
    challenge: &StoryChallenge,
    plan: &NarrativePlan,
    _names: &BTreeMap<String, String>,
) -> (f32, bool, usize) {
    let prose = plan.prose.to_lowercase();
    let mut checked = 0_usize;
    let mut matched = 0_usize;
    for event in &plan.events {
        let Some(character) = challenge.character(&event.actor) else {
            continue;
        };
        let actor_present = prose.contains(&character.name.to_lowercase())
            || prose.contains(&character.id.to_lowercase());
        for effect in &event.effects {
            let evidence = effect_evidence(challenge, effect);
            let words = content_words(&evidence);
            if words.is_empty() {
                continue;
            }
            checked += 1;
            let mentioned = words.iter().any(|word| prose.contains(word));
            if actor_present && mentioned {
                matched += 1;
            }
        }
    }
    if checked == 0 {
        return (0.5, true, 0);
    }
    (matched as f32 / checked as f32, false, checked - matched)
}

fn effect_evidence(challenge: &StoryChallenge, effect: &Effect) -> String {
    match effect {
        Effect::Reveal { fact, .. } => challenge
            .fact(fact)
            .map(|fact| fact.proposition.clone())
            .unwrap_or_else(|| fact.clone()),
        Effect::Destroy { artifact } | Effect::Recover { artifact, .. } => artifact.clone(),
        Effect::Affect { entity } => entity.clone(),
        Effect::Resolve { conflict } | Effect::Acknowledge { conflict } => conflict.clone(),
        Effect::ChangeBelief { belief, .. } => belief.clone(),
    }
}

fn content_words(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|word| word.len() >= 4)
        .map(|word| word.to_lowercase())
        .collect()
}

fn keyword_stuffed(prose: &str) -> bool {
    let tokens = prose
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|word| word.len() >= 3)
        .map(|word| word.to_lowercase())
        .collect::<Vec<_>>();
    if tokens.len() < 20 {
        return false;
    }
    let mut counts = BTreeMap::<String, usize>::new();
    for token in &tokens {
        *counts.entry(token.clone()).or_default() += 1;
    }
    counts
        .values()
        .any(|count| *count >= 8 && *count as f32 / tokens.len() as f32 > 0.25)
}

fn knowledge_names(challenge: &StoryChallenge) -> BTreeMap<String, String> {
    challenge
        .initial_state
        .characters
        .iter()
        .map(|character| (character.id.clone(), character.name.clone()))
        .collect()
}

fn violation(code: &str, detail: String) -> Violation {
    Violation {
        code: code.to_string(),
        detail,
    }
}

fn failed_parse(detail: String) -> VerificationReport {
    VerificationReport {
        parsed: false,
        consistency: 0.0,
        causality: 0.0,
        completion: 0.0,
        structure: 0.0,
        tier2_confidence: 0.0,
        tier2_uncertain: false,
        hard_pass: false,
        binary_pass: false,
        violations: vec![violation("unparsed", detail)],
        contradiction_count: 0,
        protagonist_belief_changed: false,
        other_belief_changed: false,
        reveal_order: Vec::new(),
        resolved: Vec::new(),
        acknowledged: Vec::new(),
        affected: Vec::new(),
    }
}

/// Four hand-built answers for one challenge: two valid and structurally
/// different, one knowledge leak, and one stuffed scene.
pub fn scripted_plans(challenge: &StoryChallenge) -> Vec<NarrativePlan> {
    use crate::events::{Ending, PlanEvent};

    let knower = challenge.character(&challenge.knower).unwrap();
    let ignorant = challenge.character(&challenge.ignorant).unwrap();
    let secret = challenge.fact(&challenge.secret_fact).unwrap();
    let entity = challenge.affected_entity.clone();
    let conflict_words = challenge.conflict_id.replace('_', " ");
    let prose_reveal = format!(
        "{} already knew that {}. {} did not. {} revealed it. {} was affected. The conflict {conflict_words} was resolved. {}'s belief became: silence is no longer available.",
        knower.name, secret.proposition, ignorant.name, knower.name, entity, knower.name
    );
    let valid_reveal = NarrativePlan {
        events: vec![PlanEvent {
            id: "reveal".into(),
            actor: knower.id.clone(),
            action: "tells the truth".into(),
            scene: 1,
            requires_knowledge: vec![challenge.secret_fact.clone()],
            effects: vec![
                Effect::Reveal {
                    fact: challenge.secret_fact.clone(),
                    to: ignorant.id.clone(),
                },
                Effect::Affect {
                    entity: entity.clone(),
                },
                Effect::Resolve {
                    conflict: challenge.conflict_id.clone(),
                },
                Effect::ChangeBelief {
                    character: knower.id.clone(),
                    belief: "silence is no longer available".into(),
                },
            ],
        }],
        ending: Ending {
            consequential: true,
            summary: format!("{entity} is left altered"),
            transformation: Some(Transformation::MoralAmbiguity),
        },
        unresolved_conflicts: Vec::new(),
        prose: prose_reveal,
    };

    let prose_quiet = format!(
        "{} kept the record. {}'s belief became: the silence itself is an answer. The conflict {conflict_words} stayed open, and {} was affected.",
        knower.name, ignorant.name, entity
    );
    let valid_quiet = NarrativePlan {
        events: vec![PlanEvent {
            id: "withhold".into(),
            actor: knower.id.clone(),
            action: "acts without confessing".into(),
            scene: 1,
            requires_knowledge: vec![challenge.secret_fact.clone()],
            effects: vec![
                Effect::Affect {
                    entity: entity.clone(),
                },
                Effect::Acknowledge {
                    conflict: challenge.conflict_id.clone(),
                },
                Effect::ChangeBelief {
                    character: ignorant.id.clone(),
                    belief: "the silence itself is an answer".into(),
                },
            ],
        }],
        ending: Ending {
            consequential: true,
            summary: format!("{entity} changes without a confession"),
            transformation: Some(Transformation::FlatArc),
        },
        unresolved_conflicts: vec![challenge.conflict_id.clone()],
        prose: prose_quiet,
    };

    let leaked = NarrativePlan {
        events: vec![PlanEvent {
            id: "leak".into(),
            actor: ignorant.id.clone(),
            action: "uses the secret".into(),
            scene: 1,
            requires_knowledge: vec![challenge.secret_fact.clone()],
            effects: vec![
                Effect::Affect {
                    entity: entity.clone(),
                },
                Effect::Resolve {
                    conflict: challenge.conflict_id.clone(),
                },
            ],
        }],
        ending: Ending {
            consequential: true,
            summary: "an unearned action".into(),
            transformation: None,
        },
        unresolved_conflicts: Vec::new(),
        prose: format!(
            "{} acted on {}. {} was affected.",
            ignorant.name, secret.proposition, entity
        ),
    };

    let stuffed = NarrativePlan {
        events: valid_reveal.events.clone(),
        ending: valid_reveal.ending.clone(),
        unresolved_conflicts: Vec::new(),
        prose: format!("{} ", challenge.artifact).repeat(30),
    };
    vec![valid_reveal, valid_quiet, leaked, stuffed]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{Ending, PlanEvent};
    use crate::generator::generate_split;

    fn sample() -> StoryChallenge {
        generate_split(3, 1, 0).unwrap().train.remove(0)
    }

    #[test]
    fn valid_plans_pass_and_seeded_violations_fail() {
        let challenge = sample();
        let plans = scripted_plans(&challenge);
        let reveal = verify_plan(&challenge, &plans[0]);
        let quiet = verify_plan(&challenge, &plans[1]);
        assert!(reveal.binary_pass, "{:?}", reveal.violations);
        assert!(quiet.binary_pass, "{:?}", quiet.violations);
        assert_ne!(reveal.reveal_order, quiet.reveal_order);
        assert_ne!(
            reveal.protagonist_belief_changed,
            quiet.protagonist_belief_changed
        );

        let leaked = verify_plan(&challenge, &plans[2]);
        assert!(!leaked.hard_pass);
        assert!(
            leaked
                .violations
                .iter()
                .any(|violation| violation.code == "missing_knowledge")
        );

        let stuffed = verify_plan(&challenge, &plans[3]);
        assert!(!stuffed.hard_pass);
        assert!(
            stuffed
                .violations
                .iter()
                .any(|violation| violation.code == "keyword_stuffing")
        );
    }

    #[test]
    fn unexplained_recovery_is_rejected() {
        let challenge = sample();
        let knower = challenge.knower.clone();
        let plan = NarrativePlan {
            events: vec![
                PlanEvent {
                    id: "lose".into(),
                    actor: knower.clone(),
                    action: "destroys the record".into(),
                    scene: 1,
                    requires_knowledge: vec![],
                    effects: vec![Effect::Destroy {
                        artifact: challenge.artifact.clone(),
                    }],
                },
                PlanEvent {
                    id: "back".into(),
                    actor: knower,
                    action: "produces it again".into(),
                    scene: 2,
                    requires_knowledge: vec![],
                    effects: vec![
                        Effect::Recover {
                            artifact: challenge.artifact.clone(),
                            explanation: "somehow".into(),
                        },
                        Effect::Affect {
                            entity: challenge.affected_entity.clone(),
                        },
                        Effect::Resolve {
                            conflict: challenge.conflict_id.clone(),
                        },
                    ],
                },
            ],
            ending: Ending {
                consequential: true,
                summary: "it came back".into(),
                transformation: None,
            },
            unresolved_conflicts: Vec::new(),
            prose: "A record vanished and returned.".into(),
        };
        let report = verify_plan(&challenge, &plan);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.code == "recover_unexplained")
        );
        assert!(!report.hard_pass);
    }

    #[test]
    fn explained_recovery_is_allowed() {
        let challenge = sample();
        let knower = challenge.character(&challenge.knower).unwrap();
        let name = knower.name.clone();
        let knower_id = knower.id.clone();
        let artifact = challenge.artifact.clone();
        let plan = NarrativePlan {
            events: vec![
                PlanEvent {
                    id: "lose".into(),
                    actor: knower_id.clone(),
                    action: "hides the record".into(),
                    scene: 1,
                    requires_knowledge: vec![],
                    effects: vec![Effect::Destroy {
                        artifact: artifact.clone(),
                    }],
                },
                PlanEvent {
                    id: "copy".into(),
                    actor: knower_id,
                    action: "produces the copy".into(),
                    scene: 2,
                    requires_knowledge: vec![],
                    effects: vec![
                        Effect::Recover {
                            artifact: artifact.clone(),
                            explanation: "a clerk had already copied the pages".into(),
                        },
                        Effect::Affect {
                            entity: challenge.affected_entity.clone(),
                        },
                        Effect::Acknowledge {
                            conflict: challenge.conflict_id.clone(),
                        },
                    ],
                },
            ],
            ending: Ending {
                consequential: true,
                summary: "the copy remains".into(),
                transformation: None,
            },
            unresolved_conflicts: vec![challenge.conflict_id.clone()],
            prose: format!(
                "{name} hid the {artifact} and later produced it because a clerk had already copied the pages. {} was affected.",
                challenge.affected_entity
            ),
        };
        let report = verify_plan(&challenge, &plan);
        assert!(
            !report
                .violations
                .iter()
                .any(|violation| violation.code == "recover_unexplained"),
            "{:?}",
            report.violations
        );
    }

    #[test]
    fn json_output_parses_and_prompt_does_not_contain_a_solved_ending() {
        let challenge = sample();
        let prompt = render_prompt(&challenge, ThinkingMode::High);
        assert!(prompt.contains("thinking:high"));
        assert!(!prompt.contains("\"prose\":"));
        let plan = &scripted_plans(&challenge)[0];
        let text = serde_json::to_string(plan).unwrap();
        let wrapped = format!("Here is the scene:\n```json\n{text}\n```");
        let report = verify_output(&challenge, &wrapped);
        assert!(report.parsed);
        assert!(report.binary_pass, "{:?}", report.violations);
    }
}
