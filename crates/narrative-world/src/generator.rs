//! Challenges are combinations of a desire, an incompatible obligation, an
//! information asymmetry, a costly decision, and an external event.
//!
//! Each combination stays inside one scenario template, so the variation is
//! coherent rather than a shuffle of unrelated plot words.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::characters::{Character, Relationship};
use crate::state::{Constraint, Fact, Genre, NarrativeState, StoryChallenge};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GeneratorError {
    #[error("template banks must be non-empty")]
    EmptyBank,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChallengeSet {
    pub train: Vec<StoryChallenge>,
    pub held_out: Vec<StoryChallenge>,
}

struct Template {
    family: &'static str,
    genre: Genre,
    names: &'static [(&'static str, &'static str, &'static str, &'static str)],
    desires: &'static [&'static str],
    obligations: &'static [&'static str],
    other_desires: &'static [&'static str],
    secrets: &'static [(&'static str, &'static str)],
    stakes: &'static [&'static str],
    artifact: (&'static str, &'static str),
    affected: &'static str,
    conflict: (&'static str, &'static str),
    events: &'static [&'static str],
    bonds: &'static [&'static str],
}

const TEMPLATES: &[Template] = &[
    Template {
        family: "last_letter",
        genre: Genre::Drama,
        names: &[
            ("mara", "Mara", "lina", "Lina"),
            ("edda", "Edda", "noor", "Noor"),
            ("iris", "Iris", "clem", "Clem"),
            ("ruth", "Ruth", "anya", "Anya"),
        ],
        desires: &[
            "to keep the people she loves from being publicly ruined",
            "to protect a friendship that has lasted since childhood",
            "to leave the archive without becoming the person who ends a family",
        ],
        obligations: &[
            "the school ledger shows money that was meant for students",
            "her post requires her to report the diversion of school funds",
            "the board will audit the school unless the ledger is explained",
        ],
        other_desires: &[
            "to keep the workshop her parent left her",
            "to stay in the town where the school employs her",
            "to believe her parent was careful with other people's money",
        ],
        secrets: &[
            (
                "diverted_funds",
                "a deceased benefactor secretly diverted school funds",
            ),
            (
                "false_grant",
                "a celebrated gift to the school was never paid",
            ),
        ],
        stakes: &[
            "publishing the ledger would destroy the daughter's livelihood",
            "the daughter still draws her wages from the endowment",
        ],
        artifact: ("ledger", "the financial ledger"),
        affected: "the school",
        conflict: (
            "loyalty_versus_truth",
            "loyalty to a friend against the truth in the records",
        ),
        events: &[
            "an auditor arrives a day early",
            "a pupil's scholarship is about to be cancelled",
            "the daughter asks to borrow the archive key",
            "a newspaper requests the benefactor's papers",
        ],
        bonds: &["closest friend", "foster sister"],
    },
    Template {
        family: "unopened_door",
        genre: Genre::Mystery,
        names: &[
            ("caleb", "Caleb", "mina", "Mina"),
            ("jonah", "Jonah", "petra", "Petra"),
            ("seth", "Seth", "willa", "Willa"),
            ("oscar", "Oscar", "daria", "Daria"),
        ],
        desires: &[
            "to settle the estate without opening old harm",
            "to keep the house in the family",
            "to finish the caretaker's year and leave",
        ],
        obligations: &[
            "the will cannot be executed while a room stays sealed",
            "the insurer will condemn the house unless every room is inspected",
            "a sealed codicil names someone the family has refused to acknowledge",
        ],
        other_desires: &[
            "to inherit the house unchanged",
            "to stop strangers from reading the family's letters",
            "to sell quickly and leave the town",
        ],
        secrets: &[
            ("hidden_heir", "the sealed room holds proof of another heir"),
            (
                "forged_will",
                "the displayed will was altered after the death",
            ),
        ],
        stakes: &[
            "opening the room would cost the sibling the inheritance they expect",
            "the sibling has already promised the house to a creditor",
        ],
        artifact: ("codicil", "the sealed codicil"),
        affected: "the house",
        conflict: (
            "shelter_versus_disclosure",
            "keeping a home against disclosing who it belongs to",
        ),
        events: &[
            "a storm breaks the lock",
            "a creditor comes to inventory the furniture",
            "the sibling schedules the sale for morning",
            "a clerk demands the missing page",
        ],
        bonds: &["younger sibling", "cousin"],
    },
    Template {
        family: "borrowed_name",
        genre: Genre::Drama,
        names: &[
            ("leo", "Leo", "saskia", "Saskia"),
            ("nico", "Nico", "freya", "Freya"),
            ("paolo", "Paolo", "ines", "Ines"),
            ("hugo", "Hugo", "mira", "Mira"),
        ],
        desires: &[
            "to be believed on stage under his own name",
            "to stop performing someone else's reputation",
            "to keep the company from closing on opening night",
        ],
        obligations: &[
            "the lead's name is already printed on a contract she did not sign",
            "the understudy knows the signature on the engagement was forged",
            "the theater cannot pay the cast unless the forged booking stands",
        ],
        other_desires: &[
            "to debut without a scandal",
            "to trust the letter that promised her the role",
            "to leave with her reputation intact",
        ],
        secrets: &[
            (
                "forged_booking",
                "the engagement letter was forged in the star's name",
            ),
            (
                "sold_review",
                "a guaranteed review was bought with the payroll",
            ),
        ],
        stakes: &[
            "exposing the forgery would end the star's first contract",
            "the company will fold if the booking is withdrawn tonight",
        ],
        artifact: ("letter", "the engagement letter"),
        affected: "the company",
        conflict: (
            "credit_versus_truth",
            "taking credit against telling the truth about the booking",
        ),
        events: &[
            "the critic asks to see the contract",
            "the real star sends word that she is in the city",
            "the box office has already sold out",
            "a musician recognizes the handwriting",
        ],
        bonds: &["scene partner", "old classmate"],
    },
    Template {
        family: "quiet_witness",
        genre: Genre::Mystery,
        names: &[
            ("hana", "Hana", "tomas", "Tomas"),
            ("leah", "Leah", "boris", "Boris"),
            ("nina", "Nina", "erik", "Erik"),
            ("ada", "Ada", "marko", "Marko"),
        ],
        desires: &[
            "to leave the courthouse without being noticed",
            "to keep her post as clerk",
            "to avoid choosing who the verdict harms",
        ],
        obligations: &[
            "a misplaced exhibit would change the verdict if it were read",
            "the clerk is the only person who saw where the exhibit was moved",
            "procedure requires every admitted exhibit to reach the judge",
        ],
        other_desires: &[
            "to deliver the verdict he has already written",
            "to protect a defendant he believes is innocent",
            "to finish the trial before the holiday recess",
        ],
        secrets: &[
            (
                "moved_exhibit",
                "an exhibit was moved so the judge would not see it",
            ),
            (
                "swapped_statement",
                "two witness statements were swapped in the bundle",
            ),
        ],
        stakes: &[
            "correcting the record would ruin the judge's remaining authority",
            "the clerk's correction would free one person and condemn another",
        ],
        artifact: ("exhibit", "the misplaced exhibit"),
        affected: "the verdict",
        conflict: (
            "procedure_versus_mercy",
            "following procedure against the mercy already decided",
        ),
        events: &[
            "the jury is about to be sent out",
            "a reporter is waiting in the corridor",
            "the defendant asks the clerk for water and sees the file",
            "the judge requests the exhibit by number",
        ],
        bonds: &["mentor", "uncle"],
    },
    Template {
        family: "winter_debt",
        genre: Genre::Fable,
        names: &[
            ("soren", "Soren", "bryn", "Bryn"),
            ("idris", "Idris", "kell", "Kell"),
            ("yara", "Yara", "pim", "Pim"),
            ("oskar", "Oskar", "lott", "Lott"),
        ],
        desires: &[
            "to keep every patient through the thaw",
            "to be remembered as a healer who did not choose",
            "to repay the village that trained him",
        ],
        obligations: &[
            "the ration book shows there is not enough medicine for every name",
            "one dose remains and two patients need it tonight",
            "the healer promised both families they would be first",
        ],
        other_desires: &[
            "to live long enough to see the road open",
            "to believe the healer tells the truth about the stores",
            "to take the last dose home to a child",
        ],
        secrets: &[
            (
                "short_stores",
                "the ration book shows the medicine stores were already short",
            ),
            (
                "reserved_dose",
                "a dose was reserved for someone who has not fallen ill",
            ),
        ],
        stakes: &[
            "telling the patient would start a run on the remaining stores",
            "hiding the shortage leaves the village to discover empty shelves",
        ],
        artifact: ("ration_book", "the ration book"),
        affected: "the village",
        conflict: (
            "care_versus_honesty",
            "caring for one patient against honesty to the village",
        ),
        events: &[
            "the road closes under snow",
            "a second patient collapses at the door",
            "the council asks for the book at dawn",
            "a child is sent to fetch the healer",
        ],
        bonds: &["patient", "apprentice"],
    },
    Template {
        family: "false_beacon",
        genre: Genre::Tragedy,
        names: &[
            ("nell", "Nell", "harun", "Harun"),
            ("tess", "Tess", "ivar", "Ivar"),
            ("joan", "Joan", "rafa", "Rafa"),
            ("maud", "Maud", "elio", "Elio"),
        ],
        desires: &[
            "to keep ships off the rocks in the dark",
            "to finish her watch without a wreck",
            "to trust the chart she was given",
        ],
        obligations: &[
            "the working chart has been altered and the true shoal is unmarked",
            "lighting the beacon as ordered will steer a ship onto the shoal",
            "the harbor master signed the altered chart",
        ],
        other_desires: &[
            "to make the tide and be paid",
            "to believe the harbor lights are honest",
            "to bring the cargo in before the storm",
        ],
        secrets: &[
            (
                "altered_chart",
                "someone altered the harbor chart to hide a shoal",
            ),
            (
                "paid_wreck",
                "a wreck on that shoal would pay a salvage claim",
            ),
        ],
        stakes: &[
            "warning the captain would expose the harbor master who employs her brother",
            "staying silent leaves the ship to follow the false mark",
        ],
        artifact: ("chart", "the harbor chart"),
        affected: "the harbor",
        conflict: (
            "duty_versus_kin",
            "the watch keeper's duty against the kin who depend on the harbor",
        ),
        events: &[
            "a ship answers the first light",
            "fog closes the channel",
            "the harbor master sends an order to keep the usual signal",
            "a fishing boat reports broken water where the chart says none",
        ],
        bonds: &["brother's employer", "old captain"],
    },
];

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
        debug_assert!(len > 0);
        (self.next() as usize) % len
    }
}

fn pick<'a>(rng: &mut Rng, items: &[&'a str]) -> &'a str {
    items[rng.index(items.len())]
}

pub fn generate_split(
    seed: u64,
    train: usize,
    held_out: usize,
) -> Result<ChallengeSet, GeneratorError> {
    let mut rng = Rng(seed);
    let mut train_set = Vec::with_capacity(train);
    let mut held_set = Vec::with_capacity(held_out);
    for index in 0..train {
        train_set.push(generate_one(&mut rng, index, false)?);
    }
    for index in 0..held_out {
        held_set.push(generate_one(&mut rng, train + index, true)?);
    }
    Ok(ChallengeSet {
        train: train_set,
        held_out: held_set,
    })
}

fn generate_one(
    rng: &mut Rng,
    index: usize,
    held_out: bool,
) -> Result<StoryChallenge, GeneratorError> {
    if TEMPLATES.iter().any(|template| {
        template.names.is_empty()
            || template.desires.is_empty()
            || template.obligations.is_empty()
            || template.secrets.is_empty()
            || template.events.is_empty()
    }) {
        return Err(GeneratorError::EmptyBank);
    }
    let template = &TEMPLATES[rng.index(TEMPLATES.len())];
    let (pid, pname, oid, oname) = template.names[rng.index(template.names.len())];
    let desire = pick(rng, template.desires);
    let obligation = pick(rng, template.obligations);
    let other_desire = pick(rng, template.other_desires);
    let (secret_id, secret) = template.secrets[rng.index(template.secrets.len())];
    let stake = pick(rng, template.stakes);
    let event = pick(rng, template.events);
    let bond = pick(rng, template.bonds);
    let extra = rng.index(2) == 0;
    let target_words = 300 + rng.index(201);
    let prefix = if held_out { "heldout" } else { "train" };

    let mut facts = vec![
        Fact {
            id: secret_id.to_string(),
            proposition: secret.to_string(),
            exclusive_with: None,
        },
        Fact {
            id: format!("{}_held", template.artifact.0),
            proposition: format!("{pname} possesses {}.", template.artifact.1),
            exclusive_with: None,
        },
        Fact {
            id: "stakes".into(),
            proposition: stake.to_string(),
            exclusive_with: None,
        },
        Fact {
            id: "bond".into(),
            proposition: format!("{oname} is {pname}'s {bond}."),
            exclusive_with: None,
        },
    ];
    if extra {
        facts.push(Fact {
            id: "external".into(),
            proposition: format!("Tonight, {event}."),
            exclusive_with: None,
        });
    }

    let protagonist = Character {
        id: pid.into(),
        name: pname.into(),
        desires: vec![desire.into()],
        beliefs: vec!["the record can still be handled quietly".into()],
        knowledge: vec![
            secret_id.into(),
            format!("{}_held", template.artifact.0),
            "stakes".into(),
            "bond".into(),
        ],
        relationships: vec![Relationship {
            other: oid.into(),
            kind: bond.into(),
        }],
    };
    let other_knowledge = vec!["bond".into(), "stakes".into()];
    let other = Character {
        id: oid.into(),
        name: oname.into(),
        desires: vec![other_desire.into()],
        beliefs: vec!["nothing in the records concerns them".into()],
        knowledge: if extra {
            let mut knowledge = other_knowledge;
            knowledge.push("external".into());
            knowledge
        } else {
            other_knowledge
        },
        relationships: vec![Relationship {
            other: pid.into(),
            kind: bond.into(),
        }],
    };
    let mut protagonist = protagonist;
    if extra {
        protagonist.knowledge.push("external".into());
    }

    let premise = format!(
        "{pname} wants {desire}. {obligation}. {pname} knows that {secret}, and {oname} does not. {stake}. {pname} holds {artifact}. Then {event}.",
        artifact = template.artifact.1,
    );

    Ok(StoryChallenge {
        id: format!("{prefix}-{}-{index}", template.family),
        family: template.family.into(),
        premise,
        initial_state: NarrativeState {
            characters: vec![protagonist, other],
            established_facts: facts,
            unresolved_conflicts: vec![template.conflict.0.into()],
            artifacts: vec![template.artifact.0.into()],
        },
        constraints: vec![
            Constraint::MustNotKnowInitially {
                character: oid.into(),
                fact: secret_id.into(),
            },
            Constraint::ArtifactPersistence {
                artifact: template.artifact.0.into(),
            },
            Constraint::MustAffect {
                entity: template.affected.into(),
            },
            Constraint::MustReachConsequentialEnding,
            Constraint::ConflictResolvedOrAcknowledged {
                conflict: template.conflict.0.into(),
            },
        ],
        target_words,
        genre: template.genre,
        held_out,
        knower: pid.into(),
        ignorant: oid.into(),
        secret_fact: secret_id.into(),
        artifact: template.artifact.0.into(),
        affected_entity: template.affected.into(),
        conflict_id: template.conflict.0.into(),
        protagonist: pid.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_is_deterministic_varied_and_well_formed() {
        let first = generate_split(7, 200, 20).unwrap();
        let second = generate_split(7, 200, 20).unwrap();
        assert_eq!(first, second);
        assert!(first.train.iter().all(|challenge| !challenge.held_out));
        assert!(first.held_out.iter().all(|challenge| challenge.held_out));
        assert!(
            first
                .train
                .iter()
                .all(|challenge| !challenge.id.starts_with("heldout"))
        );

        let mut families = std::collections::BTreeSet::new();
        let mut premises = std::collections::BTreeSet::new();
        for challenge in first.train.iter().chain(first.held_out.iter()) {
            families.insert(challenge.family.clone());
            premises.insert(challenge.premise.clone());
            assert_eq!(challenge.initial_state.characters.len(), 2);
            let facts = challenge.initial_state.established_facts.len();
            assert!((3..=5).contains(&facts), "facts={facts}");
            let knower = challenge.character(&challenge.knower).unwrap();
            let ignorant = challenge.character(&challenge.ignorant).unwrap();
            assert!(
                knower
                    .knowledge
                    .iter()
                    .any(|id| id == &challenge.secret_fact)
            );
            assert!(
                !ignorant
                    .knowledge
                    .iter()
                    .any(|id| id == &challenge.secret_fact)
            );
            assert!(
                challenge
                    .constraints
                    .iter()
                    .any(|constraint| matches!(constraint, Constraint::ArtifactPersistence { .. }))
            );
            assert!(challenge.target_words >= 300 && challenge.target_words <= 500);
        }
        assert_eq!(families.len(), TEMPLATES.len());
        assert!(premises.len() > 180, "premises={}", premises.len());
    }
}
