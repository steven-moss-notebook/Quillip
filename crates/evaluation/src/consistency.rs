use narrative_world::{StoryChallenge, VerificationReport, scripted_plans, verify_plan};

pub struct AdversarialCase {
    pub name: &'static str,
    pub hard_pass: bool,
}

/// Known-bad generations the verifier must reject, plus the two valid controls.
pub fn adversarial_cases(challenge: &StoryChallenge) -> Vec<AdversarialCase> {
    let plans = scripted_plans(challenge);
    let mut cases = vec![
        case("valid_reveal", verify_plan(challenge, &plans[0])),
        case("valid_alternative", verify_plan(challenge, &plans[1])),
        case("knowledge_leak", verify_plan(challenge, &plans[2])),
        case("keyword_stuffing", verify_plan(challenge, &plans[3])),
    ];
    let mut empty = plans[0].clone();
    empty.prose.clear();
    cases.push(case("empty_prose", verify_plan(challenge, &empty)));
    cases
}

fn case(name: &'static str, report: VerificationReport) -> AdversarialCase {
    AdversarialCase {
        name,
        hard_pass: report.hard_pass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use narrative_world::generate_split;

    #[test]
    fn adversarial_prose_fails_and_valid_controls_pass() {
        let challenge = generate_split(11, 1, 0).unwrap().train.remove(0);
        let cases = adversarial_cases(&challenge);
        let by_name = |name| {
            cases
                .iter()
                .find(|case| case.name == name)
                .unwrap()
                .hard_pass
        };
        assert!(by_name("valid_reveal"));
        assert!(by_name("valid_alternative"));
        assert!(!by_name("knowledge_leak"));
        assert!(!by_name("keyword_stuffing"));
        assert!(!by_name("empty_prose"));
    }
}
