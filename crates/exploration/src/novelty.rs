use std::collections::BTreeSet;

use crate::archive::{NarrativeArchive, NarrativeSignature};

pub fn token_set(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|word| word.len() >= 4)
        .map(|word| word.to_lowercase())
        .collect()
}

pub fn token_set_words<'a>(words: impl IntoIterator<Item = &'a str>) -> BTreeSet<String> {
    words
        .into_iter()
        .flat_map(|word| word.split(|ch: char| !ch.is_alphanumeric()))
        .filter(|token| token.len() >= 4)
        .map(|token| token.to_lowercase())
        .collect()
}

/// Type-token ratio of the text, clamped to `[0, 1]`.
///
/// An empty text scores 0. A text where every word is distinct scores 1.
pub fn lexical_diversity(text: &str) -> f32 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }
    let unique: BTreeSet<&str> = words.iter().copied().collect();
    (unique.len() as f32 / words.len() as f32).clamp(0.0, 1.0)
}

pub fn jaccard_distance(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f32 {
    if left.is_empty() && right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(right).count() as f32;
    let union = left.union(right).count() as f32;
    1.0 - intersection / union.max(1.0)
}

/// Distance to the nearest archived story. An empty archive scores 1.
///
/// Structural distance weighs more than the crude word-set distance. The word
/// set is a stand-in for a later embedding, not a literary judgment.
pub fn novelty_score(
    archive: &NarrativeArchive,
    signature: &NarrativeSignature,
    prose: &BTreeSet<String>,
) -> f32 {
    if archive.is_empty() {
        return 1.0;
    }
    archive
        .signatures
        .iter()
        .zip(&archive.prose)
        .map(|(stored, stored_prose)| {
            0.7 * jaccard_distance(&signature.features, &stored.features)
                + 0.3 * jaccard_distance(prose, stored_prose)
        })
        .fold(f32::MAX, f32::min)
}

/// Measures the thematic and conceptual progression across the passage.
///
/// It splits the prose into three chronological slices: opening, middle, and ending.
/// It rewards forward conceptual movement and penalizes circular loops where the
/// closing merely returns to and rephrases the opening vocabulary.
pub fn intra_story_trajectory(prose: &str) -> f32 {
    let words: Vec<&str> = prose.split_whitespace().collect();
    if words.len() < 24 {
        return 0.5;
    }
    let slice = words.len() / 3;
    let head = token_set_words(words[..slice].iter().copied());
    let mid = token_set_words(words[slice..2 * slice].iter().copied());
    let tail = token_set_words(words[words.len() - slice..].iter().copied());

    let d_head_tail = jaccard_distance(&head, &tail);
    let d_head_mid = jaccard_distance(&head, &mid);
    let d_mid_tail = jaccard_distance(&mid, &tail);

    // A closed loop occurs when tail circles back towards head
    // (head-tail distance is smaller than mid-tail and head-mid transitions).
    let loop_penalty = if d_head_tail < d_mid_tail && d_head_tail < d_head_mid {
        0.5 * (d_mid_tail - d_head_tail)
    } else {
        0.0
    };

    (d_head_tail * 0.7 + d_mid_tail * 0.3 - loop_penalty).clamp(0.0, 1.0)
}

/// Novelty score gated by intra-story progression.
///
/// A story that repeats itself internally cannot claim full novelty reward,
/// even if its vocabulary is distinct from previous archived stories.
pub fn progressive_novelty_score(
    archive: &NarrativeArchive,
    signature: &NarrativeSignature,
    prose_tokens: &BTreeSet<String>,
    prose_text: &str,
) -> f32 {
    let archive_novelty = novelty_score(archive, signature, prose_tokens);
    let progression = intra_story_trajectory(prose_text);
    archive_novelty * (0.35 + 0.65 * progression)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::NarrativeSignature;

    #[test]
    fn a_duplicate_is_less_novel_than_the_first_acceptance() {
        let mut archive = NarrativeArchive::default();
        let signature = NarrativeSignature {
            features: BTreeSet::from(["reveal:secret".into(), "affect:school".into()]),
        };
        let prose = token_set("Mara told Lina and the school changed");
        assert_eq!(novelty_score(&archive, &signature, &prose), 1.0);
        archive.insert(signature.clone(), prose.clone());
        let again = novelty_score(&archive, &signature, &prose);
        assert!(again < 0.05, "duplicate novelty {again}");

        let other = NarrativeSignature {
            features: BTreeSet::from(["acknowledge:loyalty".into()]),
        };
        let other_prose = token_set("a different ending with no confession");
        let fresh = novelty_score(&archive, &other, &other_prose);
        assert!(fresh > again);
    }

    #[test]
    fn circular_prose_scores_lower_progression_than_evolving_prose() {
        // Circular prose: opens and closes with identical emotional/thematic tokens
        let circular = "The heavy gloom settled over the quiet empty hall with bitter sorrow and silent regret. \
                        He considered the distant mountains and wondered how the hours passed. \
                        The bitter sorrow and heavy gloom remained in the quiet empty hall with silent regret.";
        
        // Evolving prose: opening predicament transforms into decisive revelation and forward consequence
        let evolving = "The heavy gloom settled over the quiet empty hall with bitter sorrow and silent regret. \
                        He searched behind the decaying altar until his fingers caught the iron latch. \
                        The hidden vault swung outward toward the morning sunlight, revealing the signed treaty.";

        let score_circular = intra_story_trajectory(circular);
        let score_evolving = intra_story_trajectory(evolving);

        assert!(
            score_evolving > score_circular + 0.25,
            "evolving {score_evolving} should outscore circular {score_circular}"
        );

        let archive = NarrativeArchive::default();
        let signature = NarrativeSignature {
            features: BTreeSet::from(["reveal:secret".into()]),
        };
        let circular_tokens = token_set(circular);
        let evolving_tokens = token_set(evolving);

        let novelty_circ = progressive_novelty_score(&archive, &signature, &circular_tokens, circular);
        let novelty_evol = progressive_novelty_score(&archive, &signature, &evolving_tokens, evolving);
        assert!(novelty_evol > novelty_circ);
    }

    #[test]
    fn lexical_diversity_measures_word_uniqueness() {
        assert_eq!(lexical_diversity(""), 0.0);
        assert_eq!(lexical_diversity("hello world"), 1.0);
        assert_eq!(lexical_diversity("the the the the"), 0.25);
    }
}
