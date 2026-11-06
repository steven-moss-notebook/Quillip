/// Group-relative advantage for one prompt's candidates.
///
/// `A_i = (R_i - mean) / max(std, min_std)`, with the population standard
/// deviation. The floor keeps a near-tie from becoming an O(1) step.
/// A group whose rewards are identical has advantage zero, because the
/// numerator is zero too. That is the case contrastive shaping is allowed to fill in.
#[derive(Debug, Clone, Copy)]
pub struct GrpoConfig {
    /// Smallest standard deviation used in the denominator.
    ///
    /// Rubric scores live in `[0, 1]`. A floor of `1e-6` turned a gap of
    /// `1e-4` into the same advantage as a gap of `0.5`.
    pub min_std: f32,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self { min_std: 0.01 }
    }
}

/// Reward minus the group mean, before dividing by the standard deviation.
pub fn group_centered(rewards: &[f32]) -> Vec<f32> {
    let count = rewards.len();
    if count == 0 {
        return Vec::new();
    }
    let mean = rewards.iter().sum::<f32>() / count as f32;
    rewards.iter().map(|reward| reward - mean).collect()
}

pub fn group_advantages(rewards: &[f32], config: &GrpoConfig) -> Vec<f32> {
    let count = rewards.len();
    if count == 0 {
        return Vec::new();
    }
    let centered = group_centered(rewards);
    let variance = centered.iter().map(|delta| delta * delta).sum::<f32>() / count as f32;
    let scale = variance.sqrt().max(config.min_std);
    centered.into_iter().map(|delta| delta / scale).collect()
}

pub fn zero_variance(rewards: &[f32]) -> bool {
    if rewards.is_empty() {
        return true;
    }
    let first = rewards[0];
    rewards.iter().all(|reward| (reward - first).abs() <= 1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advantages_are_centered_and_zero_when_rewards_match() {
        let config = GrpoConfig::default();
        let advantages = group_advantages(&[1.0, 0.0, 1.0, 0.0], &config);
        let mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
        assert!(mean.abs() < 1e-5);
        assert!(advantages[0] > 0.0 && advantages[1] < 0.0);
        let flat = group_advantages(&[0.4, 0.4, 0.4, 0.4], &config);
        assert!(flat.iter().all(|advantage| advantage.abs() < 1e-5));
        assert!(zero_variance(&[0.4, 0.4, 0.4, 0.4]));
        let near_tie = group_advantages(&[0.5, 0.5, 0.5, 0.5001], &config);
        assert!(
            near_tie.iter().all(|advantage| advantage.abs() < 0.02),
            "a 1e-4 gap must not become a full update, got {near_tie:?}"
        );
        let centered = group_centered(&[0.5, 0.5, 0.5, 0.5001]);
        assert!(centered.iter().all(|gap| gap.abs() < 1e-3));
    }
}
