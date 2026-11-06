/// Schulman k3 estimator of `KL(new || reference)` for one token.
///
/// `exp(log_ref - log_new) - (log_ref - log_new) - 1`. It is zero when the
/// two log probabilities match and positive when they differ.
pub fn k3_kl(log_ref: f32, log_new: f32) -> f32 {
    let delta = log_ref - log_new;
    delta.exp() - delta - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_is_zero_at_equality_and_positive_otherwise() {
        assert!(k3_kl(-0.2, -0.2).abs() < 1e-6);
        assert!(k3_kl(-0.2, -0.5) > 0.0);
    }
}
