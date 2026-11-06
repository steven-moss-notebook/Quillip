/// Log probabilities aligned to completion tokens.
///
/// `aligned` is one entry per sequence token. Index 0 has no context and must
/// stay masked. `prompt_len` is the number of conditioning tokens, including
/// that first position.
pub fn completion_logprobs(aligned: &[f32], prompt_len: usize) -> &[f32] {
    let start = prompt_len.min(aligned.len());
    &aligned[start..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_slice_drops_the_prompt() {
        let aligned = [0.0, -0.2, -0.4, -0.6];
        assert_eq!(completion_logprobs(&aligned, 2), &[-0.4, -0.6]);
    }
}
