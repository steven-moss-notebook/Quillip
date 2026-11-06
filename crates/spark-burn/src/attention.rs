//! Grouped-query attention with a fused QKV projection, a sliding or global
//! causal mask, and a sigmoid gate on the attention output.
//!
//! The gate is a linear map from the hidden state to one logit per query head.
//! That is the attention-output gate whose parameter count, with a 131,072
//! vocabulary, matches the published 1.708B figure. It is not part of the
//! initial LoRA target.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use burn::nn::{Initializer, Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{DType, Device, FloatDType, Tensor, TensorData};

use crate::config::RopeSpec;
use crate::rope::apply_rope;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AttentionKind {
    Sliding,
    Global,
}

/// Causal mask, optionally restricted to a sliding window.
///
/// `window` counts tokens including the current one. `None` is full causal
/// attention. Returned row-major as `[query, key]`, with blocked positions set
/// to negative infinity.
pub fn attention_mask(seq: usize, window: Option<usize>) -> Vec<f32> {
    let mut mask = vec![0.0; seq * seq];
    for query in 0..seq {
        let first_key = match window {
            Some(window) => query.saturating_sub(window - 1),
            None => 0,
        };
        for key in 0..seq {
            if key > query || key < first_key {
                mask[query * seq + key] = f32::NEG_INFINITY;
            }
        }
    }
    mask
}

/// Causal (or sliding) mask for a right-padded batch.
///
/// `lengths[row]` real tokens sit at the start of that row. A real query
/// cannot see a pad key. That is the same rule as a causal mask when the pad
/// is strictly to the right, and it still holds if a pad column ever lands
/// inside the lookback. A pad query attends only to itself: masking every key
/// makes softmax NaN, and that NaN would be the next block's value. When every
/// length equals `seq`, each row is [`attention_mask`].
pub fn attention_mask_right(seq: usize, window: Option<usize>, lengths: &[usize]) -> Vec<f32> {
    let batch = lengths.len();
    let mut mask = vec![0.0; batch * seq * seq];
    for (row, &len) in lengths.iter().enumerate() {
        let base = row * seq * seq;
        for query in 0..seq {
            let first_key = match window {
                Some(window) => query.saturating_sub(window - 1),
                None => 0,
            };
            for key in 0..seq {
                let blocked = if query >= len {
                    key != query
                } else {
                    key >= len || key > query || key < first_key
                };
                if blocked {
                    mask[base + query * seq + key] = f32::NEG_INFINITY;
                }
            }
        }
    }
    mask
}

/// Square mask whose real positions ignore a left-pad prefix.
///
/// Index `i` in a row has position `i - pad` once `i` reaches the prefix.
/// Keys inside the prefix are blocked. With a zero prefix this is
/// [`attention_mask`].
pub fn attention_mask_padded(seq: usize, window: Option<usize>, pad_prefix: &[usize]) -> Vec<f32> {
    let batch = pad_prefix.len();
    let mut mask = vec![f32::NEG_INFINITY; batch * seq * seq];
    for (row, &pad) in pad_prefix.iter().enumerate() {
        let base = row * seq * seq;
        for query in 0..seq {
            if query < pad {
                // A pad query attends only to itself. Masking every key makes
                // softmax NaN, and that NaN rides into the next layer's values.
                mask[base + query * seq + query] = 0.0;
            } else {
                let query_pos = query - pad;
                let first_key = match window {
                    Some(window) => query_pos.saturating_sub(window - 1),
                    None => 0,
                };
                let start_key = pad + first_key;
                let end_key = query + 1;
                let row_offset = base + query * seq;
                mask[row_offset + start_key..row_offset + end_key].fill(0.0);
            }
        }
    }
    mask
}

fn host_mask(seq: usize, window: Option<usize>) -> Vec<f32> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, usize), Vec<f32>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (seq, window.unwrap_or(usize::MAX));
    let mut guard = cache.lock().expect("attention mask cache");
    if let Some(found) = guard.get(&key) {
        return found.clone();
    }
    let values = attention_mask(seq, window);
    guard.insert(key, values.clone());
    values
}

struct DeviceMask {
    seq: usize,
    window: Option<usize>,
    device: Device,
    tensor: Tensor<4>,
}

thread_local! {
    static DEVICE_MASKS: RefCell<Vec<DeviceMask>> = const { RefCell::new(Vec::new()) };
}

/// Sliding and global masks are uploaded once per sequence length and reused
/// by every layer of that forward.
fn mask_tensor(seq: usize, window: Option<usize>, device: &Device) -> Tensor<4> {
    DEVICE_MASKS.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some(found) = cache
            .iter()
            .find(|item| item.seq == seq && item.window == window && item.device == *device)
        {
            return found.tensor.clone();
        }
        let values = host_mask(seq, window);
        let tensor = Tensor::<4>::from_data(TensorData::new(values, [1, 1, seq, seq]), device)
            .cast(FloatDType::F32);
        if cache.len() == 8 {
            cache.remove(0);
        }
        cache.push(DeviceMask {
            seq,
            window,
            device: device.clone(),
            tensor: tensor.clone(),
        });
        tensor
    })
}

fn repeat_kv(states: Tensor<4>, groups: usize) -> Tensor<4> {
    if groups == 1 {
        return states;
    }
    let [batch, kv_heads, seq, dim] = states.dims();
    states
        .unsqueeze_dim::<5>(2)
        .repeat_dim(2, groups)
        .reshape([batch, kv_heads * groups, seq, dim])
}

/// Keys and values already rotated, for one row.
#[derive(Clone)]
pub(crate) struct LayerKv {
    pub key: Tensor<4>,
    pub value: Tensor<4>,
    /// Absolute position of key index 0.
    pub pos0: usize,
}

#[derive(burn::module::Module, Debug)]
pub struct SparkAttention {
    pub qkv: Linear,
    pub gate: Linear,
    pub output: Linear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Zero selects global attention.
    pub sliding_window: usize,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

impl SparkAttention {
    pub fn new(
        hidden: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        bias: bool,
        initializer: Initializer,
        rope: RopeSpec,
        device: &Device,
    ) -> Self {
        let qkv_out = n_heads * head_dim + 2 * n_kv_heads * head_dim;
        Self {
            qkv: LinearConfig::new(hidden, qkv_out)
                .with_bias(bias)
                .with_initializer(initializer.clone())
                .init(device),
            gate: LinearConfig::new(hidden, n_heads)
                .with_bias(bias)
                .with_initializer(initializer.clone())
                .init(device),
            output: LinearConfig::new(n_heads * head_dim, hidden)
                .with_bias(bias)
                .with_initializer(initializer)
                .init(device),
            n_heads,
            n_kv_heads,
            head_dim,
            sliding_window,
            rope_theta: rope.theta,
            partial_rotary_factor: rope.partial_rotary_factor,
        }
    }

    pub(crate) fn rope_spec(&self) -> RopeSpec {
        RopeSpec {
            theta: self.rope_theta,
            partial_rotary_factor: self.partial_rotary_factor,
        }
    }

    fn groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    pub(crate) fn window(&self) -> Option<usize> {
        (self.sliding_window > 0).then_some(self.sliding_window)
    }

    /// Q, K, and V in `[batch, heads, seq, dim]`, before RoPE. K and V use
    /// the key/value head count.
    fn project(&self, hidden: Tensor<3>) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [batch, seq, _] = hidden.dims();
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mixed = self.qkv.forward(hidden);
        let query = mixed.clone().narrow(2, 0, q_dim);
        let key = mixed.clone().narrow(2, q_dim, kv_dim);
        let value = mixed.narrow(2, q_dim + kv_dim, kv_dim);
        let query = query
            .reshape([batch, seq, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let key = key
            .reshape([batch, seq, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let value = value
            .reshape([batch, seq, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        (query, key, value)
    }

    fn mix(
        &self,
        query: Tensor<4>,
        key: Tensor<4>,
        value: Tensor<4>,
        mask: Tensor<4>,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        let [batch, seq, _] = hidden.dims();
        let scale = (self.head_dim as f32).sqrt();
        let scores = query.matmul(key.swap_dims(2, 3)) / scale;
        let dtype = scores.dtype();
        // Softmax over a 1,536-token row is not exact in bf16: the denominator's
        // ulp is 8, and `+inf` added to the mask's `-inf` is NaN.
        let scores = scores.cast(FloatDType::F32) + mask.cast(FloatDType::F32);
        let context = softmax(scores, 3).cast(dtype).matmul(value);
        // The published gate is sigmoid in fp32, then cast back to the activation dtype.
        let gate = sigmoid(self.gate.forward(hidden).cast(FloatDType::F32));
        let gate = if dtype == DType::F32 {
            gate
        } else {
            gate.cast(dtype)
        };
        let gate = gate.unsqueeze_dim::<4>(3);
        let context = context.swap_dims(1, 2) * gate;
        let merged = context.reshape([batch, seq, self.n_heads * self.head_dim]);
        self.output.forward(merged)
    }

    pub fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        let seq = hidden.dims()[1];
        let (query, key, value) = self.project(hidden.clone());
        let query = apply_rope(query, self.rope_spec());
        let key = apply_rope(key, self.rope_spec());
        let key = repeat_kv(key, self.groups());
        let value = repeat_kv(value, self.groups());
        let device = key.device();
        let mask = mask_tensor(seq, self.window(), &device);
        self.mix(query, key, value, mask, hidden)
    }

    /// Full-sequence forward that also returns rotated keys and values.
    ///
    /// `lengths` is the real token count of each right-padded row. The mask at
    /// this block is the causal (or sliding) mask and the padding mask: a real
    /// query cannot see a pad key. The returned cache still contains every
    /// position, including right padding. Callers trim each row to its real
    /// length.
    pub(crate) fn forward_collect(
        &self,
        hidden: Tensor<3>,
        lengths: &[usize],
    ) -> (Tensor<3>, LayerKv) {
        let [batch, seq, _] = hidden.dims();
        assert_eq!(lengths.len(), batch, "one length per row");
        let (query, key, value) = self.project(hidden.clone());
        let query = apply_rope(query, self.rope_spec());
        let key = apply_rope(key, self.rope_spec());
        let cache = LayerKv {
            key: key.clone(),
            value: value.clone(),
            pos0: 0,
        };
        let key = repeat_kv(key, self.groups());
        let value = repeat_kv(value, self.groups());
        let device = key.device();
        let mask = if lengths.iter().all(|len| *len == seq) {
            mask_tensor(seq, self.window(), &device)
        } else {
            let values = attention_mask_right(seq, self.window(), lengths);
            Tensor::<4>::from_data(TensorData::new(values, [batch, 1, seq, seq]), &device)
                .cast(FloatDType::F32)
        };
        (self.mix(query, key, value, mask, hidden), cache)
    }

    /// Forward with precomputed mask and RoPE tables already on the device.
    ///
    /// Avoids reallocating and transferring megabytes of mask and frequency
    /// tables from the host to the device on every single layer.
    pub(crate) fn forward_padded_with_context(
        &self,
        hidden: Tensor<3>,
        mask: &Tensor<4>,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
    ) -> Tensor<3> {
        let (query, key, value) = self.project(hidden.clone());
        let query = rotate_explicit_tensors(query, self.rope_spec(), cos, sin);
        let key = rotate_explicit_tensors(key, self.rope_spec(), cos, sin);
        let key = repeat_kv(key, self.groups());
        let value = repeat_kv(value, self.groups());
        self.mix(query, key, value, mask.clone(), hidden)
    }

    pub(crate) fn forward_decode_batched(
        &self,
        hidden: Tensor<3>,
        cache: &mut crate::model::BatchedLayerKv,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        mask: &Tensor<4>,
    ) -> Tensor<3> {
        let (query, key, value) = self.project(hidden.clone());
        let query = rotate_explicit_tensors(query, self.rope_spec(), cos, sin);
        let key = rotate_explicit_tensors(key, self.rope_spec(), cos, sin);

        let mut full_key = Tensor::cat(vec![cache.key.clone(), key], 2);
        let mut full_value = Tensor::cat(vec![cache.value.clone(), value], 2);
        let mut key_len = full_key.dims()[2];

        if self.sliding_window > 0 && key_len > self.sliding_window {
            let drop = key_len - self.sliding_window;
            full_key = full_key.narrow(2, drop, self.sliding_window);
            full_value = full_value.narrow(2, drop, self.sliding_window);
            key_len = self.sliding_window;
            for r in 0..cache.pad_prefix.len() {
                if cache.pad_prefix[r] >= drop {
                    cache.pad_prefix[r] -= drop;
                } else {
                    let real_drop = drop - cache.pad_prefix[r];
                    cache.pad_prefix[r] = 0;
                    cache.pos0[r] += real_drop;
                }
            }
        }
        let _ = key_len;
        cache.key = full_key;
        cache.value = full_value;

        let key = repeat_kv(cache.key.clone(), self.groups());
        let value = repeat_kv(cache.value.clone(), self.groups());
        self.mix(query, key, value, mask.clone(), hidden)
    }
}

pub(crate) fn rotate_explicit_tensors(
    x: Tensor<4>,
    spec: RopeSpec,
    cos: &Tensor<4>,
    sin: &Tensor<4>,
) -> Tensor<4> {
    let [_, _, _, head_dim] = x.dims();
    let rope_dim = spec.rotary_dim(head_dim).min(head_dim);
    if rope_dim == 0 {
        return x;
    }
    let dtype = x.dtype();
    let x32 = if dtype == DType::F32 {
        x
    } else {
        x.cast(FloatDType::F32)
    };
    let rotated = x32.clone().narrow(3, 0, rope_dim);
    let half = rope_dim / 2;
    let first = rotated.clone().narrow(3, 0, half);
    let second = rotated.clone().narrow(3, half, half);
    let spun = Tensor::cat(vec![second.neg(), first], 3);
    let rotated = rotated * cos.clone() + spun * sin.clone();
    let rotated = if dtype == DType::F32 {
        rotated
    } else {
        rotated.cast(dtype)
    };
    let x = if dtype == DType::F32 {
        x32
    } else {
        x32.cast(dtype)
    };
    if rope_dim == head_dim {
        return rotated;
    }
    let passed = x.narrow(3, rope_dim, head_dim - rope_dim);
    Tensor::cat(vec![rotated, passed], 3)
}

pub(crate) fn stack_caches(rows: &[LayerKv]) -> (Tensor<4>, Tensor<4>, Vec<usize>, Vec<usize>) {
    let max_len = rows.iter().map(|row| row.key.dims()[2]).max().unwrap_or(0);
    let mut keys = Vec::with_capacity(rows.len());
    let mut values = Vec::with_capacity(rows.len());
    let mut pad_prefix = Vec::with_capacity(rows.len());
    let mut key_pos0 = Vec::with_capacity(rows.len());
    for row in rows {
        let len = row.key.dims()[2];
        let pad = max_len - len;
        pad_prefix.push(pad);
        // Absolute position of stacked index 0. Pad keys are not real, so the
        // first real key still sits at `pos0` after `pad` columns.
        key_pos0.push(row.pos0);
        if pad == 0 {
            keys.push(row.key.clone());
            values.push(row.value.clone());
        } else {
            let [_, heads, _, dim] = row.key.dims();
            let key_pad =
                Tensor::<4>::zeros([1, heads, pad, dim], &row.key.device()).cast(row.key.dtype());
            let value_pad = Tensor::<4>::zeros([1, heads, pad, dim], &row.value.device())
                .cast(row.value.dtype());
            keys.push(Tensor::cat(vec![key_pad, row.key.clone()], 2));
            values.push(Tensor::cat(vec![value_pad, row.value.clone()], 2));
        }
    }
    (
        Tensor::cat(keys, 0),
        Tensor::cat(values, 0),
        pad_prefix,
        key_pos0,
    )
}

/// Additive mask `[batch, 1, 1, key]` for a single new query per row.
///
/// Stacked keys are left-padded to one width, so the mask is two cuts. Keys
/// before `pad_prefix` are padding and blocked. The remaining keys are causal
/// (and inside the sliding window) in absolute positions: real key `offset`
/// has position `key_pos0 + offset`. Every row's pad plus its real span has to
/// equal `key_len`, the stacked width. A shorter span would leave a column
/// unmasked.
pub(crate) fn decode_mask(
    query_pos: &[usize],
    pad_prefix: &[usize],
    key_pos0: &[usize],
    key_len: usize,
    window: Option<usize>,
    device: &Device,
) -> Tensor<4> {
    let batch = query_pos.len();
    for row in 0..batch {
        let span = padded_span(query_pos[row], key_pos0[row], pad_prefix[row]);
        assert_eq!(
            span, key_len,
            "row {row} covers {span} keys but the stacked block is {key_len}"
        );
    }
    let mut values = vec![0.0_f32; batch * key_len];
    for row in 0..batch {
        let pad = pad_prefix[row];
        let origin = key_pos0[row];
        let query = query_pos[row];
        let first = match window {
            Some(window) => query.saturating_sub(window - 1),
            None => 0,
        };
        for key in 0..key_len {
            let blocked = if key < pad {
                true
            } else {
                let absolute = origin + (key - pad);
                absolute > query || absolute < first
            };
            if blocked {
                values[row * key_len + key] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::<4>::from_data(TensorData::new(values, [batch, 1, 1, key_len]), device)
        .cast(FloatDType::F32)
}

/// Pad columns plus the real keys from the first cached position through the
/// query. This is the stacked width when every row was appended once.
fn padded_span(query: usize, origin: usize, pad: usize) -> usize {
    assert!(
        query >= origin,
        "query position {query} sits before the first real key at {origin}"
    );
    pad + (query - origin) + 1
}

/// Split a batched prefill cache into per-row caches, dropping right padding
/// and keys that sit outside the sliding window.
pub(crate) fn split_cache(
    cache: &LayerKv,
    lengths: &[usize],
    sliding_window: usize,
) -> Vec<LayerKv> {
    lengths
        .iter()
        .enumerate()
        .map(|(row, len)| {
            let mut key = cache.key.clone().narrow(0, row, 1).narrow(2, 0, *len);
            let mut value = cache.value.clone().narrow(0, row, 1).narrow(2, 0, *len);
            let mut pos0 = 0usize;
            if sliding_window > 0 && *len > sliding_window {
                let drop = *len - sliding_window;
                key = key.narrow(2, drop, sliding_window);
                value = value.narrow(2, drop, sliding_window);
                pos0 = drop;
            }
            LayerKv { key, value, pos0 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_window_blocks_tokens_outside_the_lookback() {
        let mask = attention_mask(5, Some(2));
        let at = |query: usize, key: usize| mask[query * 5 + key];
        assert!(at(3, 3).is_finite());
        assert!(at(3, 2).is_finite());
        assert!(at(3, 1).is_infinite());
        assert!(at(3, 4).is_infinite());
        assert!(at(0, 0).is_finite());
    }

    #[test]
    fn global_mask_is_causal() {
        let mask = attention_mask(3, None);
        assert!(mask[2 * 3].is_finite());
        assert!(mask[0 * 3 + 1].is_infinite());
    }

    #[test]
    fn a_zero_prefix_matches_the_square_mask() {
        let plain = attention_mask(5, Some(3));
        let padded = attention_mask_padded(5, Some(3), &[0, 0]);
        assert_eq!(&padded[..plain.len()], &plain[..]);
        assert_eq!(&padded[plain.len()..], &plain[..]);
    }

    #[test]
    fn right_pad_blocks_pad_keys_and_stays_causal() {
        // Row 0 has three real tokens in a width of five. Row 1 is full.
        let mask = attention_mask_right(5, Some(3), &[3, 5]);
        let at = |row: usize, query: usize, key: usize| mask[row * 25 + query * 5 + key];
        assert!(at(0, 2, 0).is_finite());
        assert!(at(0, 2, 2).is_finite());
        assert!(at(0, 2, 3).is_infinite(), "pad key inside a later column");
        assert!(at(0, 0, 1).is_infinite(), "causal still hides the future");
        assert!(
            at(0, 3, 3).is_finite(),
            "a pad query keeps one finite logit"
        );
        assert!(at(0, 3, 2).is_infinite());
        assert!(at(0, 3, 4).is_infinite());
        // Query 4 on the full row has a window of 3, so keys 0 and 1 are out.
        assert!(at(1, 4, 1).is_infinite());
        assert!(at(1, 4, 2).is_finite());
        assert!(at(1, 4, 4).is_finite());
        let plain = attention_mask(5, Some(3));
        assert_eq!(&mask[25..], &plain[..]);
    }

    #[test]
    fn left_pad_hides_the_prefix_and_keeps_real_positions() {
        // Real tokens sit at indexes 2, 3, 4 and act as positions 0, 1, 2.
        let mask = attention_mask_padded(5, Some(2), &[2]);
        let at = |query: usize, key: usize| mask[query * 5 + key];
        assert!(at(0, 0).is_finite(), "a pad query keeps one finite logit");
        assert!(at(0, 1).is_infinite());
        assert!(at(1, 1).is_finite());
        assert!(at(2, 0).is_infinite());
        assert!(at(2, 1).is_infinite());
        assert!(at(2, 2).is_finite());
        assert!(at(4, 3).is_finite());
        assert!(at(4, 2).is_infinite(), "window 2 drops position 0");
        assert!(at(4, 4).is_finite());
    }
}
