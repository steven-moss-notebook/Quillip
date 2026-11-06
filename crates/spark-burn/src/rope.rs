//! NeoX-style RoPE used by Spark.
//!
//! Frequencies are computed for the positions in the current sequence. Sliding
//! layers rotate the whole head with theta 10,000. Global layers rotate the
//! first quarter of the head with theta 5,000,000. The remaining dimensions
//! pass through unchanged.

use std::cell::RefCell;
use std::sync::{Mutex, OnceLock};

use burn::tensor::{DType, Device, FloatDType, Tensor, TensorData};

use crate::config::RopeSpec;

pub fn apply_rope(x: Tensor<4>, spec: RopeSpec) -> Tensor<4> {
    apply_rope_at(x, spec, 0)
}

/// Rotate `x` as if its first token sat at absolute position `pos0`.
pub fn apply_rope_at(x: Tensor<4>, spec: RopeSpec, pos0: usize) -> Tensor<4> {
    let seq = x.dims()[2];
    let positions: Vec<usize> = (pos0..pos0 + seq).collect();
    apply_rope_positions(x, spec, &positions, false)
}

/// Rotate each batch row. `positions` has one absolute position per row, and
/// the sequence axis is that row's contiguous run of the same length.
///
/// When `per_row` is set, `positions.len()` is the batch and each row has its
/// own start. Otherwise `positions` is the shared sequence `pos0..pos0+seq`.
#[allow(dead_code)]
pub fn apply_rope_rows(x: Tensor<4>, spec: RopeSpec, positions: &[usize]) -> Tensor<4> {
    apply_rope_positions(x, spec, positions, true)
}

fn apply_rope_positions(
    x: Tensor<4>,
    spec: RopeSpec,
    positions: &[usize],
    per_row: bool,
) -> Tensor<4> {
    let [batch, _, seq, head_dim] = x.dims();
    let rope_dim = spec.rotary_dim(head_dim).min(head_dim);
    if rope_dim == 0 {
        return x;
    }
    let rotated = rotate_neox(
        x.clone().narrow(3, 0, rope_dim),
        spec.theta,
        positions,
        per_row,
        batch,
        seq,
    );
    if rope_dim == head_dim {
        return rotated;
    }
    let passed = x.narrow(3, rope_dim, head_dim - rope_dim);
    Tensor::cat(vec![rotated, passed], 3)
}

struct DeviceRope {
    theta_bits: u32,
    start: usize,
    seq: usize,
    rope_dim: usize,
    device: Device,
    cos: Tensor<4>,
    sin: Tensor<4>,
}

thread_local! {
    static DEVICE_ROPE: RefCell<Vec<DeviceRope>> = const { RefCell::new(Vec::new()) };
}

fn rotate_neox(
    x: Tensor<4>,
    theta: f32,
    positions: &[usize],
    per_row: bool,
    batch: usize,
    seq: usize,
) -> Tensor<4> {
    let rope_dim = x.dims()[3];
    let half = rope_dim / 2;
    let dtype = x.dtype();
    let device = x.device();
    // Angles are built on the host in f32. The published forward also multiplies
    // them in f32 and only then casts back. bf16 cannot represent consecutive
    // positions past 256, and a bf16 multiply of a full head erases the turn.
    let (cos, sin) = if per_row {
        assert_eq!(positions.len(), batch, "one rope position per batch row");
        if positions.iter().all(|&p| p == positions[0]) {
            let start = positions[0];
            let theta_bits = theta.to_bits();
            DEVICE_ROPE.with(|cell| {
                let mut cache = cell.borrow_mut();
                if let Some(found) = cache.iter().find(|item| {
                    item.theta_bits == theta_bits
                        && item.start == start
                        && item.seq == seq
                        && item.rope_dim == rope_dim
                        && item.device == device
                }) {
                    (found.cos.clone(), found.sin.clone())
                } else {
                    let (cos_data, sin_data) = cached_rows(theta, start, seq, rope_dim);
                    let cos = Tensor::<4>::from_data(
                        TensorData::new(cos_data, [1, 1, seq, rope_dim]),
                        &device,
                    )
                    .cast(FloatDType::F32);
                    let sin = Tensor::<4>::from_data(
                        TensorData::new(sin_data, [1, 1, seq, rope_dim]),
                        &device,
                    )
                    .cast(FloatDType::F32);
                    if cache.len() >= 8 {
                        cache.remove(0);
                    }
                    cache.push(DeviceRope {
                        theta_bits,
                        start,
                        seq,
                        rope_dim,
                        device: device.clone(),
                        cos: cos.clone(),
                        sin: sin.clone(),
                    });
                    (cos, sin)
                }
            })
        } else {
            let (cos_data, sin_data) = cached_rows_batch(theta, positions, seq, rope_dim);
            let cos = Tensor::<4>::from_data(
                TensorData::new(cos_data, [batch, 1, seq, rope_dim]),
                &device,
            )
            .cast(FloatDType::F32);
            let sin = Tensor::<4>::from_data(
                TensorData::new(sin_data, [batch, 1, seq, rope_dim]),
                &device,
            )
            .cast(FloatDType::F32);
            (cos, sin)
        }
    } else {
        assert_eq!(positions.len(), seq, "rope positions cover the sequence");
        let start = positions.first().copied().unwrap_or(0);
        let theta_bits = theta.to_bits();
        DEVICE_ROPE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if let Some(found) = cache.iter().find(|item| {
                item.theta_bits == theta_bits
                    && item.start == start
                    && item.seq == seq
                    && item.rope_dim == rope_dim
                    && item.device == device
            }) {
                (found.cos.clone(), found.sin.clone())
            } else {
                let (cos_data, sin_data) = cached_rows(theta, start, seq, rope_dim);
                let cos = Tensor::<4>::from_data(
                    TensorData::new(cos_data, [1, 1, seq, rope_dim]),
                    &device,
                )
                .cast(FloatDType::F32);
                let sin = Tensor::<4>::from_data(
                    TensorData::new(sin_data, [1, 1, seq, rope_dim]),
                    &device,
                )
                .cast(FloatDType::F32);
                if cache.len() >= 8 {
                    cache.remove(0);
                }
                cache.push(DeviceRope {
                    theta_bits,
                    start,
                    seq,
                    rope_dim,
                    device: device.clone(),
                    cos: cos.clone(),
                    sin: sin.clone(),
                });
                (cos, sin)
            }
        })
    };
    let x32 = if dtype == DType::F32 {
        x
    } else {
        x.cast(FloatDType::F32)
    };
    let first = x32.clone().narrow(3, 0, half);
    let second = x32.clone().narrow(3, half, half);
    let rotated = Tensor::cat(vec![second.neg(), first], 3);
    let out = x32 * cos + rotated * sin;
    if dtype == DType::F32 {
        out
    } else {
        out.cast(dtype)
    }
}

pub(crate) fn rope_tensors_for_positions(
    theta: f32,
    rope_dim: usize,
    positions: &[usize],
    device: &Device,
) -> (Tensor<4>, Tensor<4>) {
    if positions.is_empty() || rope_dim == 0 {
        let empty = Tensor::<4>::zeros([1, 1, 1, rope_dim], device).cast(FloatDType::F32);
        return (empty.clone(), empty);
    }
    let all_equal = positions.iter().all(|&p| p == positions[0]);
    if all_equal {
        let start = positions[0];
        let theta_bits = theta.to_bits();
        DEVICE_ROPE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if let Some(found) = cache.iter().find(|item| {
                item.theta_bits == theta_bits
                    && item.start == start
                    && item.seq == 1
                    && item.rope_dim == rope_dim
                    && item.device == *device
            }) {
                (found.cos.clone(), found.sin.clone())
            } else {
                let (cos_data, sin_data) = cached_rows(theta, start, 1, rope_dim);
                let cos =
                    Tensor::<4>::from_data(TensorData::new(cos_data, [1, 1, 1, rope_dim]), device)
                        .cast(FloatDType::F32);
                let sin =
                    Tensor::<4>::from_data(TensorData::new(sin_data, [1, 1, 1, rope_dim]), device)
                        .cast(FloatDType::F32);
                if cache.len() >= 8 {
                    cache.remove(0);
                }
                cache.push(DeviceRope {
                    theta_bits,
                    start,
                    seq: 1,
                    rope_dim,
                    device: device.clone(),
                    cos: cos.clone(),
                    sin: sin.clone(),
                });
                (cos, sin)
            }
        })
    } else {
        let batch = positions.len();
        let (cos_data, sin_data) = cached_rows_batch(theta, positions, 1, rope_dim);
        let cos =
            Tensor::<4>::from_data(TensorData::new(cos_data, [batch, 1, 1, rope_dim]), device)
                .cast(FloatDType::F32);
        let sin =
            Tensor::<4>::from_data(TensorData::new(sin_data, [batch, 1, 1, rope_dim]), device)
                .cast(FloatDType::F32);
        (cos, sin)
    }
}

struct RopeTable {
    theta_bits: u32,
    rope_dim: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

fn rope_cache() -> &'static Mutex<Vec<RopeTable>> {
    static CACHE: OnceLock<Mutex<Vec<RopeTable>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn ensure_table<'a>(
    cache: &'a mut Vec<RopeTable>,
    theta: f32,
    rope_dim: usize,
    end: usize,
) -> &'a RopeTable {
    let theta_bits = theta.to_bits();
    let entry = if let Some(index) = cache
        .iter()
        .position(|table| table.theta_bits == theta_bits && table.rope_dim == rope_dim)
    {
        index
    } else {
        cache.push(RopeTable {
            theta_bits,
            rope_dim,
            cos: Vec::new(),
            sin: Vec::new(),
        });
        cache.len() - 1
    };
    let table = &mut cache[entry];
    let have = table.cos.len() / rope_dim;
    if end > have {
        let pairs = rope_dim / 2;
        let ln_theta = theta.ln();
        table.cos.resize(end * rope_dim, 0.0);
        table.sin.resize(end * rope_dim, 0.0);
        for pair in 0..pairs {
            let exponent = (2 * pair) as f32 / rope_dim as f32;
            let inv_freq = (-exponent * ln_theta).exp();
            for position in have..end {
                let angle = position as f32 * inv_freq;
                let (angle_cos, angle_sin) = (angle.cos(), angle.sin());
                let row = position * rope_dim;
                table.cos[row + pair] = angle_cos;
                table.sin[row + pair] = angle_sin;
                table.cos[row + pairs + pair] = angle_cos;
                table.sin[row + pairs + pair] = angle_sin;
            }
        }
    }
    table
}

/// Batch version of [`cached_rows`] that takes the cache lock once and copies
/// contiguous slices for each start position.
pub(crate) fn cached_rows_batch(
    theta: f32,
    starts: &[usize],
    len: usize,
    rope_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    if starts.is_empty() || len == 0 || rope_dim == 0 {
        return (Vec::new(), Vec::new());
    }
    let max_end = starts.iter().map(|&start| start + len).max().unwrap_or(0);
    let mut cache = rope_cache().lock().expect("rope cache");
    let table = ensure_table(&mut cache, theta, rope_dim, max_end);
    let mut cos = Vec::with_capacity(starts.len() * len * rope_dim);
    let mut sin = Vec::with_capacity(starts.len() * len * rope_dim);
    for &start in starts {
        let from = start * rope_dim;
        let to = (start + len) * rope_dim;
        cos.extend_from_slice(&table.cos[from..to]);
        sin.extend_from_slice(&table.sin[from..to]);
    }
    (cos, sin)
}

/// Rows `start..start+len` of the NeoX table. Repeated calls reuse positions
/// that were already filled.
pub(crate) fn cached_rows(
    theta: f32,
    start: usize,
    len: usize,
    rope_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    if len == 0 || rope_dim == 0 {
        return (Vec::new(), Vec::new());
    }
    let end = start + len;
    let mut cache = rope_cache().lock().expect("rope cache");
    let table = ensure_table(&mut cache, theta, rope_dim, end);
    let from = start * rope_dim;
    let to = end * rope_dim;
    (table.cos[from..to].to_vec(), table.sin[from..to].to_vec())
}

/// Precompute left-padded RoPE tables for a batch.
///
/// Each row has `pad_prefix[row]` zeros at the start, followed by
/// positions 0..seq - pad - 1.
pub(crate) fn cached_rows_padded(
    theta: f32,
    seq: usize,
    rope_dim: usize,
    pad_prefix: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let batch = pad_prefix.len();
    if batch == 0 || seq == 0 || rope_dim == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut cache = rope_cache().lock().expect("rope cache");
    let table = ensure_table(&mut cache, theta, rope_dim, seq);
    let mut cos = Vec::with_capacity(batch * seq * rope_dim);
    let mut sin = Vec::with_capacity(batch * seq * rope_dim);
    let pos0_cos = &table.cos[0..rope_dim];
    let pos0_sin = &table.sin[0..rope_dim];
    for &pad in pad_prefix {
        let pad = pad.min(seq);
        for _ in 0..pad {
            cos.extend_from_slice(pos0_cos);
            sin.extend_from_slice(pos0_sin);
        }
        let real = seq - pad;
        if real > 0 {
            cos.extend_from_slice(&table.cos[0..real * rope_dim]);
            sin.extend_from_slice(&table.sin[0..real * rope_dim]);
        }
    }
    (cos, sin)
}

/// Cosine and sine tables, NeoX layout: each frequency is repeated in the
/// second half of the head. Computed in f32 so `theta.ln()` and the position
/// index stay exact.
#[cfg(test)]
fn rope_tables(theta: f32, seq: usize, rope_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let pairs = rope_dim / 2;
    let mut cos = vec![0.0_f32; seq * rope_dim];
    let mut sin = vec![0.0_f32; seq * rope_dim];
    let ln_theta = theta.ln();
    for pair in 0..pairs {
        let exponent = (2 * pair) as f32 / rope_dim as f32;
        let inv_freq = (-exponent * ln_theta).exp();
        for position in 0..seq {
            let angle = position as f32 * inv_freq;
            let (angle_cos, angle_sin) = (angle.cos(), angle.sin());
            let row = position * rope_dim;
            cos[row + pair] = angle_cos;
            sin[row + pair] = angle_sin;
            cos[row + pairs + pair] = angle_cos;
            sin[row + pairs + pair] = angle_sin;
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[test]
    fn position_zero_is_the_identity_and_a_partial_tail_is_unchanged() {
        let device = Device::flex();
        let spec = RopeSpec {
            theta: 10_000.0,
            partial_rotary_factor: 0.5,
        };
        let values = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = Tensor::<4>::from_data(TensorData::new(values.clone(), [1, 1, 2, 4]), &device);
        let output = apply_rope(input, spec).into_data();
        let output = output.as_slice::<f32>().unwrap();
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
        assert!((output[2] - 3.0).abs() < 1e-5);
        assert!((output[3] - 4.0).abs() < 1e-5);

        let angle = 1.0_f32;
        let (cos, sin) = (angle.cos(), angle.sin());
        let rotated_a = 5.0 * cos - 6.0 * sin;
        let rotated_b = 6.0 * cos + 5.0 * sin;
        assert!((output[4] - rotated_a).abs() < 1e-5);
        assert!((output[5] - rotated_b).abs() < 1e-5);
        assert!((output[6] - 7.0).abs() < 1e-5);
        assert!((output[7] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn adjacent_positions_past_256_keep_distinct_angles() {
        let (cos, _) = super::rope_tables(10_000.0, 1026, 4);
        let at = |position: usize| cos[position * 4];
        assert!((at(1024) - 1024.0_f32.cos()).abs() < 1e-3);
        assert!((at(1025) - 1025.0_f32.cos()).abs() < 1e-3);
        assert!(
            (at(1024) - at(1025)).abs() > 1e-3,
            "positions 1024 and 1025 collapsed to the same angle"
        );
    }

    #[test]
    fn a_bf16_activation_rotates_as_f32() {
        let device = Device::flex();
        let spec = RopeSpec {
            theta: 10_000.0,
            partial_rotary_factor: 1.0,
        };
        let values = vec![100.0_f32, -80.0, 40.0, -20.0];
        let reference =
            Tensor::<4>::from_data(TensorData::new(values.clone(), [1, 1, 2, 2]), &device);
        // The product is f32. The published forward then stores it in the
        // activation dtype, so the bf16 result is that product rounded to bf16.
        let expected = apply_rope(reference, spec)
            .cast(burn::tensor::FloatDType::BF16)
            .cast(burn::tensor::FloatDType::F32)
            .into_data();
        let expected = expected.as_slice::<f32>().unwrap();
        let low = Tensor::<4>::from_data(TensorData::new(values, [1, 1, 2, 2]), &device)
            .cast(burn::tensor::FloatDType::BF16);
        let got = apply_rope(low, spec)
            .cast(burn::tensor::FloatDType::F32)
            .into_data();
        let got = got.as_slice::<f32>().unwrap();
        for (left, right) in got.iter().zip(expected) {
            assert!(
                (left - right).abs() < 1e-3,
                "bf16 rope left the f32 turn, {left} vs {right}"
            );
        }
    }

    #[test]
    fn cached_rows_match_the_full_table() {
        let (full_cos, full_sin) = super::rope_tables(10_000.0, 6, 4);
        let (cached_cos, cached_sin) = super::cached_rows(10_000.0, 2, 3, 4);
        assert_eq!(cached_cos, full_cos[2 * 4..5 * 4]);
        assert_eq!(cached_sin, full_sin[2 * 4..5 * 4]);
    }

    #[test]
    fn cached_rows_padded_matches_individual_rows() {
        let seq = 6;
        let rope_dim = 4;
        let theta = 10_000.0;
        let pad_prefix = [2, 0, 4];
        let (padded_cos, padded_sin) = super::cached_rows_padded(theta, seq, rope_dim, &pad_prefix);
        let mut expected_cos = Vec::new();
        let mut expected_sin = Vec::new();
        for &pad in &pad_prefix {
            for idx in 0..seq {
                let pos = idx.saturating_sub(pad);
                let (c, s) = super::cached_rows(theta, pos, 1, rope_dim);
                expected_cos.extend(c);
                expected_sin.extend(s);
            }
        }
        assert_eq!(padded_cos, expected_cos);
        assert_eq!(padded_sin, expected_sin);
    }
}
