//! Decoder stack: pre-norm attention, GELU-gated MLP, tied embeddings.

use burn::nn::{
    Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig,
};
use burn::tensor::activation::gelu;
use burn::tensor::{DType, Device, FloatDType, Int, Tensor, TensorData};

use crate::attention::{LayerKv, SparkAttention, split_cache};
use crate::config::SparkConfig;

/// Vocabulary columns in one log-sum-exp chunk. 8,192 keeps the matmul dense
/// and stays well under a full 131,072-wide logit tensor.
const VOCAB_CHUNK: usize = 8192;

#[derive(burn::module::Module, Debug)]
pub struct SparkMlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl SparkMlp {
    fn new(
        hidden: usize,
        intermediate: usize,
        bias: bool,
        initializer: Initializer,
        device: &Device,
    ) -> Self {
        Self {
            gate: LinearConfig::new(hidden, intermediate)
                .with_bias(bias)
                .with_initializer(initializer.clone())
                .init(device),
            up: LinearConfig::new(hidden, intermediate)
                .with_bias(bias)
                .with_initializer(initializer.clone())
                .init(device),
            down: LinearConfig::new(intermediate, hidden)
                .with_bias(bias)
                .with_initializer(initializer)
                .init(device),
        }
    }

    fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        let activated = gelu(self.gate.forward(hidden.clone())) * self.up.forward(hidden);
        self.down.forward(activated)
    }
}

#[derive(burn::module::Module, Debug)]
pub struct SparkLayer {
    pub input_norm: RmsNorm,
    pub attn: SparkAttention,
    pub post_norm: RmsNorm,
    pub mlp: SparkMlp,
}

impl SparkLayer {
    /// Checkpoint dtype of the linears. The residual itself is fp32.
    fn compute_dtype(&self) -> DType {
        self.mlp.gate.weight.val().dtype()
    }

    fn norm_for_attention(&self, hidden: Tensor<3>, compute: DType) -> (Tensor<3>, Tensor<3>) {
        let hidden = cast_dtype(hidden, DType::F32);
        let normed = cast_dtype(norm_residual(hidden.clone(), &self.input_norm), compute);
        (hidden, normed)
    }

    fn add_block(&self, residual: Tensor<3>, attention: Tensor<3>, compute: DType) -> Tensor<3> {
        let hidden = residual + cast_dtype(attention, DType::F32);
        let normed = cast_dtype(norm_residual(hidden.clone(), &self.post_norm), compute);
        let feed_forward = self.mlp.forward(normed);
        hidden + cast_dtype(feed_forward, DType::F32)
    }

    fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        let compute = self.compute_dtype();
        let (residual, normed) = self.norm_for_attention(hidden, compute);
        let attention = self.attn.forward(normed);
        self.add_block(residual, attention, compute)
    }

    fn forward_collect(&self, hidden: Tensor<3>, lengths: &[usize]) -> (Tensor<3>, LayerKv) {
        let compute = self.compute_dtype();
        let (residual, normed) = self.norm_for_attention(hidden, compute);
        let (attention, cache) = self.attn.forward_collect(normed, lengths);
        (self.add_block(residual, attention, compute), cache)
    }

    fn forward_padded_with_context(
        &self,
        hidden: Tensor<3>,
        mask: &Tensor<4>,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
    ) -> Tensor<3> {
        let compute = self.compute_dtype();
        let (residual, normed) = self.norm_for_attention(hidden, compute);
        let attention = self
            .attn
            .forward_padded_with_context(normed, mask, cos, sin);
        self.add_block(residual, attention, compute)
    }

    fn forward_decode_batched(
        &self,
        hidden: Tensor<3>,
        layer_cache: &mut BatchedLayerKv,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        mask: &Tensor<4>,
    ) -> Tensor<3> {
        let compute = self.compute_dtype();
        let (residual, normed) = self.norm_for_attention(hidden, compute);
        let attention = self
            .attn
            .forward_decode_batched(normed, layer_cache, cos, sin, mask);
        self.add_block(residual, attention, compute)
    }
}

/// RMSNorm in fp32, matching the published Spark residual.
///
/// Burn's module multiplies in the activation dtype. The checkpoint stores
/// the norm scale in bf16, and the published forward upcasts both the residual
/// and that scale before the multiply. When they are already the same dtype,
/// the module itself runs so the f32 tests stay on that path.
fn norm_residual(x: Tensor<3>, norm: &RmsNorm) -> Tensor<3> {
    let gamma = norm.gamma.val();
    let x = cast_dtype(x, DType::F32);
    if gamma.dtype() == DType::F32 {
        return norm.forward(x);
    }
    let rms = (x.clone().square().mean_dim(2) + norm.epsilon).sqrt();
    (x / rms) * gamma.cast(FloatDType::F32).unsqueeze::<3>()
}

fn cast_dtype<const D: usize>(tensor: Tensor<D>, dtype: DType) -> Tensor<D> {
    if tensor.dtype() == dtype {
        tensor
    } else {
        tensor.cast(dtype)
    }
}

struct PaddedLayerInputs {
    window: Option<usize>,
    theta_bits: u32,
    mask: Tensor<4>,
    cos: Tensor<4>,
    sin: Tensor<4>,
}

/// Per-row keys and values for a cached decode.
#[derive(Clone)]
pub struct ModelCache {
    layers: Vec<LayerKv>,
}

/// Batched keys and values across active decode rows.
///
/// Keeps keys and values as contiguous [batch, kv_heads, seq_len, dim] tensors,
/// eliminating millions of per-row slice, concatenate, and re-stack operations
/// on the GPU.
#[derive(Clone)]
pub struct BatchedLayerKv {
    pub key: Tensor<4>,
    pub value: Tensor<4>,
    pub pos0: Vec<usize>,
    pub pad_prefix: Vec<usize>,
}

#[derive(Clone)]
pub struct BatchedModelCache {
    pub layers: Vec<BatchedLayerKv>,
}

impl BatchedModelCache {
    pub fn from_model_caches(caches: &[ModelCache]) -> Self {
        assert!(!caches.is_empty(), "caches cannot be empty");
        let n_layers = caches[0].layers.len();
        let mut layers = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let layer_rows: Vec<LayerKv> =
                caches.iter().map(|c| c.layers[layer_idx].clone()).collect();
            let (key, value, pad_prefix, pos0) = crate::attention::stack_caches(&layer_rows);
            layers.push(BatchedLayerKv {
                key,
                value,
                pos0,
                pad_prefix,
            });
        }
        BatchedModelCache { layers }
    }

    pub fn to_model_caches(&self) -> Vec<ModelCache> {
        let batch = self.layers[0].pos0.len();
        let n_layers = self.layers.len();
        let mut result = Vec::with_capacity(batch);
        for r in 0..batch {
            let mut model_cache_layers = Vec::with_capacity(n_layers);
            for layer in &self.layers {
                let pad = layer.pad_prefix[r];
                let key_len = layer.key.dims()[2];
                let real_len = key_len - pad;
                let key_row = layer.key.clone().narrow(0, r, 1);
                let val_row = layer.value.clone().narrow(0, r, 1);
                let key = if pad == 0 {
                    key_row
                } else {
                    key_row.narrow(2, pad, real_len)
                };
                let value = if pad == 0 {
                    val_row
                } else {
                    val_row.narrow(2, pad, real_len)
                };
                model_cache_layers.push(LayerKv {
                    key,
                    value,
                    pos0: layer.pos0[r],
                });
            }
            result.push(ModelCache {
                layers: model_cache_layers,
            });
        }
        result
    }

    pub fn filter_active(&self, active_slots: &[usize]) -> Self {
        let n_layers = self.layers.len();
        let mut layers = Vec::with_capacity(n_layers);
        for layer in &self.layers {
            let key_rows: Vec<Tensor<4>> = active_slots
                .iter()
                .map(|&i| layer.key.clone().narrow(0, i, 1))
                .collect();
            let val_rows: Vec<Tensor<4>> = active_slots
                .iter()
                .map(|&i| layer.value.clone().narrow(0, i, 1))
                .collect();
            let key = Tensor::cat(key_rows, 0);
            let value = Tensor::cat(val_rows, 0);
            let pos0 = active_slots.iter().map(|&i| layer.pos0[i]).collect();
            let pad_prefix = active_slots.iter().map(|&i| layer.pad_prefix[i]).collect();
            layers.push(BatchedLayerKv {
                key,
                value,
                pos0,
                pad_prefix,
            });
        }
        BatchedModelCache { layers }
    }
}

#[derive(burn::module::Module, Debug)]
pub struct SparkModel {
    pub embed: Embedding,
    pub layers: Vec<SparkLayer>,
    pub norm: RmsNorm,
    pub max_position_embeddings: usize,
    pub pad_token_id: usize,
}

impl SparkModel {
    pub fn new(config: &SparkConfig, device: &Device) -> Self {
        config
            .validate()
            .expect("SparkConfig must be validated before building the model");
        let initializer = Initializer::Normal {
            mean: 0.0,
            std: config.initializer_range,
        };
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                let sliding_window = if config.is_global_layer(index) {
                    0
                } else {
                    config.sliding_window
                };
                SparkLayer {
                    input_norm: RmsNormConfig::new(config.hidden_size)
                        .with_epsilon(config.rms_norm_eps)
                        .init(device),
                    attn: SparkAttention::new(
                        config.hidden_size,
                        config.num_attention_heads,
                        config.num_key_value_heads,
                        config.head_dim,
                        sliding_window,
                        config.attention_bias,
                        initializer.clone(),
                        config.rope_spec(index),
                        device,
                    ),
                    post_norm: RmsNormConfig::new(config.hidden_size)
                        .with_epsilon(config.rms_norm_eps)
                        .init(device),
                    mlp: SparkMlp::new(
                        config.hidden_size,
                        config.intermediate_size,
                        config.mlp_bias,
                        initializer.clone(),
                        device,
                    ),
                }
            })
            .collect();
        Self {
            embed: EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .with_initializer(initializer)
                .init(device),
            layers,
            norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            max_position_embeddings: config.max_position_embeddings,
            pad_token_id: config.pad_token_id,
        }
        .zero_pad_embedding()
    }

    /// Write a zero into the pad row of the frozen embedding table.
    ///
    /// The forward used to copy the whole vocabulary on every token to clear
    /// that one row. Embeddings are not trained, so doing it once is the same
    /// table the lookup and the tied output projection both read.
    pub fn zero_pad_embedding(mut self) -> Self {
        let pad_id = self.pad_token_id;
        self.embed.weight = self.embed.weight.map(|weight| {
            let [rows, _] = weight.dims();
            if pad_id >= rows {
                return weight;
            }
            let cleared = weight.clone().narrow(0, pad_id, 1).zeros_like();
            weight.slice_assign([pad_id..pad_id + 1], cleared)
        });
        self
    }

    fn embedding_weight(&self) -> Tensor<2> {
        self.embed.weight.val()
    }

    /// Final norm, then back to the checkpoint dtype for the tied head.
    ///
    /// The published model runs the stack in fp32 and casts to the embedding
    /// dtype only here. Leaving the residual in bf16 collapses it onto the
    /// vocabulary rows with the largest magnitude. In this tokenizer those
    /// rows are newlines, colons, and CJK pieces, and generation repeats them.
    fn finish(&self, hidden: Tensor<3>) -> Tensor<3> {
        let compute = self.embed.weight.val().dtype();
        cast_dtype(norm_residual(hidden, &self.norm), compute)
    }

    fn embed_tokens(&self, token_ids: Tensor<2, Int>) -> Tensor<3> {
        let seq = token_ids.dims()[1];
        assert!(
            seq >= 1 && seq <= self.max_position_embeddings,
            "sequence length {seq} is outside 1..={}",
            self.max_position_embeddings
        );
        burn::tensor::module::embedding(self.embedding_weight(), token_ids)
    }

    fn token_hidden(&self, token_ids: Tensor<2, Int>) -> Tensor<3> {
        let mut hidden = self.embed_tokens(token_ids);
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        self.finish(hidden)
    }

    /// Final residual at every position, after the last norm.
    ///
    /// This is the vector the tied head reads. It is not the frozen input
    /// embedding row: attention has already mixed the token with its context.
    pub fn residual_states(&self, token_ids: Tensor<2, Int>) -> Tensor<3> {
        self.token_hidden(token_ids)
    }

    fn token_hidden_collect(
        &self,
        token_ids: Tensor<2, Int>,
        lengths: &[usize],
    ) -> (Tensor<3>, Vec<LayerKv>) {
        let mut hidden = self.embed_tokens(token_ids);
        let mut caches = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (next, cache) = layer.forward_collect(hidden, lengths);
            hidden = next;
            caches.push(cache);
        }
        (self.finish(hidden), caches)
    }

    fn token_hidden_padded(&self, token_ids: Tensor<2, Int>, pad_prefix: &[usize]) -> Tensor<3> {
        if pad_prefix.iter().all(|pad| *pad == 0) {
            return self.token_hidden(token_ids);
        }
        let [batch, seq] = token_ids.dims();
        let device = token_ids.device();
        let mut precomputed: Vec<PaddedLayerInputs> = Vec::new();
        for layer in &self.layers {
            let window = layer.attn.window();
            let theta_bits = layer.attn.rope_theta.to_bits();
            if !precomputed
                .iter()
                .any(|entry| entry.window == window && entry.theta_bits == theta_bits)
            {
                let mask_values = crate::attention::attention_mask_padded(seq, window, pad_prefix);
                let mask = Tensor::<4>::from_data(
                    TensorData::new(mask_values, [batch, 1, seq, seq]),
                    &device,
                )
                .cast(FloatDType::F32);
                let rope = layer.attn.rope_spec();
                let rope_dim = rope
                    .rotary_dim(layer.attn.head_dim)
                    .min(layer.attn.head_dim);
                let (cos_values, sin_values) = crate::rope::cached_rows_padded(
                    layer.attn.rope_theta,
                    seq,
                    rope_dim,
                    pad_prefix,
                );
                let cos = Tensor::<4>::from_data(
                    TensorData::new(cos_values, [batch, 1, seq, rope_dim]),
                    &device,
                )
                .cast(FloatDType::F32);
                let sin = Tensor::<4>::from_data(
                    TensorData::new(sin_values, [batch, 1, seq, rope_dim]),
                    &device,
                )
                .cast(FloatDType::F32);
                precomputed.push(PaddedLayerInputs {
                    window,
                    theta_bits,
                    mask,
                    cos,
                    sin,
                });
            }
        }
        let mut hidden = self.embed_tokens(token_ids);
        for layer in &self.layers {
            let window = layer.attn.window();
            let theta_bits = layer.attn.rope_theta.to_bits();
            let entry = precomputed
                .iter()
                .find(|e| e.window == window && e.theta_bits == theta_bits)
                .expect("precomputed inputs cover all layers");
            hidden = layer.forward_padded_with_context(hidden, &entry.mask, &entry.cos, &entry.sin);
        }
        self.finish(hidden)
    }

    pub fn logits(&self, token_ids: Tensor<2, Int>) -> Tensor<3> {
        let hidden = self.token_hidden(token_ids);
        // Tied embeddings: the input table is also the output projection.
        // Burn's matmul is same-rank, so the token axis is folded for the multiply.
        let [batch, seq, hidden_dim] = hidden.dims();
        let weight = self.embedding_weight().transpose();
        let vocab = weight.dims()[1];
        hidden
            .reshape([batch * seq, hidden_dim])
            .matmul(weight)
            .reshape([batch, seq, vocab])
    }

    /// Logits for the final position only.
    ///
    /// Generation and attribute scoring read one next-token distribution.
    /// Projecting the last hidden state avoids a vocabulary matmul at every
    /// earlier position.
    pub fn last_logits(&self, token_ids: Tensor<2, Int>) -> Tensor<2> {
        let hidden = self.token_hidden(token_ids);
        let seq = hidden.dims()[1];
        self.project_positions(hidden, &[seq - 1])
    }

    /// Vocabulary logits at selected positions, from one forward.
    ///
    /// The rows are in `positions` order. Scoring a rubric reads every
    /// criterion from this tensor instead of running the model once per line.
    pub fn position_logits(&self, token_ids: Tensor<2, Int>, positions: &[usize]) -> Tensor<2> {
        self.position_logits_padded(token_ids, positions, &[])
    }

    /// [`position_logits`](Self::position_logits) with a left-pad prefix.
    ///
    /// `pad_prefix[row]` keys at the start of that row are invisible, and the
    /// first real token keeps position 0. An empty prefix is the unpadded
    /// forward. Shared `positions` are indexes into the padded sequence, and
    /// the result is `[batch * positions, vocab]` with each row's positions
    /// packed together.
    pub fn position_logits_padded(
        &self,
        token_ids: Tensor<2, Int>,
        positions: &[usize],
        pad_prefix: &[usize],
    ) -> Tensor<2> {
        let batch = token_ids.dims()[0];
        let prefix = if pad_prefix.is_empty() {
            vec![0; batch]
        } else {
            assert_eq!(pad_prefix.len(), batch, "pad prefix covers the batch");
            pad_prefix.to_vec()
        };
        let hidden = self.token_hidden_padded(token_ids, &prefix);
        let seq = hidden.dims()[1];
        assert!(
            !positions.is_empty() && positions.iter().all(|position| *position < seq),
            "logit positions must sit inside the sequence"
        );
        self.project_positions(hidden, positions)
    }

    /// Logits at `positions` for only the specified `digits`, with a left-pad prefix.
    ///
    /// The result is `[batch * positions, 6]`.
    pub fn digit_logits_padded(
        &self,
        token_ids: Tensor<2, Int>,
        positions: &[usize],
        pad_prefix: &[usize],
        digits: &[usize; 6],
    ) -> Tensor<2> {
        let batch = token_ids.dims()[0];
        let prefix = if pad_prefix.is_empty() {
            vec![0; batch]
        } else {
            assert_eq!(pad_prefix.len(), batch, "pad prefix covers the batch");
            pad_prefix.to_vec()
        };
        let hidden = self.token_hidden_padded(token_ids, &prefix);
        let seq = hidden.dims()[1];
        assert!(
            !positions.is_empty() && positions.iter().all(|position| *position < seq),
            "logit positions must sit inside the sequence"
        );
        self.project_digit_positions(hidden, positions, digits)
    }

    /// Logits at per-row `positions` for only the specified `digits`, with a left-pad prefix.
    ///
    /// `row_positions[row]` has the criteria token indices for row `row`.
    /// The result has shape `[total_positions, 6]`.
    pub fn digit_logits_padded_per_row(
        &self,
        token_ids: Tensor<2, Int>,
        row_positions: &[Vec<usize>],
        pad_prefix: &[usize],
        digits: &[usize; 6],
    ) -> Tensor<2> {
        let batch = token_ids.dims()[0];
        assert_eq!(row_positions.len(), batch, "one position list per row");
        let prefix = if pad_prefix.is_empty() {
            vec![0; batch]
        } else {
            assert_eq!(pad_prefix.len(), batch, "pad prefix covers the batch");
            pad_prefix.to_vec()
        };
        let hidden = self.token_hidden_padded(token_ids, &prefix);
        let seq = hidden.dims()[1];
        for (row_idx, pos_list) in row_positions.iter().enumerate() {
            assert!(
                !pos_list.is_empty() && pos_list.iter().all(|position| *position < seq),
                "logit positions must sit inside the sequence for row {row_idx}"
            );
        }
        self.project_digit_positions_per_row(hidden, row_positions, digits)
    }

    fn project_digit_positions_per_row(
        &self,
        hidden: Tensor<3>,
        row_positions: &[Vec<usize>],
        digits: &[usize; 6],
    ) -> Tensor<2> {
        let batch = row_positions.len();
        let mut row_tensors = Vec::with_capacity(batch);
        for (row_idx, positions) in row_positions.iter().enumerate() {
            let row_hidden = hidden.clone().narrow(0, row_idx, 1);
            let pos_slices = positions
                .iter()
                .map(|&p| row_hidden.clone().narrow(1, p, 1))
                .collect::<Vec<_>>();
            let row_stacked = Tensor::cat(pos_slices, 1);
            let [_, count, hidden_dim] = row_stacked.dims();
            row_tensors.push(row_stacked.reshape([count, hidden_dim]));
        }
        let stacked = Tensor::cat(row_tensors, 0);
        let embed = self.embedding_weight();
        let digit_rows = digits
            .iter()
            .map(|&id| embed.clone().narrow(0, id, 1))
            .collect::<Vec<_>>();
        let digit_weight = Tensor::cat(digit_rows, 0).transpose();
        stacked.matmul(digit_weight)
    }

    fn project_digit_positions(
        &self,
        hidden: Tensor<3>,
        positions: &[usize],
        digits: &[usize; 6],
    ) -> Tensor<2> {
        let rows = positions
            .iter()
            .map(|position| hidden.clone().narrow(1, *position, 1))
            .collect::<Vec<_>>();
        let stacked = Tensor::cat(rows, 1);
        let [batch, count, hidden_dim] = stacked.dims();
        let embed = self.embedding_weight();
        let digit_rows = digits
            .iter()
            .map(|&id| embed.clone().narrow(0, id, 1))
            .collect::<Vec<_>>();
        let digit_weight = Tensor::cat(digit_rows, 0).transpose();
        stacked
            .reshape([batch * count, hidden_dim])
            .matmul(digit_weight)
    }

    /// Prefill a right-padded batch.
    ///
    /// `lengths` is the real token count of each row. Every attention block
    /// masks padding and applies the causal or sliding window, so a pad key is
    /// invisible to a real query. Logits are the next-token distribution at the
    /// last real position, shape `[batch, vocab]`. Each returned cache belongs
    /// to one row and already drops padding and keys outside that layer's
    /// sliding window.
    pub fn prefill(
        &self,
        token_ids: Tensor<2, Int>,
        lengths: &[usize],
    ) -> (Tensor<2>, Vec<ModelCache>) {
        let [batch, seq] = token_ids.dims();
        assert_eq!(lengths.len(), batch, "one length per row");
        assert!(
            lengths.iter().all(|len| *len >= 1 && *len <= seq),
            "prefill lengths must sit inside the padded sequence"
        );
        let (hidden, caches) = self.token_hidden_collect(token_ids, lengths);
        let positions: Vec<usize> = lengths.iter().map(|len| len - 1).collect();
        let logits = self.project_row(hidden, &positions);
        let per_layer: Vec<Vec<LayerKv>> = caches
            .iter()
            .zip(&self.layers)
            .map(|(cache, layer)| split_cache(cache, lengths, layer.attn.sliding_window))
            .collect();
        let rows = (0..batch)
            .map(|row| ModelCache {
                layers: per_layer.iter().map(|layer| layer[row].clone()).collect(),
            })
            .collect();
        (logits, rows)
    }

    /// Fast batched decode: steps one token per row across active batch rows using
    /// unified contiguous KV caches, precomputed device RoPE tables, and zero-overhead
    /// attention masks.
    pub fn decode_batched(
        &self,
        token_ids: Tensor<2, Int>,
        cache: &mut BatchedModelCache,
    ) -> Tensor<2> {
        let [batch, seq] = token_ids.dims();
        assert_eq!(seq, 1, "decode steps one token at a time");
        assert_eq!(cache.layers.len(), self.layers.len());
        let device = token_ids.device();

        let first_layer = &cache.layers[0];
        let key_len = first_layer.key.dims()[2];
        let query_pos: Vec<usize> = (0..batch)
            .map(|r| first_layer.pos0[r] + (key_len - first_layer.pad_prefix[r]))
            .collect();

        let mut rope_tables: Vec<(u32, usize, Tensor<4>, Tensor<4>)> = Vec::with_capacity(2);
        for layer in &self.layers {
            let spec = layer.attn.rope_spec();
            let rope_dim = spec
                .rotary_dim(layer.attn.head_dim)
                .min(layer.attn.head_dim);
            let theta_bits = spec.theta.to_bits();
            if !rope_tables
                .iter()
                .any(|t| t.0 == theta_bits && t.1 == rope_dim)
            {
                let (cos, sin) = crate::rope::rope_tensors_for_positions(
                    spec.theta, rope_dim, &query_pos, &device,
                );
                rope_tables.push((theta_bits, rope_dim, cos, sin));
            }
        }

        let mut mask_cache: Vec<(Option<usize>, usize, bool, Tensor<4>)> = Vec::with_capacity(4);
        for (index, layer) in self.layers.iter().enumerate() {
            let window = layer.attn.window();
            let layer_cache = &cache.layers[index];
            let old_key_len = layer_cache.key.dims()[2];
            let layer_key_len =
                if layer.attn.sliding_window > 0 && old_key_len + 1 > layer.attn.sliding_window {
                    layer.attn.sliding_window
                } else {
                    old_key_len + 1
                };
            let layer_pad_zero = layer_cache.pad_prefix.iter().all(|&p| p == 0);
            if !mask_cache
                .iter()
                .any(|(w, l, z, _)| *w == window && *l == layer_key_len && *z == layer_pad_zero)
            {
                let mask = if layer_pad_zero {
                    Tensor::<4>::zeros([1, 1, 1, layer_key_len], &device).cast(FloatDType::F32)
                } else {
                    let drop = if layer.attn.sliding_window > 0
                        && old_key_len + 1 > layer.attn.sliding_window
                    {
                        old_key_len + 1 - layer.attn.sliding_window
                    } else {
                        0
                    };
                    let mut pad_prefix = layer_cache.pad_prefix.clone();
                    let mut pos0 = layer_cache.pos0.clone();
                    if drop > 0 {
                        for r in 0..pad_prefix.len() {
                            if pad_prefix[r] >= drop {
                                pad_prefix[r] -= drop;
                            } else {
                                let real_drop = drop - pad_prefix[r];
                                pad_prefix[r] = 0;
                                pos0[r] += real_drop;
                            }
                        }
                    }
                    crate::attention::decode_mask(
                        &query_pos,
                        &pad_prefix,
                        &pos0,
                        layer_key_len,
                        window,
                        &device,
                    )
                };
                mask_cache.push((window, layer_key_len, layer_pad_zero, mask));
            }
        }

        let mut hidden = self.embed_tokens(token_ids);
        for (index, layer) in self.layers.iter().enumerate() {
            let spec = layer.attn.rope_spec();
            let rope_dim = spec
                .rotary_dim(layer.attn.head_dim)
                .min(layer.attn.head_dim);
            let theta_bits = spec.theta.to_bits();
            let (_, _, cos, sin) = rope_tables
                .iter()
                .find(|t| t.0 == theta_bits && t.1 == rope_dim)
                .expect("rope table");
            let window = layer.attn.window();
            let layer_cache = &cache.layers[index];
            let old_key_len = layer_cache.key.dims()[2];
            let layer_key_len =
                if layer.attn.sliding_window > 0 && old_key_len + 1 > layer.attn.sliding_window {
                    layer.attn.sliding_window
                } else {
                    old_key_len + 1
                };
            let layer_pad_zero = layer_cache.pad_prefix.iter().all(|&p| p == 0);
            let (_, _, _, mask) = mask_cache
                .iter()
                .find(|(w, l, z, _)| *w == window && *l == layer_key_len && *z == layer_pad_zero)
                .expect("mask cache");

            hidden = layer.forward_decode_batched(hidden, &mut cache.layers[index], cos, sin, mask);
        }

        let hidden = self.finish(hidden);
        let [b, _, hidden_dim] = hidden.dims();
        let weight = self.embedding_weight().transpose();
        hidden.reshape([b, hidden_dim]).matmul(weight)
    }

    /// Score one new token per row and append it to that row's cache.
    ///
    /// `token_ids` has shape `[batch, 1]`. The result is `[batch, vocab]`.
    pub fn decode(&self, token_ids: Tensor<2, Int>, caches: &mut [ModelCache]) -> Tensor<2> {
        let [batch, seq] = token_ids.dims();
        assert_eq!(seq, 1, "decode consumes one token per row");
        assert_eq!(caches.len(), batch, "one cache per row");
        assert!(
            caches
                .iter()
                .all(|cache| cache.layers.len() == self.layers.len()),
            "cache does not match the model"
        );
        let mut batched = BatchedModelCache::from_model_caches(caches);
        let logits = self.decode_batched(token_ids, &mut batched);
        let unpacked = batched.to_model_caches();
        for (i, cache) in caches.iter_mut().enumerate() {
            *cache = unpacked[i].clone();
        }
        logits
    }

    fn project_positions(&self, hidden: Tensor<3>, positions: &[usize]) -> Tensor<2> {
        let rows = positions
            .iter()
            .map(|position| hidden.clone().narrow(1, *position, 1))
            .collect::<Vec<_>>();
        let stacked = Tensor::cat(rows, 1);
        let [batch, count, hidden_dim] = stacked.dims();
        let weight = self.embedding_weight().transpose();
        stacked.reshape([batch * count, hidden_dim]).matmul(weight)
    }

    /// Logits at one index per batch row. Result shape is `[batch, vocab]`.
    fn project_row(&self, hidden: Tensor<3>, positions: &[usize]) -> Tensor<2> {
        let [batch, seq, hidden_dim] = hidden.dims();
        let weight = self.embedding_weight().transpose();
        if seq == 1 && positions.iter().all(|&p| p == 0) {
            return hidden.reshape([batch, hidden_dim]).matmul(weight);
        }
        let rows = positions
            .iter()
            .enumerate()
            .map(|(row, position)| hidden.clone().narrow(0, row, 1).narrow(1, *position, 1))
            .collect::<Vec<_>>();
        let stacked = Tensor::cat(rows, 0);
        stacked.reshape([batch, hidden_dim]).matmul(weight)
    }

    /// Log probability of every token given the tokens before it.
    ///
    /// Index 0 is zero because the first token has no preceding context.
    /// Callers mask it out of the policy loss.
    ///
    /// The vocabulary has 131,072 entries. `log_softmax` in bfloat16 cannot
    /// sum that many exponentials, and rounding the gathered log probability
    /// back to bfloat16 makes `exp(δ) - δ - 1` zero until the policy has
    /// already moved by about a tenth of a nat. The returned tensor stays f32.
    /// The sum is taken in vocabulary chunks so the `[sequence, vocab]` tensor
    /// is never stored. The value and the hidden-state gradient match a full
    /// f32 log-softmax up to summation order.
    pub fn aligned_logprobs(&self, token_ids: Tensor<2, Int>) -> Tensor<2> {
        let seq = token_ids.dims()[1];
        assert!(seq >= 2, "logprobs need at least two tokens");
        let device = token_ids.device();
        let batch = token_ids.dims()[0];
        let hidden = self.token_hidden(token_ids.clone());
        let hidden = hidden.narrow(1, 0, seq - 1);
        let targets = token_ids.narrow(1, 1, seq - 1);
        let gathered =
            chunked_token_logprobs(hidden, self.embedding_weight(), targets, VOCAB_CHUNK);
        let pad = Tensor::<2>::zeros([batch, 1], &device).cast(FloatDType::F32);
        Tensor::cat(vec![pad, gathered], 1)
    }

    /// Forward-only aligned log probabilities without surrogate backward gradients.
    ///
    /// Used for reference policy evaluation where weights are frozen and no gradients
    /// are needed, cutting computation in half by skipping the backward surrogate pass.
    pub fn aligned_logprobs_forward(&self, token_ids: Tensor<2, Int>) -> Tensor<2> {
        let seq = token_ids.dims()[1];
        assert!(seq >= 2, "logprobs need at least two tokens");
        let device = token_ids.device();
        let batch = token_ids.dims()[0];
        let hidden = self.token_hidden(token_ids.clone());
        let hidden = hidden.narrow(1, 0, seq - 1);
        let targets = token_ids.narrow(1, 1, seq - 1);
        let gathered =
            chunked_token_logprobs_forward(hidden, self.embedding_weight(), targets, VOCAB_CHUNK);
        let pad = Tensor::<2>::zeros([batch, 1], &device).cast(FloatDType::F32);
        Tensor::cat(vec![pad, gathered], 1)
    }
}

/// Forward-only gathered token log-probabilities without computing surrogate gradients.
fn chunked_token_logprobs_forward(
    hidden: Tensor<3>,
    weight: Tensor<2>,
    targets: Tensor<2, Int>,
    chunk: usize,
) -> Tensor<2> {
    let [batch, seq, hidden_dim] = hidden.dims();
    let vocab = weight.dims()[0];
    let device = hidden.device();
    let hidden_det = hidden.detach().without_autodiff();
    let weight_det = weight.detach().without_autodiff();
    let flat = hidden_det.reshape([batch * seq, hidden_dim]);
    let mut running_max =
        Tensor::<2>::full([batch, seq], -1.0e30_f32, &device).cast(FloatDType::F32);
    let mut running_sum = Tensor::<2>::zeros([batch, seq], &device).cast(FloatDType::F32);
    let mut token_logit = Tensor::<2>::zeros([batch, seq], &device).cast(FloatDType::F32);

    let mut start = 0;
    while start < vocab {
        let len = chunk.max(1).min(vocab - start);
        let columns = weight_det.clone().narrow(0, start, len);
        let logits = flat
            .clone()
            .matmul(columns.transpose())
            .cast(FloatDType::F32)
            .reshape([batch, seq, len]);
        let chunk_max = logits.clone().max_dim(2).squeeze_dim::<2>(2);
        let new_max = running_max.clone().max_pair(chunk_max);
        let sum_exp = (logits.clone() - new_max.clone().unsqueeze_dim::<3>(2))
            .exp()
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let scale = (running_max - new_max.clone()).exp();
        running_sum = running_sum * scale + sum_exp;
        running_max = new_max;

        let local = targets.clone().sub_scalar(start as i32);
        let in_chunk = local
            .clone()
            .greater_equal_elem(0)
            .bool_and(local.clone().lower_elem(len as i32));
        let index = local.clamp(0, (len as i32) - 1).unsqueeze_dim::<3>(2);
        let picked = logits.gather(2, index).squeeze_dim::<2>(2);
        token_logit = token_logit + picked * in_chunk.float().cast(FloatDType::F32);
        start += len;
    }

    let lse = running_max + running_sum.log();
    token_logit - lse
}

/// Gathered f32 log-probability of `targets`, with a hidden-state gradient
/// equal to the cross-entropy Jacobian.
///
/// Chunk logits are detached. The returned tensor is tied back to `hidden` by
/// a surrogate whose value is the log-probability and whose gradient is
/// `embed[target] - softmax @ embed`.
fn chunked_token_logprobs(
    hidden: Tensor<3>,
    weight: Tensor<2>,
    targets: Tensor<2, Int>,
    chunk: usize,
) -> Tensor<2> {
    let [batch, seq, hidden_dim] = hidden.dims();
    let vocab = weight.dims()[0];
    let device = hidden.device();
    let hidden_det = hidden.clone().detach().without_autodiff();
    let weight_det = weight.detach().without_autodiff();
    let flat = hidden_det.reshape([batch * seq, hidden_dim]);
    let mut running_max =
        Tensor::<2>::full([batch, seq], -1.0e30_f32, &device).cast(FloatDType::F32);
    let mut running_sum = Tensor::<2>::zeros([batch, seq], &device).cast(FloatDType::F32);
    let mut token_logit = Tensor::<2>::zeros([batch, seq], &device).cast(FloatDType::F32);

    let mut start = 0;
    while start < vocab {
        let len = chunk.max(1).min(vocab - start);
        let columns = weight_det.clone().narrow(0, start, len);
        let logits = flat
            .clone()
            .matmul(columns.transpose())
            .cast(FloatDType::F32)
            .reshape([batch, seq, len]);
        let chunk_max = logits.clone().max_dim(2).squeeze_dim::<2>(2);
        let new_max = running_max.clone().max_pair(chunk_max);
        let sum_exp = (logits.clone() - new_max.clone().unsqueeze_dim::<3>(2))
            .exp()
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let scale = (running_max - new_max.clone()).exp();
        running_sum = running_sum * scale + sum_exp;
        running_max = new_max;

        let local = targets.clone().sub_scalar(start as i32);
        let in_chunk = local
            .clone()
            .greater_equal_elem(0)
            .bool_and(local.clone().lower_elem(len as i32));
        let index = local.clamp(0, (len as i32) - 1).unsqueeze_dim::<3>(2);
        let picked = logits.gather(2, index).squeeze_dim::<2>(2);
        token_logit = token_logit + picked * in_chunk.float().cast(FloatDType::F32);
        start += len;
    }

    let lse = running_max + running_sum.log();
    let weight_f32 = weight_det.clone().cast(FloatDType::F32);
    let mut expected = Tensor::<3>::zeros([batch, seq, hidden_dim], &device).cast(FloatDType::F32);
    start = 0;
    while start < vocab {
        let len = chunk.max(1).min(vocab - start);
        let columns = weight_det.clone().narrow(0, start, len);
        let logits = flat
            .clone()
            .matmul(columns.transpose())
            .cast(FloatDType::F32)
            .reshape([batch, seq, len]);
        let probs = (logits - lse.clone().unsqueeze_dim::<3>(2)).exp();
        let contrib = probs
            .reshape([batch * seq, len])
            .matmul(weight_f32.clone().narrow(0, start, len))
            .reshape([batch, seq, hidden_dim]);
        expected = expected + contrib;
        start += len;
    }
    let rows = burn::tensor::module::embedding(weight_f32, targets);
    let direction = (rows - expected).detach();
    let hidden_f32 = hidden.cast(FloatDType::F32);
    let dot = (hidden_f32.clone() * direction.clone())
        .sum_dim(2)
        .squeeze_dim::<2>(2);
    let correction = (hidden_f32.detach() * direction)
        .sum_dim(2)
        .squeeze_dim::<2>(2);
    token_logit.detach() - lse.detach() + dot - correction
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SparkConfig;
    use burn::module::Module;
    use burn::tensor::TensorData;

    fn device() -> Device {
        Device::flex()
    }

    #[test]
    fn tiny_parameter_count_matches_the_module() {
        let config = SparkConfig::tiny();
        let model = SparkModel::new(&config, &device());
        assert_eq!(model.num_params() as u64, config.parameter_count());
        assert_eq!(model.layers[0].attn.sliding_window, config.sliding_window);
        assert_eq!(model.layers[1].attn.sliding_window, 0);
    }

    #[test]
    fn aligned_logprobs_are_negative_except_the_first_token() {
        let config = SparkConfig::tiny();
        let model = SparkModel::new(&config, &device());
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(vec![1, 2, 3, 4], [1, 4]), &device());
        let values = model.aligned_logprobs(tokens).into_data();
        let values = values.as_slice::<f32>().unwrap();
        assert_eq!(values[0], 0.0);
        assert!(values[1..].iter().all(|value| *value <= 0.0), "{values:?}");
    }

    #[test]
    fn aligned_logprobs_on_bf16_device() {
        if let Ok(device) = crate::device::bf16_device(false) {
            let config = SparkConfig::tiny();
            let model = SparkModel::new(&config, &device);
            let tokens =
                Tensor::<2, Int>::from_data(TensorData::new(vec![1, 2, 3, 4], [1, 4]), &device);
            let values = model.aligned_logprobs(tokens).into_data();
            let values = values.as_slice::<f32>().unwrap();
            assert_eq!(values[0], 0.0);
            assert!(values[1..].iter().all(|value| *value <= 0.0), "{values:?}");
        }
    }

    #[test]
    fn logits_have_the_vocabulary_dimension() {
        let config = SparkConfig::tiny();
        let model = SparkModel::new(&config, &device());
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(vec![1, 2, 3, 4], [1, 4]), &device());
        let logits = model.logits(tokens.clone());
        assert_eq!(logits.dims(), [1, 4, config.vocab_size]);
        let last = model.last_logits(tokens.clone());
        assert_eq!(last.dims(), [1, config.vocab_size]);
        let full = logits.into_data();
        let full = full.as_slice::<f32>().unwrap();
        let only = last.into_data();
        let only = only.as_slice::<f32>().unwrap();
        let vocab = config.vocab_size;
        let diff = full[3 * vocab..]
            .iter()
            .zip(only)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(diff < 1e-4, "last position logits differ by {diff}");
        let picked = model.position_logits(tokens, &[0, 3]);
        assert_eq!(picked.dims(), [2, config.vocab_size]);
        let picked = picked.into_data();
        let picked = picked.as_slice::<f32>().unwrap();
        let first = full[..vocab]
            .iter()
            .zip(&picked[..vocab])
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(first < 1e-4, "position 0 logits differ by {first}");
    }

    #[test]
    fn causal_mask_hides_future_tokens() {
        let config = SparkConfig::tiny();
        let model = SparkModel::new(&config, &device());
        let left =
            Tensor::<2, Int>::from_data(TensorData::new(vec![1, 2, 3, 4], [1, 4]), &device());
        let right =
            Tensor::<2, Int>::from_data(TensorData::new(vec![1, 2, 3, 9], [1, 4]), &device());
        let left = model.logits(left).into_data();
        let right = model.logits(right).into_data();
        let left = left.as_slice::<f32>().unwrap();
        let right = right.as_slice::<f32>().unwrap();
        let vocab = config.vocab_size;
        // Position 2 can attend through position 2, which is identical. Position 3 differs.
        let shared = 3 * vocab;
        let diff = left[..shared]
            .iter()
            .zip(right[..shared].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(diff < 1e-4, "future token changed past logits by {diff}");
    }

    #[test]
    fn strict_local_attention_does_not_read_distant_tokens() {
        let mut config = SparkConfig::tiny();
        config.global_every = 0;
        config.sliding_window = 1;
        let model = SparkModel::new(&config, &device());
        let left = Tensor::<2, Int>::from_data(TensorData::new(vec![1, 4, 7], [1, 3]), &device());
        let right = Tensor::<2, Int>::from_data(TensorData::new(vec![2, 5, 7], [1, 3]), &device());
        let left = model.logits(left).into_data();
        let right = model.logits(right).into_data();
        let left = left.as_slice::<f32>().unwrap();
        let right = right.as_slice::<f32>().unwrap();
        let vocab = config.vocab_size;
        let last = &left[2 * vocab..];
        let other = &right[2 * vocab..];
        let diff = last
            .iter()
            .zip(other)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            diff < 1e-4,
            "window-1 layer leaked a distant token, max abs diff {diff}"
        );
    }

    #[test]
    fn the_pad_row_is_zero() {
        let model = SparkModel::new(&SparkConfig::tiny(), &device());
        let row = model
            .embed
            .weight
            .val()
            .narrow(0, model.pad_token_id, 1)
            .into_data();
        let row = row.as_slice::<f32>().unwrap();
        assert!(row.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn cached_decode_matches_the_full_forward() {
        let model = SparkModel::new(&SparkConfig::tiny(), &device());
        let mut tokens = vec![1_i32, 4, 5, 6];
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(tokens.clone(), [1, tokens.len()]),
            &device(),
        );
        let (mut logits, mut caches) = model.prefill(input, &[tokens.len()]);
        let full = Tensor::<2, Int>::from_data(
            TensorData::new(tokens.clone(), [1, tokens.len()]),
            &device(),
        );
        assert_close(&host2(logits), &host2(model.last_logits(full)));
        for next in [7_i32, 8, 3, 9, 1, 5] {
            let step = Tensor::<2, Int>::from_data(TensorData::new(vec![next], [1, 1]), &device());
            logits = model.decode(step, &mut caches);
            tokens.push(next);
            let full = Tensor::<2, Int>::from_data(
                TensorData::new(tokens.clone(), [1, tokens.len()]),
                &device(),
            );
            assert_close(&host2(logits), &host2(model.last_logits(full)));
        }
    }

    #[test]
    fn batched_cache_matches_each_row() {
        let model = SparkModel::new(&SparkConfig::tiny(), &device());
        let pad = model.pad_token_id as i32;
        let left = vec![1_i32, 4, 5];
        let right = vec![3_i32, 6, 8, 9];
        let mut packed = left.clone();
        packed.push(pad);
        packed.extend(&right);
        let input = Tensor::<2, Int>::from_data(TensorData::new(packed, [2, 4]), &device());
        let (logits, mut caches) = model.prefill(input, &[left.len(), right.len()]);
        let flat = host2(logits);
        let vocab = flat.len() / 2;
        let left_full =
            Tensor::<2, Int>::from_data(TensorData::new(left.clone(), [1, left.len()]), &device());
        let right_full = Tensor::<2, Int>::from_data(
            TensorData::new(right.clone(), [1, right.len()]),
            &device(),
        );
        assert_close(&flat[..vocab], &host2(model.last_logits(left_full)));
        assert_close(&flat[vocab..], &host2(model.last_logits(right_full)));

        let step = Tensor::<2, Int>::from_data(TensorData::new(vec![7_i32, 1], [2, 1]), &device());
        let logits = model.decode(step, &mut caches);
        let flat = host2(logits);
        let mut left = left;
        let mut right = right;
        left.push(7);
        right.push(1);
        let left_full =
            Tensor::<2, Int>::from_data(TensorData::new(left.clone(), [1, left.len()]), &device());
        let right_full = Tensor::<2, Int>::from_data(
            TensorData::new(right.clone(), [1, right.len()]),
            &device(),
        );
        assert_close(&flat[..vocab], &host2(model.last_logits(left_full)));
        assert_close(&flat[vocab..], &host2(model.last_logits(right_full)));
    }

    #[test]
    fn a_hostile_right_pad_cannot_change_real_logits() {
        let model = SparkModel::new(&SparkConfig::tiny(), &device());
        let left = vec![1_i32, 4, 5];
        let right = vec![3_i32, 6, 8, 9];
        // The pad columns are real vocabulary rows, not the zero pad embedding.
        let mut packed = left.clone();
        packed.extend([11, 12]);
        packed.extend(&right);
        packed.push(13);
        let input = Tensor::<2, Int>::from_data(TensorData::new(packed, [2, 5]), &device());
        let (logits, _) = model.prefill(input, &[left.len(), right.len()]);
        let flat = host2(logits);
        let vocab = flat.len() / 2;
        let left_full =
            Tensor::<2, Int>::from_data(TensorData::new(left.clone(), [1, left.len()]), &device());
        let right_full = Tensor::<2, Int>::from_data(
            TensorData::new(right.clone(), [1, right.len()]),
            &device(),
        );
        assert_close(&flat[..vocab], &host2(model.last_logits(left_full)));
        assert_close(&flat[vocab..], &host2(model.last_logits(right_full)));
    }

    #[test]
    fn batched_decode_matches_past_the_sliding_window() {
        let model = SparkModel::new(&SparkConfig::tiny(), &device());
        let pad = model.pad_token_id as i32;
        let mut left = vec![1_i32, 4, 5];
        let mut right = vec![3_i32, 6, 8, 9];
        let mut packed = left.clone();
        packed.push(pad);
        packed.extend(&right);
        let input = Tensor::<2, Int>::from_data(TensorData::new(packed, [2, 4]), &device());
        let (mut logits, mut caches) = model.prefill(input, &[left.len(), right.len()]);
        let steps = [7_i32, 8, 3, 9, 1, 5];
        for next in steps {
            let flat = host2(logits);
            let vocab = flat.len() / 2;
            let left_full = Tensor::<2, Int>::from_data(
                TensorData::new(left.clone(), [1, left.len()]),
                &device(),
            );
            let right_full = Tensor::<2, Int>::from_data(
                TensorData::new(right.clone(), [1, right.len()]),
                &device(),
            );
            assert_close(&flat[..vocab], &host2(model.last_logits(left_full)));
            assert_close(&flat[vocab..], &host2(model.last_logits(right_full)));
            let step =
                Tensor::<2, Int>::from_data(TensorData::new(vec![next, next], [2, 1]), &device());
            logits = model.decode(step, &mut caches);
            left.push(next);
            right.push(next);
        }
        let flat = host2(logits);
        let vocab = flat.len() / 2;
        let left_full =
            Tensor::<2, Int>::from_data(TensorData::new(left.clone(), [1, left.len()]), &device());
        let right_full = Tensor::<2, Int>::from_data(
            TensorData::new(right.clone(), [1, right.len()]),
            &device(),
        );
        assert_close(&flat[..vocab], &host2(model.last_logits(left_full)));
        assert_close(&flat[vocab..], &host2(model.last_logits(right_full)));
    }

    #[test]
    fn left_padded_positions_match_the_unpadded_story() {
        let model = SparkModel::new(&SparkConfig::tiny(), &device());
        let pad = model.pad_token_id as i32;
        let short = vec![4_i32, 5, 3, 1];
        let long = vec![6_i32, 7, 8, 3, 1];
        let short_ids = Tensor::<2, Int>::from_data(
            TensorData::new(short.clone(), [1, short.len()]),
            &device(),
        );
        let long_ids =
            Tensor::<2, Int>::from_data(TensorData::new(long.clone(), [1, long.len()]), &device());
        let short_logits = host2(model.position_logits(short_ids, &[2, 3]));
        let long_logits = host2(model.position_logits(long_ids, &[3, 4]));

        let mut packed = vec![pad, 4, 5, 3, 1];
        packed.extend(&long);
        let batch = Tensor::<2, Int>::from_data(TensorData::new(packed, [2, 5]), &device());
        let scored = host2(model.position_logits_padded(batch, &[3, 4], &[1, 0]));
        let vocab = model.embed.weight.dims()[0];
        assert_eq!(scored.len(), vocab * 4);
        assert_close(&scored[..vocab], &short_logits[..vocab]);
        assert_close(&scored[vocab..2 * vocab], &short_logits[vocab..]);
        assert_close(&scored[2 * vocab..3 * vocab], &long_logits[..vocab]);
        assert_close(&scored[3 * vocab..], &long_logits[vocab..]);

        // Longer than the sliding window, with the suffix positions a rubric
        // batch reads. The pad query must stay finite or these rows go NaN.
        let short = vec![7_i32, 8, 3, 4, 5];
        let long = vec![9_i32, 10, 11, 3, 4, 5];
        let short_ids = Tensor::<2, Int>::from_data(
            TensorData::new(short.clone(), [1, short.len()]),
            &device(),
        );
        let long_ids =
            Tensor::<2, Int>::from_data(TensorData::new(long.clone(), [1, long.len()]), &device());
        let short_logits = host2(model.position_logits(short_ids, &[2, 4]));
        let long_logits = host2(model.position_logits(long_ids, &[3, 5]));
        let mut packed = vec![pad, 7, 8, 3, 4, 5];
        packed.extend(&long);
        let batch = Tensor::<2, Int>::from_data(TensorData::new(packed, [2, 6]), &device());
        let scored = host2(model.position_logits_padded(batch, &[3, 5], &[1, 0]));
        assert_close(&scored[..vocab], &short_logits[..vocab]);
        assert_close(&scored[vocab..2 * vocab], &short_logits[vocab..]);
        assert_close(&scored[2 * vocab..3 * vocab], &long_logits[..vocab]);
        assert_close(&scored[3 * vocab..], &long_logits[vocab..]);
    }

    #[test]
    fn chunked_logprobs_match_full_log_softmax_and_its_gradient() {
        use burn::optim::GradientsParams;
        use burn::tensor::activation::log_softmax;

        use crate::{LoraSpec, attach_lora};

        let plain = Device::flex();
        let hidden = Tensor::<3>::from_data(
            TensorData::new(
                (0..2 * 3 * 8)
                    .map(|value| ((value % 11) as f32 - 5.0) / 7.0)
                    .collect(),
                [2, 3, 8],
            ),
            &plain,
        );
        let weight = Tensor::<2>::from_data(
            TensorData::new(
                (0..40 * 8)
                    .map(|value| ((value % 9) as f32 - 4.0) / 5.0)
                    .collect(),
                [40, 8],
            ),
            &plain,
        );
        let targets = Tensor::<2, Int>::from_data(
            TensorData::new(vec![1_i32, 7, 39, 0, 8, 15], [2, 3]),
            &plain,
        );
        let wide = host2(chunked_token_logprobs(
            hidden.clone(),
            weight.clone(),
            targets.clone(),
            8,
        ));
        let narrow = host2(chunked_token_logprobs(
            hidden.clone(),
            weight.clone(),
            targets.clone(),
            64,
        ));
        let logits = hidden
            .clone()
            .reshape([6, 8])
            .matmul(weight.transpose())
            .reshape([2, 3, 40]);
        let index = targets.unsqueeze_dim::<3>(2);
        let full = host2(
            log_softmax(logits.cast(FloatDType::F32), 2)
                .gather(2, index)
                .squeeze_dim::<2>(2),
        );
        assert_close(&wide, &narrow);
        assert_close(&wide, &full);

        let device = Device::flex().autodiff();
        let model = attach_lora(
            SparkModel::new(&SparkConfig::tiny(), &device),
            LoraSpec::attention_default(),
        )
        .unwrap();
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(vec![1_i32, 4, 5, 6], [1, 4]), &device);
        let loss = model.aligned_logprobs(tokens.clone()).sum();
        let mut grads = GradientsParams::from_grads(loss.backward(), &model);
        let id = model.layers[0].attn.qkv.weight.adapter().unwrap().b.id;
        let learned = host2(grads.remove::<2>(id).expect("chunked logprob gradient"));

        let seq = 4;
        let logits = model.logits(tokens.clone());
        let next_logits = logits.narrow(1, 0, seq - 1);
        let next_tokens = tokens.narrow(1, 1, seq - 1).unsqueeze_dim::<3>(2);
        let gathered = log_softmax(next_logits.cast(FloatDType::F32), 2)
            .gather(2, next_tokens)
            .squeeze_dim::<2>(2);
        let pad = Tensor::<2>::zeros([1, 1], &device).cast(FloatDType::F32);
        let reference = Tensor::cat(vec![pad, gathered], 1).sum();
        let mut grads = GradientsParams::from_grads(reference.backward(), &model);
        let baseline = host2(grads.remove::<2>(id).expect("full log_softmax gradient"));
        assert_close(&learned, &baseline);
    }

    #[test]
    fn digit_logits_match_position_logits_at_digit_indices() {
        let model = SparkModel::new(&SparkConfig::tiny(), &device());
        let tokens = Tensor::<2, Int>::from_data(
            TensorData::new(vec![1_i32, 2, 3, 4, 5, 6], [1, 6]),
            &device(),
        );
        let positions = [2, 4];
        let digits = [0_usize, 1, 2, 3, 4, 5];
        let full = host2(model.position_logits_padded(tokens.clone(), &positions, &[]));
        let vocab = model.embed.weight.val().dims()[0];
        let digit_logits = host2(model.digit_logits_padded(tokens, &positions, &[], &digits));
        assert_eq!(digit_logits.len(), positions.len() * 6);
        for (pos_idx, _) in positions.iter().enumerate() {
            for (digit_slot, &digit_id) in digits.iter().enumerate() {
                let expected = full[pos_idx * vocab + digit_id];
                let actual = digit_logits[pos_idx * 6 + digit_slot];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "at pos {pos_idx} digit {digit_id}: actual {actual} vs expected {expected}"
                );
            }
        }
    }

    #[test]
    fn digit_logits_padded_per_row_matches_single_rows() {
        let model = SparkModel::new(&SparkConfig::tiny(), &device());
        let tokens0 = vec![1_i32, 2, 3, 4, 5, 6];
        let tokens1 = vec![7_i32, 8, 9, 10, 11, 12, 13];
        let pad_prefix = vec![1, 0];
        let mut packed = vec![0_i32];
        packed.extend(&tokens0);
        packed.extend(&tokens1);
        let batch = Tensor::<2, Int>::from_data(TensorData::new(packed, [2, 7]), &device());
        let row_positions = vec![vec![2, 4], vec![1, 5]];
        let digits = [0_usize, 1, 2, 3, 4, 5];
        let batched =
            host2(model.digit_logits_padded_per_row(batch, &row_positions, &pad_prefix, &digits));
        assert_eq!(batched.len(), 4 * 6);

        let mut row0_packed = vec![0_i32];
        row0_packed.extend(&tokens0);
        let row0_tokens =
            Tensor::<2, Int>::from_data(TensorData::new(row0_packed, [1, 7]), &device());
        let row0_logits = host2(model.digit_logits_padded(row0_tokens, &[2, 4], &[1], &digits));

        let row1_tokens = Tensor::<2, Int>::from_data(TensorData::new(tokens1, [1, 7]), &device());
        let row1_logits = host2(model.digit_logits_padded(row1_tokens, &[1, 5], &[0], &digits));

        for idx in 0..12 {
            assert!(
                (batched[idx] - row0_logits[idx]).abs() < 1e-4,
                "row 0 mismatch at {idx}: {} vs {}",
                batched[idx],
                row0_logits[idx]
            );
        }
        for idx in 0..12 {
            assert!(
                (batched[12 + idx] - row1_logits[idx]).abs() < 1e-4,
                "row 1 mismatch at {idx}: {} vs {}",
                batched[12 + idx],
                row1_logits[idx]
            );
        }
    }

    #[test]
    fn aligned_logprobs_forward_matches_aligned_logprobs() {
        let model = SparkModel::new(&SparkConfig::tiny(), &device());
        let tokens = Tensor::<2, Int>::from_data(
            TensorData::new(vec![1_i32, 4, 5, 6, 2, 3], [1, 6]),
            &device(),
        );
        let backward_surrogate = host2(model.aligned_logprobs(tokens.clone()));
        let forward_only = host2(model.aligned_logprobs_forward(tokens));
        assert_close(&forward_only, &backward_surrogate);
    }

    fn host2<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
        tensor
            .cast(FloatDType::F32)
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let mut diff = 0.0_f32;
        for (left, right) in actual.iter().zip(expected) {
            assert!(left.is_finite() && right.is_finite(), "{left} vs {right}");
            diff = diff.max((left - right).abs());
        }
        assert!(diff < 1e-4, "max abs diff {diff}");
    }
}
