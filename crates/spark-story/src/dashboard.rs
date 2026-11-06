//! Burn's training dashboard for the GRPO adapter loop.
//!
//! A terminal shows the TUI renderer from `burn::train`. Any other stdout
//! gets the CLI renderer. Both are the renderers `default_renderer` selects.
//!
//! Metrics tracked across each backward slice:
//! - **Loss**: GRPO clipped policy loss (lower is better)
//! - **Reward**: Contrast reward vs negative format baseline (higher is better)
//! - **Rubric**: Effective harmonic rubric score after progression penalty (higher is better)
//! - **RawRubric**: Raw harmonic rubric score before progression penalty (higher is better)
//! - **Progression**: Thematic movement across opening -> middle -> ending (higher is better)
//! - **ProgPenalty**: Gap between raw rubric and effective score due to circular loops (lower is better)
//! - **StoryLen**: Completion token length (detects length collapse or runaway)
//! - **Diversity**: Lexical diversity via type-token ratio (detects repetitive loops)
//! - **Advantage**: Mean absolute advantage magnitude (reinforcement signal strength)
//! - **Centered**: In-group score centered around format mean
//! - **KL**: Token KL divergence from frozen reference model
//! - **LogProb**: Mean completion token log probability under current policy
//! - **LearningRate**: Optimizer learning rate

use std::io::IsTerminal;
use std::sync::Arc;

use burn::data::dataloader::Progress;
use burn::tensor::TensorReadError;
use burn::train::Interrupter;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{
    LearningRateMetric, Metric, MetricAttributes, MetricDefinition, MetricEntry, MetricId,
    MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry, SerializedEntry,
};
use burn::train::renderer::tui::TuiMetricsRendererWrapper;
use burn::train::renderer::{CliMetricsRenderer, MetricState, MetricsRenderer};

/// Snapshot of all metrics for one training backward slice or micro-batch.
#[derive(Debug, Clone, Copy)]
pub struct StepMetrics {
    /// Policy loss for this chunk (lower is better)
    pub loss: f64,
    /// Contrast reward: own rubric minus negative format baseline (higher is better)
    pub reward: f64,
    /// Effective harmonic mean rubric score after progression penalty (higher is better)
    pub rubric: f64,
    /// Raw harmonic mean rubric score before penalty (higher is better)
    pub raw_rubric: f64,
    /// Thematic progression score across opening -> middle -> ending (higher is better)
    pub progression: f64,
    /// Progression penalty: raw_rubric - rubric (lower is better, 0 means no circularity)
    pub prog_penalty: f64,
    /// Mean story completion length in tokens (monitors against mode collapse)
    pub story_len: f64,
    /// Lexical diversity: type-token ratio (higher is better, detects repetitive degeneration)
    pub diversity: f64,
    /// Mean absolute advantage magnitude (higher means stronger reinforcement signal)
    pub advantage: f64,
    /// Within-format score minus format mean (before 1/std)
    pub centered: f64,
    /// Token KL divergence from frozen reference model (lower is better)
    pub kl: f64,
    /// Mean completion token log probability under current policy
    pub logprob: f64,
    /// Current optimizer learning rate
    pub learning_rate: f64,
    /// Number of stories in this slice
    pub stories: usize,
}

#[derive(Debug, Default, Clone)]
pub struct MetricsHistory {
    pub total_steps: usize,
    pub total_stories: usize,
    pub loss_sum: f64,
    pub reward_sum: f64,
    pub rubric_sum: f64,
    pub raw_rubric_sum: f64,
    pub progression_sum: f64,
    pub prog_penalty_sum: f64,
    pub story_len_sum: f64,
    pub diversity_sum: f64,
    pub advantage_sum: f64,
    pub centered_sum: f64,
    pub kl_sum: f64,
    pub logprob_sum: f64,
    pub last_metrics: Option<StepMetrics>,
}

impl MetricsHistory {
    pub fn record(&mut self, metrics: &StepMetrics) {
        let weight = metrics.stories.max(1) as f64;
        self.total_steps += 1;
        self.total_stories += metrics.stories;
        self.loss_sum += metrics.loss * weight;
        self.reward_sum += metrics.reward * weight;
        self.rubric_sum += metrics.rubric * weight;
        self.raw_rubric_sum += metrics.raw_rubric * weight;
        self.progression_sum += metrics.progression * weight;
        self.prog_penalty_sum += metrics.prog_penalty * weight;
        self.story_len_sum += metrics.story_len * weight;
        self.diversity_sum += metrics.diversity * weight;
        self.advantage_sum += metrics.advantage * weight;
        self.centered_sum += metrics.centered * weight;
        self.kl_sum += metrics.kl * weight;
        self.logprob_sum += metrics.logprob * weight;
        self.last_metrics = Some(*metrics);
    }
}

pub struct TrainingDashboard {
    renderer: Box<dyn MetricsRenderer>,
    interrupter: Interrupter,
    steps: usize,
    loss: MetricSlot,
    reward: MetricSlot,
    rubric: MetricSlot,
    raw_rubric: MetricSlot,
    progression: MetricSlot,
    prog_penalty: MetricSlot,
    story_len: MetricSlot,
    diversity: MetricSlot,
    advantage: MetricSlot,
    centered: MetricSlot,
    kl: MetricSlot,
    logprob: MetricSlot,
    learning_rate: LearningRateSlot,
    history: MetricsHistory,
}

impl TrainingDashboard {
    pub fn open(steps: usize) -> Self {
        let interrupter = Interrupter::new();
        let mut renderer: Box<dyn MetricsRenderer> = if std::io::stdout().is_terminal() {
            Box::new(TuiMetricsRendererWrapper::new(interrupter.clone(), None))
        } else {
            Box::new(CliMetricsRenderer::new())
        };

        let loss = MetricSlot::new(&mut *renderer, "Loss", false);
        let reward = MetricSlot::new(&mut *renderer, "Reward", true);
        let rubric = MetricSlot::new(&mut *renderer, "Rubric", true);
        let raw_rubric = MetricSlot::new(&mut *renderer, "RawRubric", true);
        let progression = MetricSlot::new(&mut *renderer, "Progression", true);
        let prog_penalty = MetricSlot::new(&mut *renderer, "ProgPenalty", false);
        let story_len = MetricSlot::new(&mut *renderer, "StoryLen", true);
        let diversity = MetricSlot::new(&mut *renderer, "Diversity", true);
        let advantage = MetricSlot::new(&mut *renderer, "Advantage", true);
        let centered = MetricSlot::new(&mut *renderer, "Centered", false);
        let kl = MetricSlot::new(&mut *renderer, "KL", false);
        let logprob = MetricSlot::new(&mut *renderer, "LogProb", true);
        let learning_rate = LearningRateSlot::new(&mut *renderer);

        renderer.start(1, 1, Some(steps), Some("GRPO LoRA"));
        renderer.start_split("train", steps);

        Self {
            renderer,
            interrupter,
            steps,
            loss,
            reward,
            rubric,
            raw_rubric,
            progression,
            prog_penalty,
            story_len,
            diversity,
            advantage,
            centered,
            kl,
            logprob,
            learning_rate,
            history: MetricsHistory::default(),
        }
    }

    pub fn stopped(&self) -> bool {
        self.interrupter.should_stop()
    }

    #[allow(dead_code)]
    pub fn history(&self) -> &MetricsHistory {
        &self.history
    }

    pub fn record(&mut self, step: usize, metrics: &StepMetrics) -> Result<(), String> {
        if self.stopped() {
            return Err("training stopped from the dashboard".into());
        }
        self.renderer.update_split(step);
        let metadata = MetricMetadata {
            progress: Progress::new(step, self.steps, Some("batches".to_string())),
            iteration: Some(step),
            lr: Some(metrics.learning_rate.into()),
        };
        let stories = metrics.stories.max(1);
        self.loss
            .update(&mut *self.renderer, metrics.loss, stories, &metadata)?;
        self.reward
            .update(&mut *self.renderer, metrics.reward, stories, &metadata)?;
        self.rubric
            .update(&mut *self.renderer, metrics.rubric, stories, &metadata)?;
        self.raw_rubric
            .update(&mut *self.renderer, metrics.raw_rubric, stories, &metadata)?;
        self.progression
            .update(&mut *self.renderer, metrics.progression, stories, &metadata)?;
        self.prog_penalty.update(
            &mut *self.renderer,
            metrics.prog_penalty,
            stories,
            &metadata,
        )?;
        self.story_len
            .update(&mut *self.renderer, metrics.story_len, stories, &metadata)?;
        self.diversity
            .update(&mut *self.renderer, metrics.diversity, stories, &metadata)?;
        self.advantage
            .update(&mut *self.renderer, metrics.advantage, stories, &metadata)?;
        self.centered
            .update(&mut *self.renderer, metrics.centered, stories, &metadata)?;
        self.kl
            .update(&mut *self.renderer, metrics.kl, stories, &metadata)?;
        self.logprob
            .update(&mut *self.renderer, metrics.logprob, stories, &metadata)?;
        self.learning_rate.update(&mut *self.renderer, &metadata)?;

        self.history.record(metrics);
        Ok(())
    }

    pub fn summary_report(&self) -> String {
        let count = self.history.total_stories.max(1) as f64;
        let last = self.history.last_metrics.unwrap_or(StepMetrics {
            loss: 0.0,
            reward: 0.0,
            rubric: 0.0,
            raw_rubric: 0.0,
            progression: 0.0,
            prog_penalty: 0.0,
            story_len: 0.0,
            diversity: 0.0,
            advantage: 0.0,
            centered: 0.0,
            kl: 0.0,
            logprob: 0.0,
            learning_rate: 0.0,
            stories: 0,
        });

        format!(
            "================================================================================\n\
             \x20                           Training Metrics Summary\n\
             ================================================================================\n\
             Optimization & Policy Health:\n\
               • Policy Loss:       {avg_loss:.4}  (last: {last_loss:.4})\n\
               • KL Divergence:     {avg_kl:.6}  (last: {last_kl:.6})\n\
               • Log Probability:   {avg_logprob:.4}  (last: {last_logprob:.4})\n\
               • Mean |Advantage|:  {avg_adv:.4}  (last: {last_adv:.4})\n\
               • Learning Rate:     {last_lr:.2e}\n\n\
             Reinforcement & Contrast Signals:\n\
               • Contrast Reward:   {avg_reward:+.4}  (last: {last_reward:+.4})\n\
               • Centered Margin:   {avg_centered:+.4}  (last: {last_centered:+.4})\n\n\
             Rubric & Story Quality:\n\
               • Effective Rubric:  {avg_rubric:.4}  (last: {last_rubric:.4})\n\
               • Raw Rubric:        {avg_raw:.4}  (last: {last_raw:.4})\n\
               • Progression:       {avg_prog:.4}  (last: {last_prog:.4})\n\
               • Prog. Penalty:     {avg_pen:.4}  (last: {last_pen:.4})\n\n\
             Generation Characteristics:\n\
               • Story Length:      {avg_len:.1} tokens  (last: {last_len:.1})\n\
               • Lexical Diversity: {avg_div:.4}  (last: {last_div:.4})\n\
               • Batches Recorded:  {batches}  ({total_stories} stories evaluated)\n\
             ================================================================================",
            avg_loss = self.history.loss_sum / count,
            last_loss = last.loss,
            avg_kl = self.history.kl_sum / count,
            last_kl = last.kl,
            avg_logprob = self.history.logprob_sum / count,
            last_logprob = last.logprob,
            avg_adv = self.history.advantage_sum / count,
            last_adv = last.advantage,
            last_lr = last.learning_rate,
            avg_reward = self.history.reward_sum / count,
            last_reward = last.reward,
            avg_centered = self.history.centered_sum / count,
            last_centered = last.centered,
            avg_rubric = self.history.rubric_sum / count,
            last_rubric = last.rubric,
            avg_raw = self.history.raw_rubric_sum / count,
            last_raw = last.raw_rubric,
            avg_prog = self.history.progression_sum / count,
            last_prog = last.progression,
            avg_pen = self.history.prog_penalty_sum / count,
            last_pen = last.prog_penalty,
            avg_len = self.history.story_len_sum / count,
            last_len = last.story_len,
            avg_div = self.history.diversity_sum / count,
            last_div = last.diversity,
            batches = self.history.total_steps,
            total_stories = self.history.total_stories,
        )
    }

    pub fn close(self) -> Result<(), String> {
        let mut renderer = self.renderer;
        renderer.end_split();
        renderer.update_epoch(1);
        renderer.end();
        let ended = renderer.on_train_end(None).map_err(|err| err.to_string());
        drop(renderer);
        release_terminal();
        ended
    }
}

struct MetricSlot {
    metric: ScalarMetric,
    id: MetricId,
}

impl MetricSlot {
    fn new(renderer: &mut dyn MetricsRenderer, name: &str, higher_is_better: bool) -> Self {
        let metric = ScalarMetric::new(name, higher_is_better);
        let id = register(renderer, &metric);
        Self { metric, id }
    }

    fn update(
        &mut self,
        renderer: &mut dyn MetricsRenderer,
        value: f64,
        count: usize,
        metadata: &MetricMetadata,
    ) -> Result<(), String> {
        let entry = self
            .metric
            .update(
                &WeightedScalar {
                    value,
                    count: count.max(1),
                },
                metadata,
            )
            .map_err(|err| err.to_string())?;
        let numeric_value = self.metric.value();
        let id = self.id.clone();
        renderer.update_train(MetricState::Numeric(
            MetricEntry::new(id, entry),
            numeric_value,
        ));
        Ok(())
    }
}

struct LearningRateSlot {
    metric: LearningRateMetric,
    id: MetricId,
}

impl LearningRateSlot {
    fn new(renderer: &mut dyn MetricsRenderer) -> Self {
        let metric = LearningRateMetric::new();
        let id = register(renderer, &metric);
        Self { metric, id }
    }

    fn update(
        &mut self,
        renderer: &mut dyn MetricsRenderer,
        metadata: &MetricMetadata,
    ) -> Result<(), String> {
        let entry = self
            .metric
            .update(&(), metadata)
            .map_err(|err| err.to_string())?;
        let numeric_value = self.metric.value();
        let id = self.id.clone();
        renderer.update_train(MetricState::Numeric(
            MetricEntry::new(id, entry),
            numeric_value,
        ));
        Ok(())
    }
}

/// Restores the tty after the dashboard, including when training returns early.
pub struct TerminalGuard;

impl TerminalGuard {
    pub fn arm() -> Self {
        Self
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        release_terminal();
    }
}

fn release_terminal() {
    use std::io::{IsTerminal, Write};

    let mut out = std::io::stdout();
    if !out.is_terminal() {
        return;
    }
    let _ = write!(
        out,
        "\r\n\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l\x1b[?2004l\x1b[?1049l\x1b[?25h"
    );
    let _ = out.flush();
    drain_pending_input();
}

fn drain_pending_input() {
    #[cfg(unix)]
    {
        const STDIN: i32 = 0;
        #[cfg(target_os = "linux")]
        const O_NONBLOCK: i32 = 0o4000;
        #[cfg(target_os = "macos")]
        const O_NONBLOCK: i32 = 0x4;
        const F_GETFL: i32 = 3;
        const F_SETFL: i32 = 4;
        unsafe extern "C" {
            fn fcntl(fd: i32, cmd: i32, ...) -> i32;
            fn read(fd: i32, buf: *mut u8, count: usize) -> isize;
        }
        unsafe {
            let flags = fcntl(STDIN, F_GETFL);
            if flags < 0 {
                return;
            }
            if fcntl(STDIN, F_SETFL, flags | O_NONBLOCK) < 0 {
                return;
            }
            let mut buf = [0u8; 4096];
            loop {
                let read_bytes = read(STDIN, buf.as_mut_ptr(), buf.len());
                if read_bytes <= 0 {
                    break;
                }
            }
            let _ = fcntl(STDIN, F_SETFL, flags);
        }
    }
}

fn register(renderer: &mut dyn MetricsRenderer, metric: &impl Metric) -> MetricId {
    let id = MetricId::new(metric.name());
    renderer.register_metric(MetricDefinition::new(id.clone(), metric));
    id
}

struct WeightedScalar {
    value: f64,
    count: usize,
}

#[derive(Clone)]
struct ScalarMetric {
    name: MetricName,
    higher_is_better: bool,
    state: NumericMetricState,
}

impl ScalarMetric {
    fn new(name: &str, higher_is_better: bool) -> Self {
        Self {
            name: Arc::new(name.to_string()),
            higher_is_better,
            state: NumericMetricState::new(),
        }
    }
}

impl Metric for ScalarMetric {
    type Input = WeightedScalar;

    fn update(
        &mut self,
        item: &WeightedScalar,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        self.state.update(item.value, item.count.max(1));
        Ok(self
            .state
            .compute_update(FormatOptions::new(self.name()).precision(4)))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self
            .state
            .compute_final(FormatOptions::new(self.name()).precision(4)))
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: self.higher_is_better,
        }
        .into()
    }
}

impl Numeric for ScalarMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(self.state.running_value())
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_metrics_and_history_record_and_summarize() {
        let mut history = MetricsHistory::default();
        let metrics = StepMetrics {
            loss: 0.5,
            reward: 0.2,
            rubric: 0.7,
            raw_rubric: 0.8,
            progression: 0.6,
            prog_penalty: 0.1,
            story_len: 300.0,
            diversity: 0.75,
            advantage: 0.4,
            centered: 0.05,
            kl: 0.002,
            logprob: -2.5,
            learning_rate: 1e-6,
            stories: 4,
        };
        history.record(&metrics);
        assert_eq!(history.total_steps, 1);
        assert_eq!(history.total_stories, 4);
        assert!((history.loss_sum - 2.0).abs() < 1e-5);
        assert!((history.reward_sum - 0.8).abs() < 1e-5);
    }
}
