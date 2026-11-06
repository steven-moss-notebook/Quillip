//! 24 kHz mono speech clips from Kokoro, played by Bevy's audio engine.

use std::time::Duration;

use bevy::audio::{ChannelCount, SampleRate, Source};
use bevy::prelude::*;

pub const SAMPLE_RATE: u32 = 24_000;

#[derive(Asset, TypePath, Clone)]
pub struct Speech {
    pub samples: Vec<f32>,
}

pub struct SpeechDecoder {
    samples: Vec<f32>,
    index: usize,
    rate: SampleRate,
}

impl Iterator for SpeechDecoder {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let sample = self.samples.get(self.index).copied()?;
        self.index += 1;
        Some(sample)
    }
}

impl Source for SpeechDecoder {
    fn current_span_len(&self) -> Option<usize> {
        Some(self.samples.len().saturating_sub(self.index))
    }

    fn channels(&self) -> ChannelCount {
        ChannelCount::new(1).expect("one channel")
    }

    fn sample_rate(&self) -> SampleRate {
        self.rate
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs_f64(
            self.samples.len() as f64 / f64::from(SAMPLE_RATE),
        ))
    }
}

impl Decodable for Speech {
    type Decoder = SpeechDecoder;

    fn decoder(&self) -> Self::Decoder {
        SpeechDecoder {
            samples: self.samples.clone(),
            index: 0,
            rate: SampleRate::new(SAMPLE_RATE).expect("24 kHz"),
        }
    }
}

pub fn with_gap(samples: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len() + 3_600);
    out.extend_from_slice(samples);
    out.extend(std::iter::repeat_n(0.0, 3_600));
    out
}
