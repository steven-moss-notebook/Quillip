//! Fetch Spark and the Kokoro voice model when they are not already cached.
//!
//! Spark comes from the Hugging Face hub cache. Kokoro's runtime files are the
//! `0.onnx` model and `0.bin` voices that kokoro-micro loads from `~/.cache/k`.
//! A copy is kept under the Hugging Face hub cache so a later launch can see
//! both without downloading again.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;

use hf_hub::HFClientSync;
use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};

use crate::engine::{Phase, WorkerEvent};

const SPARK_OWNER: &str = "XHToken";
const SPARK_NAME: &str = "Spark-X2.6-1.7B";
const KOKORO_MODEL_URL: &str = "https://github.com/8b-is/kokoro-tiny/raw/main/models/0.onnx";
const KOKORO_VOICES_URL: &str = "https://github.com/8b-is/kokoro-tiny/raw/main/models/0.bin";
const KOKORO_MODEL_BYTES: u64 = 310 * 1024 * 1024;
const KOKORO_VOICES_BYTES: u64 = 27 * 1024 * 1024;

const SPARK_GLOBS: &[&str] = &[
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
];

pub fn prepare(checkpoint: &mut PathBuf, events: &Sender<WorkerEvent>) -> Result<(), String> {
    if let Some(found) = find_ready_checkpoint(checkpoint) {
        *checkpoint = found;
    } else if let Some(found) = spark_snapshot_in_cache() {
        *checkpoint = found;
    } else {
        let dest = download_destination(checkpoint);
        download_spark(&dest, events)?;
        *checkpoint = dest;
    }
    ensure_kokoro(events)?;
    Ok(())
}

fn download_spark(directory: &Path, events: &Sender<WorkerEvent>) -> Result<(), String> {
    report(
        events,
        Phase::DownloadingSpark,
        "Asking the hub for the file sizes",
        0,
        1,
    );
    fs::create_dir_all(directory).map_err(|err| err.to_string())?;
    let client = HFClientSync::new().map_err(|err| err.to_string())?;
    let repo = client.model(SPARK_OWNER, SPARK_NAME);
    let patterns = SPARK_GLOBS
        .iter()
        .map(|pattern| (*pattern).to_string())
        .collect();
    repo.snapshot_download()
        .local_dir(directory)
        .allow_patterns(patterns)
        .progress(Progress::new(HubProgress {
            events: events.clone(),
            files: Mutex::new(HashMap::new()),
            total: AtomicU64::new(0),
        }))
        .send()
        .map_err(|err| err.to_string())?;
    if !dir_ready(directory) {
        return Err(format!(
            "Spark finished downloading, but {} is still missing weights or a tokenizer",
            directory.display()
        ));
    }
    report(
        events,
        Phase::DownloadingSpark,
        "Spark is in the hub cache",
        1,
        1,
    );
    Ok(())
}

struct HubProgress {
    events: Sender<WorkerEvent>,
    files: Mutex<HashMap<String, (u64, u64)>>,
    total: AtomicU64,
}

impl ProgressHandler for HubProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                self.total.store(*total_bytes, Ordering::Relaxed);
                let (step, total) = steps(0, *total_bytes);
                report(
                    &self.events,
                    Phase::DownloadingSpark,
                    &format!("0 of {}", megabytes(*total_bytes)),
                    step,
                    total,
                );
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => {
                let (step, total) = steps(*bytes_completed, *total_bytes);
                report(
                    &self.events,
                    Phase::DownloadingSpark,
                    &format!(
                        "{} of {}",
                        megabytes(*bytes_completed),
                        megabytes(*total_bytes)
                    ),
                    step,
                    total,
                );
            }
            DownloadEvent::Progress { files } => {
                let Ok(mut known) = self.files.lock() else {
                    return;
                };
                for file in files {
                    known.insert(
                        file.filename.clone(),
                        (file.bytes_completed, file.total_bytes),
                    );
                }
                let done: u64 = known.values().map(|(done, _)| *done).sum();
                let hinted = self.total.load(Ordering::Relaxed);
                let summed: u64 = known.values().map(|(_, total)| *total).sum();
                let total_bytes = if hinted > 0 { hinted } else { summed };
                let (step, total) = steps(done, total_bytes);
                report(
                    &self.events,
                    Phase::DownloadingSpark,
                    &format!("{} of {}", megabytes(done), megabytes(total_bytes)),
                    step,
                    total,
                );
            }
            DownloadEvent::Complete => {
                report(
                    &self.events,
                    Phase::DownloadingSpark,
                    "Spark is in the hub cache",
                    1,
                    1,
                );
            }
        }
    }
}

fn ensure_kokoro(events: &Sender<WorkerEvent>) -> Result<(), String> {
    let runtime = kokoro_runtime_dir();
    let cached = kokoro_cache_dir();
    fs::create_dir_all(&runtime).map_err(|err| err.to_string())?;
    fs::create_dir_all(&cached).map_err(|err| err.to_string())?;
    if pair_ready(&cached) && !pair_ready(&runtime) {
        copy_pair(&cached, &runtime)?;
        return Ok(());
    }
    if pair_ready(&runtime) && !pair_ready(&cached) {
        copy_pair(&runtime, &cached)?;
        return Ok(());
    }
    if pair_ready(&runtime) && pair_ready(&cached) {
        return Ok(());
    }
    let files = [
        ("0.onnx", KOKORO_MODEL_URL, KOKORO_MODEL_BYTES),
        ("0.bin", KOKORO_VOICES_URL, KOKORO_VOICES_BYTES),
    ];
    for (index, (name, url, estimate)) in files.iter().enumerate() {
        let destination = cached.join(name);
        if file_ready(&destination) {
            continue;
        }
        download_file(url, &destination, index, files.len(), *estimate, events)?;
    }
    copy_pair(&cached, &runtime)?;
    report(
        events,
        Phase::DownloadingVoice,
        "Kokoro is in the hub cache",
        1,
        1,
    );
    Ok(())
}

fn download_file(
    url: &str,
    destination: &Path,
    index: usize,
    count: usize,
    estimate: u64,
    events: &Sender<WorkerEvent>,
) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .map_err(|err| err.to_string())?;
    let mut response = client.get(url).send().map_err(|err| err.to_string())?;
    if !response.status().is_success() {
        return Err(format!(
            "Kokoro download returned {} for {url}",
            response.status()
        ));
    }
    let total = response.content_length().unwrap_or(estimate);
    let partial = destination.with_extension("partial");
    let mut file = File::create(&partial).map_err(|err| err.to_string())?;
    let mut done = 0u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = response.read(&mut buf).map_err(|err| err.to_string())?;
        if read == 0 {
            break;
        }
        file.write_all(&buf[..read])
            .map_err(|err| err.to_string())?;
        done += read as u64;
        let (step, steps_total) = steps(done, total);
        report(
            events,
            Phase::DownloadingVoice,
            &format!(
                "File {} of {count}, {} of {}",
                index + 1,
                megabytes(done),
                megabytes(total)
            ),
            step,
            steps_total,
        );
    }
    file.flush().map_err(|err| err.to_string())?;
    fs::rename(&partial, destination).map_err(|err| err.to_string())?;
    Ok(())
}

fn copy_pair(from: &Path, to: &Path) -> Result<(), String> {
    fs::create_dir_all(to).map_err(|err| err.to_string())?;
    for name in ["0.onnx", "0.bin"] {
        let source = from.join(name);
        let dest = to.join(name);
        if file_ready(&source) && !file_ready(&dest) {
            fs::copy(&source, &dest).map_err(|err| err.to_string())?;
        }
    }
    if !pair_ready(to) {
        return Err(format!("Kokoro files are missing from {}", to.display()));
    }
    Ok(())
}

fn find_ready_checkpoint(requested: &Path) -> Option<PathBuf> {
    if requested.is_absolute() {
        return dir_ready(requested).then(|| requested.to_path_buf());
    }
    let mut roots = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        roots.push(parent.to_path_buf());
    }
    roots.into_iter().flat_map(ancestors).find_map(|dir| {
        let candidate = dir.join(requested);
        dir_ready(&candidate).then_some(candidate)
    })
}

fn download_destination(requested: &Path) -> PathBuf {
    if requested.is_absolute() {
        return requested.to_path_buf();
    }
    if let Ok(cwd) = std::env::current_dir() {
        for dir in ancestors(cwd) {
            if dir.join("Cargo.toml").is_file() && dir.join("crates").is_dir() {
                return dir.join(requested);
            }
        }
    }
    requested.to_path_buf()
}

fn ancestors(start: PathBuf) -> impl Iterator<Item = PathBuf> {
    std::iter::successors(Some(start), |dir| dir.parent().map(Path::to_path_buf)).take(8)
}

fn spark_snapshot_in_cache() -> Option<PathBuf> {
    let snapshots = hf_hub_cache()
        .join(format!("models--{SPARK_OWNER}--{SPARK_NAME}"))
        .join("snapshots");
    let entries = fs::read_dir(snapshots).ok()?;
    entries.filter_map(|entry| entry.ok()).find_map(|entry| {
        let path = entry.path();
        dir_ready(&path).then_some(path)
    })
}

pub(crate) fn hf_hub_cache() -> PathBuf {
    if let Ok(path) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(path);
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("huggingface").join("hub");
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

fn kokoro_cache_dir() -> PathBuf {
    hf_hub_cache().join("models--kokoro-micro--base/snapshots/main")
}

fn kokoro_runtime_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("k")
}

fn pair_ready(dir: &Path) -> bool {
    file_ready(&dir.join("0.onnx")) && file_ready(&dir.join("0.bin"))
}

fn dir_ready(dir: &Path) -> bool {
    file_ready(&dir.join("tokenizer.json"))
        && (file_ready(&dir.join("model.safetensors"))
            || dir.join("model.safetensors.index.json").is_file()
                && shard_files_ready(dir).unwrap_or(false))
}

fn shard_files_ready(dir: &Path) -> Result<bool, String> {
    let index = dir.join("model.safetensors.index.json");
    let text = fs::read_to_string(&index).map_err(|err| err.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| err.to_string())?;
    let Some(map) = value.get("weight_map").and_then(|map| map.as_object()) else {
        return Ok(false);
    };
    let mut names: Vec<&str> = map.values().filter_map(|value| value.as_str()).collect();
    names.sort();
    names.dedup();
    if names.is_empty() {
        return Ok(false);
    }
    Ok(names.iter().all(|name| file_ready(&dir.join(name))))
}

fn file_ready(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() > 0)
}

fn report(events: &Sender<WorkerEvent>, phase: Phase, detail: &str, step: u32, total: u32) {
    let _ = events.send(WorkerEvent::Status {
        phase,
        detail: detail.to_string(),
        step,
        total: total.max(1),
    });
}

fn steps(done: u64, total: u64) -> (u32, u32) {
    if total == 0 {
        return (0, 1000);
    }
    let step = ((done as u128 * 1000) / total as u128).min(1000) as u32;
    (step, 1000)
}

fn megabytes(bytes: u64) -> String {
    format!("{:.0} MB", bytes as f64 / 1_048_576.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_checkpoint_needs_a_tokenizer_and_weights() {
        let dir = std::env::temp_dir().join(format!("story-desk-fetch-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        assert!(!dir_ready(&dir));
        fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(dir.join("model.safetensors"), b"weights").unwrap();
        assert!(dir_ready(&dir));
        let _ = fs::remove_dir_all(&dir);
    }
}
