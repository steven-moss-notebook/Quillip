//! Download the Spark-X2.6-1.7B snapshot from Hugging Face.
//!
//! The repository is fixed to `XHToken/Spark-X2.6-1.7B`. Weights are published
//! as two safetensors shards plus `model.safetensors.index.json`. This module
//! fetches those shards, the config, and the tokenizer, and leaves the
//! modeling scripts and images on the Hub.

use std::path::{Path, PathBuf};

use hf_hub::HFClientSync;
use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};

pub const REPO_OWNER: &str = "XHToken";
pub const REPO_NAME: &str = "Spark-X2.6-1.7B";
pub const DEFAULT_DIR: &str = "checkpoints/XHToken/Spark-X2.6-1.7B";

const SNAPSHOT_GLOBS: &[&str] = &[
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

pub struct SparkSnapshot {
    pub directory: PathBuf,
    pub config: PathBuf,
    pub weights: Vec<PathBuf>,
}

struct DownloadLog;

impl ProgressHandler for DownloadLog {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Download(DownloadEvent::Start {
                total_files,
                total_bytes,
            }) => {
                eprintln!(
                    "downloading {total_files} files ({total_bytes} bytes) from {REPO_OWNER}/{REPO_NAME}"
                );
            }
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            }) => {
                eprint!("\r{bytes_completed} / {total_bytes} bytes");
            }
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                for file in files {
                    if file.status == hf_hub::progress::FileStatus::Complete {
                        eprintln!("\n{}", file.filename);
                    }
                }
            }
            ProgressEvent::Download(DownloadEvent::Complete) => {
                eprintln!("\nsnapshot complete");
            }
            ProgressEvent::Upload(_) => {}
        }
    }
}

pub fn download_spark(directory: &Path) -> Result<SparkSnapshot, String> {
    std::fs::create_dir_all(directory).map_err(|err| err.to_string())?;
    let client = HFClientSync::new().map_err(|err| err.to_string())?;
    let repo = client.model(REPO_OWNER, REPO_NAME);
    let patterns = SNAPSHOT_GLOBS
        .iter()
        .map(|pattern| (*pattern).to_string())
        .collect();
    repo.snapshot_download()
        .local_dir(directory)
        .allow_patterns(patterns)
        .progress(Progress::new(DownloadLog))
        .send()
        .map_err(|err| err.to_string())?;
    snapshot_from_directory(directory)
}

/// Use a snapshot that is already on disk. Search the working directory and
/// the executable's parents, so the binary still finds `checkpoints/` when it
/// is launched from `target/release`.
pub fn resolve_checkpoint(requested: Option<&Path>) -> Result<PathBuf, String> {
    let relative = requested
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DIR));
    if let Some(found) = find_ready(&relative) {
        return Ok(found);
    }
    if let Some(partial) = find_directory(&relative) {
        return Err(missing_snapshot_message(&partial));
    }
    if requested.is_some() {
        return Err(format!("checkpoint {} does not exist", relative.display()));
    }
    let dest = find_project_root()
        .map(|root| root.join(&relative))
        .unwrap_or(relative);
    download_spark(&dest).map(|snapshot| snapshot.directory)
}

fn find_ready(relative: &Path) -> Option<PathBuf> {
    find_candidate(relative, snapshot_ready)
}

fn find_directory(relative: &Path) -> Option<PathBuf> {
    find_candidate(relative, |path| path.is_dir())
}

fn find_candidate(relative: &Path, accept: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    if relative.is_absolute() {
        return accept(relative).then(|| relative.to_path_buf());
    }
    search_roots()
        .into_iter()
        .find_map(|start| find_candidate_from(&start, relative, &accept))
}

fn find_candidate_from(
    start: &Path,
    relative: &Path,
    accept: &impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    let mut current = Some(start.to_path_buf());
    for _ in 0..10 {
        let Some(dir) = current else { break };
        let candidate = dir.join(relative);
        if accept(&candidate) {
            return Some(candidate);
        }
        current = dir.parent().map(Path::to_path_buf);
    }
    None
}

fn search_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        roots.push(parent.to_path_buf());
    }
    roots
}

fn find_project_root() -> Option<PathBuf> {
    for start in search_roots() {
        let mut current = Some(start);
        for _ in 0..10 {
            let Some(dir) = current else { break };
            if dir.join("Cargo.toml").is_file() && dir.join("crates/spark-story").is_dir() {
                return Some(dir);
            }
            current = dir.parent().map(Path::to_path_buf);
        }
    }
    None
}

fn snapshot_ready(dir: &Path) -> bool {
    missing_snapshot_files(dir).is_empty()
}

fn missing_snapshot_message(dir: &Path) -> String {
    let missing = missing_snapshot_files(dir);
    let lines = missing
        .iter()
        .map(|path| format!("  {}", path.display()))
        .collect::<Vec<_>>()
        .join("\n");
    format!("Spark checkpoint {} is missing:\n{lines}", dir.display())
}

fn missing_snapshot_files(dir: &Path) -> Vec<PathBuf> {
    let mut missing = Vec::new();
    for name in ["config.json", "tokenizer.json"] {
        let path = dir.join(name);
        if !path.is_file() {
            missing.push(path);
        }
    }
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        match shard_names(&index) {
            Ok(names) if !names.is_empty() => {
                for name in names {
                    let path = dir.join(name);
                    if !path.is_file() {
                        missing.push(path);
                    }
                }
            }
            _ => missing.push(index),
        }
    } else if !directory_has_safetensors(dir) {
        missing.push(index);
    }
    missing
}

fn shard_names(index: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(index).map_err(|err| err.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| err.to_string())?;
    let Some(map) = value.get("weight_map").and_then(|map| map.as_object()) else {
        return Ok(Vec::new());
    };
    let mut names: Vec<String> = map
        .values()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}

fn directory_has_safetensors(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        })
}

pub fn snapshot_from_directory(directory: &Path) -> Result<SparkSnapshot, String> {
    let config = directory.join("config.json");
    if !config.is_file() {
        return Err(format!("missing {}", config.display()));
    }
    let mut weights = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(|err| err.to_string())? {
        let path = entry.map_err(|err| err.to_string())?.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            weights.push(path);
        }
    }
    weights.sort();
    if weights.is_empty() {
        return Err(format!(
            "{} has no safetensors weights",
            directory.display()
        ));
    }
    Ok(SparkSnapshot {
        directory: directory.to_path_buf(),
        config,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_requires_config_and_a_weight_file() {
        let dir = std::env::temp_dir().join(format!("spark-snapshot-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(snapshot_from_directory(&dir).is_err());
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(
            dir.join("model-00001-of-00002.safetensors"),
            b"not a tensor",
        )
        .unwrap();
        let snapshot = snapshot_from_directory(&dir).unwrap();
        assert_eq!(snapshot.weights.len(), 1);
        assert!(snapshot.config.ends_with("config.json"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_snapshot_is_found_from_a_nested_working_directory() {
        let root = std::env::temp_dir().join(format!("spark-locate-{}", std::process::id()));
        let snapshot = root.join(DEFAULT_DIR);
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("config.json"), "{}").unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            r#"{"weight_map":{"embed":"model-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let nested = root.join("target/release");
        let found = find_candidate_from(&nested, Path::new(DEFAULT_DIR), &snapshot_ready);
        assert!(found.is_none());
        std::fs::write(
            snapshot.join("model-00001-of-00002.safetensors"),
            b"weights",
        )
        .unwrap();
        let found = find_candidate_from(&nested, Path::new(DEFAULT_DIR), &snapshot_ready).unwrap();
        assert_eq!(found, snapshot);
        let message = missing_snapshot_message(&snapshot);
        assert!(!message.contains("model-00001"));
        std::fs::remove_file(snapshot.join("tokenizer.json")).unwrap();
        let message = missing_snapshot_message(&snapshot);
        assert!(message.contains("tokenizer.json"), "{message}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
