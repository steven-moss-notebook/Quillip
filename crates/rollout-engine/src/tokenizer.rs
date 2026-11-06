use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
}

/// Hugging Face `tokenizer.json` shipped with Spark-X2.6-1.7B.
pub struct SparkTokenizer {
    inner: tokenizers::Tokenizer,
}

impl SparkTokenizer {
    pub fn open(path: &Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|err| format!("reading {}: {err}", path.display()))?;
        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|err| format!("reading {}: {err}", path.display()))?;
        Ok(Self { inner })
    }

    pub fn try_encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|err| err.to_string())?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Token ids and byte offsets into `text`.
    ///
    /// Offsets come from the tokenizer's default byte referential, so they
    /// index the UTF-8 bytes of `text` rather than its characters.
    pub fn encode_offsets(&self, text: &str) -> Result<(Vec<u32>, Vec<(usize, usize)>), String> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|err| err.to_string())?;
        let ids = encoding.get_ids().to_vec();
        let offsets = encoding.get_offsets().to_vec();
        if ids.len() != offsets.len() {
            return Err("tokenizer offsets do not match token ids".into());
        }
        Ok((ids, offsets))
    }
}

impl Tokenizer for SparkTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.try_encode(text).unwrap_or_default()
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).unwrap_or_default()
    }
}

/// ASCII byte tokenizer used for pipeline tests.
///
/// These ids are not the Spark vocabulary.
#[derive(Debug, Default, Clone, Copy)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|byte| u32::from(byte)).collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        let bytes = ids
            .iter()
            .filter_map(|id| u8::try_from(*id).ok())
            .collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[derive(Debug, Clone)]
pub struct VocabTokenizer {
    token_to_id: BTreeMap<String, u32>,
    id_to_token: BTreeMap<u32, String>,
}

#[derive(Debug, Deserialize)]
struct VocabFile {
    tokens: BTreeMap<String, u32>,
}

impl VocabTokenizer {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let file: VocabFile = serde_json::from_str(text).map_err(|err| err.to_string())?;
        let id_to_token = file
            .tokens
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect();
        Ok(Self {
            token_to_id: file.tokens,
            id_to_token,
        })
    }
}

impl Tokenizer for VocabTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|token| self.token_to_id.get(token).copied().unwrap_or(0))
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|id| {
                self.id_to_token
                    .get(id)
                    .map(String::as_str)
                    .unwrap_or("<unk>")
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_round_trips_ascii() {
        let tokenizer = ByteTokenizer;
        let text = "Mara opened the ledger.";
        assert_eq!(tokenizer.decode(&tokenizer.encode(text)), text);
    }

    #[test]
    fn vocab_tokenizer_uses_declared_ids() {
        let tokenizer = VocabTokenizer::from_json(r#"{"tokens":{"Mara":3,"ledger":9}}"#).unwrap();
        assert_eq!(tokenizer.encode("Mara ledger"), vec![3, 9]);
        assert_eq!(tokenizer.decode(&[3, 9]), "Mara ledger");
    }

    #[test]
    fn spark_byte_level_round_trips_the_words_that_came_back_corrupted() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../checkpoints/XHToken/Spark-X2.6-1.7B/tokenizer.json");
        if !path.exists() {
            return;
        }
        let tokenizer = SparkTokenizer::open(&path).unwrap();
        let samples = [
            "tools",
            "sun-dried",
            "surfaces",
            "metals",
            "there's",
            "trails",
            "The tools were sun-dried on metal surfaces along the trails.",
            "There's a box of tools.",
        ];
        for text in samples {
            let ids = tokenizer.encode(text);
            let decoded = tokenizer.decode(&ids);
            assert_eq!(decoded, text, "ids={ids:?} for {text:?}");
        }
    }
}
