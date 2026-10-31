//! Checkpoint-derived Engram metadata. Rust tokenization and scalar RNG setup
//! replace the release's non-persistent Python/NumPy initialization artifacts.
use super::{DeepseekV41EngramTokens, DeepseekV41TextConfig};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use tokenizers::{normalizers::NormalizerWrapper, NormalizedString, Normalizer, Tokenizer};

impl DeepseekV41EngramTokens {
    pub fn from_tokenizer(tokenizer: &Tokenizer, text: &DeepseekV41TextConfig) -> Result<Self> {
        let normalizer: NormalizerWrapper = serde_json::from_value(serde_json::json!({
            "type":"Sequence", "normalizers":[
                {"type":"NFKC"}, {"type":"NFD"}, {"type":"StripAccents"}, {"type":"Lowercase"},
                {"type":"Replace","pattern":{"Regex":"[ \\t\\r\\n]+"},"content":" "},
                {"type":"Replace","pattern":{"Regex":"^ $"},"content":"\u{e000}"},
                {"type":"Strip","strip_left":true,"strip_right":true},
                {"type":"Replace","pattern":{"String":"\u{e000}"},"content":" "}
            ]
        }))?;
        if tokenizer.get_vocab_size(true) != text.vocab_size {
            bail!("V4.1 tokenizer vocabulary does not match the embedding table");
        }
        let mut ids = HashMap::new();
        let mut token_map = Vec::with_capacity(text.vocab_size);
        for id in 0..text.vocab_size as u32 {
            let decoded = tokenizer
                .decode(&[id], false)
                .map_err(|e| anyhow::anyhow!("V4.1 tokenizer decode: {e}"))?;
            let key = if decoded.contains('\u{fffd}') {
                tokenizer
                    .id_to_token(id)
                    .context("V4.1 tokenizer has a vocabulary hole")?
            } else {
                let mut normalized = NormalizedString::from(decoded.as_str());
                normalizer
                    .normalize(&mut normalized)
                    .map_err(|e| anyhow::anyhow!("V4.1 token normalization: {e}"))?;
                if normalized.get().is_empty() {
                    decoded
                } else {
                    normalized.get().to_owned()
                }
            };
            let next = ids.len() as u32;
            token_map.push(*ids.entry(key).or_insert(next));
        }
        if ids.len() != text.engram_compressed_vocab_size {
            bail!(
                "V4.1 compressed tokenizer has {} ids, expected {}",
                ids.len(),
                text.engram_compressed_vocab_size
            );
        }
        let multipliers = text
            .engram_layer_ids
            .iter()
            .map(|&layer| hash_multipliers(layer, text.engram_max_ngram_size, ids.len()))
            .collect::<Result<_>>()?;
        Ok(Self {
            token_map,
            multipliers,
            pad_token_id: text.engram_pad_token_id,
        })
    }
}

/// SeedSequence -> PCG64 XSL-RR -> bounded integers, matching NumPy's published
/// integer-seed algorithm. This is initialization only, never model arithmetic.
/// NumPy/PCG attribution and licenses are in THIRD_PARTY_NOTICES.md.
fn hash_multipliers(layer: usize, count: usize, vocab: usize) -> Result<Vec<u64>> {
    if vocab == 0 {
        bail!("V4.1 compressed vocabulary must be nonzero");
    }
    let seed = (layer as u64)
        .checked_mul(10007)
        .context("V4.1 hash seed overflow")?;
    let bound = ((i64::MAX as u64 / vocab as u64) / 2).max(1);
    let mut generator = Pcg64::new(seed);
    Ok((0..count)
        .map(|_| generator.bounded(bound) * 2 + 1)
        .collect())
}

fn seed_words(seed: u64) -> [u64; 4] {
    fn hash(value: u32, constant: &mut u32) -> u32 {
        let mut result = value ^ *constant;
        *constant = constant.wrapping_mul(0x931e8875);
        result = result.wrapping_mul(*constant);
        result ^ (result >> 16)
    }
    let entropy = [seed as u32, (seed >> 32) as u32];
    let mut constant = 0x43b0d7e5;
    let mut pool = [0u32; 4];
    for i in 0..4 {
        pool[i] = hash(if i < 2 { entropy[i] } else { 0 }, &mut constant);
    }
    for src in 0..4 {
        for dst in 0..4 {
            if src != dst {
                let value = hash(pool[src], &mut constant);
                let mixed = pool[dst]
                    .wrapping_mul(0xca01f9dd)
                    .wrapping_sub(value.wrapping_mul(0x4973f715));
                pool[dst] = mixed ^ (mixed >> 16);
            }
        }
    }
    let mut words = [0u32; 8];
    constant = 0x8b51f9dd;
    for i in 0..8 {
        let value = pool[i % 4] ^ constant;
        constant = constant.wrapping_mul(0x58f38ded);
        let value = value.wrapping_mul(constant);
        words[i] = value ^ (value >> 16);
    }
    std::array::from_fn(|i| words[2 * i] as u64 | ((words[2 * i + 1] as u64) << 32))
}

struct Pcg64 {
    state: u128,
    increment: u128,
    pending: Option<u32>,
}
impl Pcg64 {
    const MULT: u128 = ((2549297995355413924u128) << 64) | 4865540595714422341u128;
    fn new(seed: u64) -> Self {
        let words = seed_words(seed);
        let initial = (words[0] as u128) << 64 | words[1] as u128;
        let sequence = (words[2] as u128) << 64 | words[3] as u128;
        let increment = sequence.wrapping_shl(1) | 1;
        Self {
            state: increment
                .wrapping_add(initial)
                .wrapping_mul(Self::MULT)
                .wrapping_add(increment),
            increment,
            pending: None,
        }
    }
    fn next64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(Self::MULT)
            .wrapping_add(self.increment);
        ((self.state >> 64) as u64 ^ self.state as u64).rotate_right((self.state >> 122) as u32)
    }
    fn next32(&mut self) -> u32 {
        if let Some(value) = self.pending.take() {
            return value;
        }
        let value = self.next64();
        self.pending = Some((value >> 32) as u32);
        value as u32
    }
    fn bounded(&mut self, bound: u64) -> u64 {
        if bound == 1 {
            return 0;
        }
        if bound <= (1u64 << 32) {
            if bound == (1u64 << 32) {
                return self.next32() as u64;
            }
            let threshold = (bound as u32).wrapping_neg() % bound as u32;
            loop {
                let product = self.next32() as u64 * bound;
                if product as u32 >= threshold {
                    return product >> 32;
                }
            }
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = self.next64() as u128 * bound as u128;
            if product as u64 >= threshold {
                return (product >> 64) as u64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;

    /// The checkpoint does not carry Engram metadata. A tokenizer whose raw
    /// spellings normalize onto the fixture's compressed ids must reproduce the
    /// exact lookup table and layer-seeded multipliers without sidecar files.
    #[test]
    fn tokenizer_derives_fixture_engram_map_and_multipliers() -> Result<()> {
        let source: serde_json::Value = serde_json::from_str(include_str!(
            "../../validation/fixtures/deepseek_v41_bf16_model.json"
        ))?;
        let case = &source["cases"][0];
        let text: DeepseekV41TextConfig =
            serde_json::from_value(case["config"]["text_config"].clone())?;
        let expected: Vec<u32> = serde_json::from_value(case["tokens"]["token_map"].clone())?;
        assert_eq!(expected.len(), text.vocab_size);
        assert_eq!(text.engram_compressed_vocab_size, 8);
        let mut seen = vec![0usize; text.engram_compressed_vocab_size];
        let mut vocab = HashMap::new();
        for (id, &group) in expected.iter().enumerate() {
            let base = (b'a' + group as u8) as char;
            let token = match seen[group as usize] {
                0 => base.to_string(),
                1 => base.to_ascii_uppercase().to_string(),
                2 => format!("{base} "),
                _ => format!(" {base} "),
            };
            seen[group as usize] += 1;
            vocab.insert(token, id as u32);
        }
        assert_eq!(vocab.len(), text.vocab_size);
        let tokenizer: Tokenizer = serde_json::from_value(serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "a"}
        }))?;
        let tokens = DeepseekV41EngramTokens::from_tokenizer(&tokenizer, &text)?;
        assert_eq!(tokens.token_map, expected);
        assert_eq!(tokens.pad_token_id, text.engram_pad_token_id);

        let hashes: serde_json::Value = serde_json::from_str(include_str!(
            "../../validation/fixtures/deepseek_v41_hash_seeding.json"
        ))?;
        let reference = hashes
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["layer"] == text.engram_layer_ids[0] && case["vocab"] == 8)
            .unwrap();
        let multipliers: Vec<u64> = serde_json::from_value(reference["multipliers"].clone())?;
        assert_eq!(
            tokens.multipliers[0],
            multipliers[..text.engram_max_ngram_size]
        );
        Ok(())
    }

    /// Opt-in check against the published V4.1 tokenizer (not shipped in the
    /// weights repository). Set `V41_TOKENIZER` to its tokenizer.json.
    #[test]
    #[ignore = "requires the published V4.1 tokenizer.json via V41_TOKENIZER"]
    fn published_v4_1_tokenizer_reproduces_released_engram_metadata() -> Result<()> {
        let path = std::env::var("V41_TOKENIZER")
            .context("V41_TOKENIZER must point to the published tokenizer.json")?;
        let package: serde_json::Value = serde_json::from_str(include_str!(
            "../../validation/fixtures/deepseek_v41_flash_1017_config.json"
        ))?;
        let text = crate::deepseek_v41::DeepseekV41Config::from_hf_value(&package)?.text;
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("V4.1 published tokenizer: {e}"))?;
        let tokens = DeepseekV41EngramTokens::from_tokenizer(&tokenizer, &text)?;
        assert_eq!(tokens.token_map.len(), text.vocab_size);
        assert_eq!(tokens.multipliers.len(), text.engram_layer_ids.len());
        assert_eq!(tokens.multipliers[0].len(), text.engram_max_ngram_size);
        let hashes: serde_json::Value = serde_json::from_str(include_str!(
            "../../validation/fixtures/deepseek_v41_hash_seeding.json"
        ))?;
        let reference = hashes
            .as_array()
            .unwrap()
            .iter()
            .find(|case| {
                case["layer"] == text.engram_layer_ids[0]
                    && case["vocab"] == text.engram_compressed_vocab_size
            })
            .unwrap();
        let multipliers: Vec<u64> = serde_json::from_value(reference["multipliers"].clone())?;
        assert_eq!(
            tokens.multipliers[0],
            multipliers[..text.engram_max_ngram_size]
        );
        Ok(())
    }

    #[test]
    fn numpy_engram_hash_multipliers_match_frozen_reference() -> Result<()> {
        let cases: serde_json::Value = serde_json::from_str(include_str!(
            "../../validation/fixtures/deepseek_v41_hash_seeding.json"
        ))?;
        for case in cases.as_array().unwrap() {
            let expected: Vec<u64> = serde_json::from_value(case["multipliers"].clone())?;
            assert_eq!(
                hash_multipliers(
                    case["layer"].as_u64().unwrap() as usize,
                    expected.len(),
                    case["vocab"].as_u64().unwrap() as usize
                )?,
                expected
            );
        }
        Ok(())
    }
}
