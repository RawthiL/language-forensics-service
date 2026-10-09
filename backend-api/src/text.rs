//! Shared implementation for sequence-classification text models: load and
//! verify the exported tokenizer, cut text into overlapping windows, pack
//! token tensors, softmax the logits. A family crate wraps it and adds only
//! what is specific to its model (validation, label conventions).

use std::path::Path;

use sha2::{Digest, Sha256};
use tokenizers::{Encoding, Tokenizer, TruncationDirection};

use crate::util::{self, softmax_positive};
use crate::{
    BackendError, CommonMeta, InputTensor, ModelInfo, OutputTensor, Result, TensorData, Window,
    WindowParams,
};

/// What each declared ONNX input carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Feed {
    Ids,
    Mask,
    TypeIds,
}

pub struct TextClassifier {
    pub meta: CommonMeta,
    tokenizer: Tokenizer,
    /// Special tokens the post-processor adds to a single sequence.
    specials: usize,
    feeds: Vec<(String, Feed)>,
}

impl TextClassifier {
    /// Load `meta.tokenizer` from the model directory, check its sha256
    /// against the exporter's, and disable any truncation/padding baked into
    /// the file (the gateway's window policy replaces both).
    pub fn load(info: &ModelInfo) -> Result<Self> {
        let meta = CommonMeta::parse(info)?;
        let path = info.model_dir.join(&meta.tokenizer);
        let tokenizer = load_tokenizer(info.name, &path, &meta.tokenizer_sha256)?;
        let specials = tokenizer
            .get_post_processor()
            .map(|p| tokenizers::PostProcessor::added_tokens(p, false))
            .unwrap_or(0);
        if specials >= meta.window.saturating_sub(meta.stride) {
            return Err(BackendError::InvalidModel(format!(
                "model '{}': window {} minus stride {} leaves no room for content \
                 after {specials} special tokens",
                info.name, meta.window, meta.stride
            )));
        }
        let feeds = meta
            .inputs
            .iter()
            .map(|n| {
                let feed = match n.as_str() {
                    "input_ids" => Feed::Ids,
                    "attention_mask" => Feed::Mask,
                    "token_type_ids" => Feed::TypeIds,
                    other => {
                        return Err(BackendError::InvalidModel(format!(
                            "model '{}': unsupported input '{other}' \
                             (expected input_ids, attention_mask, token_type_ids)",
                            info.name
                        )));
                    }
                };
                Ok((n.clone(), feed))
            })
            .collect::<Result<Vec<_>>>()?;
        if !feeds.iter().any(|(_, f)| *f == Feed::Ids) {
            return Err(BackendError::InvalidModel(format!(
                "model '{}': inputs {:?} lack input_ids",
                info.name, meta.inputs
            )));
        }
        Ok(Self {
            meta,
            tokenizer,
            specials,
            feeds,
        })
    }

    pub fn default_window(&self) -> WindowParams {
        WindowParams {
            window: self.meta.window,
            stride: self.meta.stride,
        }
    }

    /// Special tokens added around each window's content.
    pub fn special_tokens(&self) -> usize {
        self.specials
    }

    /// Cut `text` into windows of at most `params.window` tokens (special
    /// tokens included), consecutive windows sharing `params.stride` content
    /// tokens. The windows cover every token of the text.
    pub fn windows(&self, text: &str, params: &WindowParams) -> Result<Vec<Window>> {
        let content = params.window.saturating_sub(self.specials);
        if params.window > self.meta.max_tokens || content == 0 || params.stride >= content {
            return Err(BackendError::BadInput(format!(
                "window {} / stride {} invalid for this model (max {} tokens, {} special)",
                params.window, params.stride, self.meta.max_tokens, self.specials
            )));
        }
        // Char offsets refer to the ORIGINAL text: the tokenizer's normalizer
        // (NFKC, invisible-character stripping, ...) tracks alignments.
        let mut enc = self
            .tokenizer
            .encode_char_offsets(text, false)
            .map_err(|e| BackendError::BadInput(format!("tokenization failed: {e}")))?;
        let pieces = if enc.len() > content {
            enc.truncate(content, params.stride, TruncationDirection::Right);
            let rest = enc.take_overflowing();
            std::iter::once(enc).chain(rest).collect()
        } else {
            vec![enc]
        };
        pieces.into_iter().map(|p| self.finish(p)).collect()
    }

    fn finish(&self, piece: Encoding) -> Result<Window> {
        let offsets = piece.get_offsets();
        let (start, end) = match (offsets.first(), offsets.last()) {
            (Some(a), Some(b)) => (a.0, b.1.max(a.0)),
            _ => (0, 0),
        };
        let content_tokens = piece.len();
        let done = self
            .tokenizer
            .post_process(piece, None, true)
            .map_err(|e| BackendError::BadInput(format!("post-processing failed: {e}")))?;
        Ok(Window {
            ids: done.get_ids().to_vec(),
            start,
            end,
            content_tokens,
        })
    }

    /// Right-pad every window to `seq_len` and build the declared inputs.
    pub fn pack(&self, windows: &[&Window], seq_len: usize) -> Vec<InputTensor> {
        let n = windows.len();
        let shape = vec![n as u64, seq_len as u64];
        self.feeds
            .iter()
            .map(|(name, feed)| {
                let mut data = vec![0i64; n * seq_len];
                for (row, w) in windows.iter().enumerate() {
                    let dst = &mut data[row * seq_len..(row + 1) * seq_len];
                    let len = w.ids.len().min(seq_len);
                    match feed {
                        Feed::Ids => {
                            dst.fill(self.meta.pad_id as i64);
                            for (d, id) in dst.iter_mut().zip(&w.ids[..len]) {
                                *d = *id as i64;
                            }
                        }
                        Feed::Mask => dst[..len].fill(1),
                        Feed::TypeIds => {}
                    }
                }
                InputTensor {
                    name: name.clone(),
                    shape: shape.clone(),
                    data: TensorData::I64(data),
                }
            })
            .collect()
    }

    /// Logits `[n, labels]` -> summed softmax probability of the positive
    /// labels, one per window. A single-logit head is read as a sigmoid.
    pub fn scores(&self, outputs: &[OutputTensor], n: usize) -> Result<Vec<f64>> {
        let logits = util::output(outputs, &self.meta.outputs[0], 0)?;
        let width = self.meta.labels.len();
        let dims = &logits.shape;
        if dims.len() != 2 || dims[0] as usize != n || dims[1] as usize != width {
            return Err(BackendError::BadOutput(format!(
                "'{}' shape {dims:?}, expected [{n}, {width}]",
                logits.name
            )));
        }
        util::require_len(logits, n * width)?;
        Ok(logits
            .data
            .chunks_exact(width)
            .map(|row| {
                if width == 1 {
                    util::sigmoid(row[0] as f64)
                } else {
                    softmax_positive(row, &self.meta.positive)
                }
            })
            .collect())
    }
}

fn load_tokenizer(model: &str, path: &Path, expected_sha256: &str) -> Result<Tokenizer> {
    let bytes = std::fs::read(path).map_err(|e| {
        BackendError::InvalidModel(format!(
            "model '{model}': cannot read tokenizer {}: {e} \
             (is `model_repository` the same directory Triton serves?)",
            path.display()
        ))
    })?;
    let actual = sha256_hex(&bytes);
    if !actual.eq_ignore_ascii_case(expected_sha256) {
        return Err(BackendError::InvalidModel(format!(
            "model '{model}': {} has sha256 {actual}, but the exported model expects \
             {expected_sha256}; the gateway's model repository is out of sync with Triton's",
            path.display()
        )));
    }
    let mut tok = Tokenizer::from_bytes(&bytes).map_err(|e| {
        BackendError::InvalidModel(format!(
            "model '{model}': invalid tokenizer {}: {e}",
            path.display()
        ))
    })?;
    tok.with_truncation(None)
        .map_err(|e| BackendError::InvalidModel(format!("model '{model}': {e}")))?;
    tok.with_padding(None);
    Ok(tok)
}

/// Lowercase hex sha256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use serde_json::json;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::normalizers::NFKC;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::processors::template::TemplateProcessing;

    use super::*;

    /// Word-level tokenizer: `[PAD]=0 [CLS]=1 [SEP]=2 [UNK]=3`, then `words`.
    pub(crate) fn tiny_tokenizer(dir: &Path, words: &[&str]) -> (PathBuf, String) {
        let mut vocab: HashMap<String, u32> = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32))
            .collect();
        for (i, w) in words.iter().enumerate() {
            vocab.insert(w.to_string(), 4 + i as u32);
        }
        let model = WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_normalizer(Some(NFKC)).unwrap();
        tok.with_pre_tokenizer(Some(Whitespace {}));
        tok.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]")
                .unwrap()
                .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
                .build()
                .unwrap(),
        ));
        let path = dir.join("tokenizer.json");
        tok.save(&path, false).unwrap();
        let sha = sha256_hex(&std::fs::read(&path).unwrap());
        (path, sha)
    }

    fn classifier(window: usize, stride: usize) -> (TextClassifier, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "lfs-text-{}-{window}-{stride}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (_, sha) = tiny_tokenizer(&dir, &["a", "b", "c", "ignore", "this"]);
        let meta = json!({
            "family": "test", "labels": ["SAFE", "INJECTION"], "positive": [1],
            "inputs": ["input_ids", "attention_mask"], "outputs": ["logits"],
            "tokenizer": "tokenizer.json", "tokenizer_sha256": sha,
            "max_tokens": 64, "window": window, "stride": stride, "pad_id": 0
        });
        let info = ModelInfo {
            name: "t",
            metadata: &meta,
            max_batch_size: 8,
            model_dir: &dir,
        };
        (TextClassifier::load(&info).unwrap(), dir)
    }

    #[test]
    fn short_text_is_one_window_with_specials() {
        let (c, _) = classifier(8, 2);
        let w = c.windows("a b ignore", &c.default_window()).unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].ids, vec![1, 4, 5, 7, 2]);
        assert_eq!((w[0].start, w[0].end, w[0].content_tokens), (0, 10, 3));
    }

    #[test]
    fn long_text_windows_overlap_and_cover_everything() {
        // window 6 = CLS + 4 content + SEP; stride 2 -> step 2 content tokens.
        let (c, _) = classifier(6, 2);
        let text = "a b c a b c ignore a"; // 8 tokens
        let w = c.windows(text, &c.default_window()).unwrap();
        assert!(w.iter().all(|w| w.ids.len() <= 6 && w.ids[0] == 1));
        // every content token is in some window, and the trigger survives
        assert!(w.iter().any(|w| w.ids.contains(&7)));
        assert_eq!(w.first().unwrap().start, 0);
        assert_eq!(w.last().unwrap().end, text.chars().count());
        for pair in w.windows(2) {
            assert!(
                pair[1].start < pair[0].end,
                "consecutive windows must overlap"
            );
        }
    }

    #[test]
    fn spans_are_code_points_of_the_original_text() {
        let (c, _) = classifier(4, 1); // CLS + 2 content + SEP
        // "ｉｇｎｏｒｅ" (full-width) NFKC-folds to "ignore"
        let text = "é b ｉｇｎｏｒｅ";
        let w = c.windows(text, &c.default_window()).unwrap();
        let hit = w
            .iter()
            .find(|w| w.ids.contains(&7))
            .expect("folded trigger");
        let span: String = text
            .chars()
            .skip(hit.start)
            .take(hit.end - hit.start)
            .collect();
        assert!(span.ends_with("ｉｇｎｏｒｅ"), "{span:?}");
    }

    #[test]
    fn empty_text_scores_a_specials_only_window() {
        let (c, _) = classifier(8, 2);
        let w = c.windows("", &c.default_window()).unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].ids, vec![1, 2]);
    }

    #[test]
    fn pack_pads_ids_and_masks() {
        let (c, _) = classifier(8, 2);
        let a = c.windows("a", &c.default_window()).unwrap().remove(0);
        let b = c.windows("a b c", &c.default_window()).unwrap().remove(0);
        let t = c.pack(&[&a, &b], 5);
        assert_eq!(t[0].shape, vec![2, 5]);
        assert_eq!(
            t[0].data,
            TensorData::I64(vec![1, 4, 2, 0, 0, 1, 4, 5, 6, 2])
        );
        assert_eq!(
            t[1].data,
            TensorData::I64(vec![1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
        );
    }

    #[test]
    fn tokenizer_hash_mismatch_is_refused() {
        let (_, dir) = classifier(8, 2);
        let meta = json!({
            "labels": ["SAFE", "INJECTION"], "positive": [1],
            "inputs": ["input_ids"], "outputs": ["logits"],
            "tokenizer": "tokenizer.json", "tokenizer_sha256": "00",
            "max_tokens": 64, "window": 8, "stride": 2, "pad_id": 0
        });
        let info = ModelInfo {
            name: "t",
            metadata: &meta,
            max_batch_size: 8,
            model_dir: &dir,
        };
        let err = TextClassifier::load(&info).err().unwrap().to_string();
        assert!(err.contains("out of sync"), "{err}");
    }

    #[test]
    fn scores_check_shape() {
        let (c, _) = classifier(8, 2);
        let out = OutputTensor {
            name: "logits".into(),
            shape: vec![2, 2],
            data: vec![0.0, 0.0, -5.0, 5.0],
        };
        let s = c.scores(std::slice::from_ref(&out), 2).unwrap();
        assert!((s[0] - 0.5).abs() < 1e-12 && s[1] > 0.99);
        assert!(c.scores(&[out], 3).is_err());
    }
}
