//! Helpers shared by backends (output lookup, shape checks, probability
//! math). Nothing here is model-specific.

use crate::{BackendError, OutputTensor, Result};

/// Find an output by name, falling back to its position in the response.
pub fn output<'a>(
    outputs: &'a [OutputTensor],
    name: &str,
    index: usize,
) -> Result<&'a OutputTensor> {
    outputs
        .iter()
        .find(|o| o.name == name)
        .or_else(|| outputs.get(index))
        .ok_or_else(|| BackendError::BadOutput(format!("triton returned no '{name}' output")))
}

/// Ensure an output carries at least `n` values.
pub fn require_len(t: &OutputTensor, n: usize) -> Result<()> {
    if t.data.len() < n {
        return Err(BackendError::BadOutput(format!(
            "'{}' output too small: {} values, expected {n}",
            t.name,
            t.data.len()
        )));
    }
    Ok(())
}

/// Numerically stable logistic sigmoid.
pub fn sigmoid(x: f64) -> f64 {
    let x = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-x).exp())
}

/// Softmax over `logits` (in f64, so the tails do not saturate), summed over
/// the `positive` positions.
pub fn softmax_positive(logits: &[f32], positive: &[usize]) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)) as f64;
    let exps: Vec<f64> = logits.iter().map(|&x| (x as f64 - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    positive.iter().filter_map(|&i| exps.get(i)).sum::<f64>() / total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_binary_matches_sigmoid_of_difference() {
        let p = softmax_positive(&[-1.0, 2.0], &[1]);
        assert!((p - sigmoid(3.0)).abs() < 1e-12);
    }

    #[test]
    fn softmax_sums_multiple_positive_labels() {
        // BENIGN / INJECTION / JAILBREAK style heads
        let p = softmax_positive(&[0.0, 0.0, 0.0], &[1, 2]);
        assert!((p - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let p = softmax_positive(&[1000.0, -1000.0], &[1]);
        assert!(p.is_finite() && p < 1e-300_f64.max(0.0) + 1e-12);
    }
}
