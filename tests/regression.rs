//! Regression coverage for translation correctness bugs.
//!
//! These cases come from real production regressions documented in
//! translator-rs/bug.md. Notably:
//!
//! - `bonjour, tre` (fr→en) once produced `Hello, Hello, Hello` because the
//!   decoder threaded `start=0` into `transform_embedding` for every step
//!   instead of incrementing the absolute decoder position.
//! - `hello how are` (en→es) produced `Hola, ¿hola cómo son` from the same
//!   bug, manifesting as a substantively wrong translation rather than a
//!   loop.
//! - `hello ho` (en→ja) produced `こんにちはhoforefulforeforefore` because
//!   two-vocab bergamot models ship no calibrated activation alpha for the
//!   output projection — Io.cc synthesizes one from the embedding's *weight*
//!   multiplier, ~10× too large, which saturates every activation to ±127
//!   in the int8 GEMM. Modules.cc clamps `a_quant` to a non-saturating
//!   value when the static alpha is too aggressive; bergamot itself runs
//!   full-vocab on this model and gets the same correct outputs.
//!
//! All stayed broken under several numerical "fixes" (`-ffp-contract=fast`,
//! layer-norm vectorize pragmas, beam>1) that masked the real cause for some
//! inputs but not others. They guard against regressions in either the
//! decoder positional offset or in the broader numerical contract with
//! marian-bergamot.
//!
//! The test loads models from `SLIMT_TEST_MODELS_DIR/<pair>/`; each pair
//! directory must contain `model.<pair>.intgemm.alphas.bin` and
//! `lex.50.50.<pair>.s2t.bin`, plus either `vocab.<pair>.spm` (shared-vocab
//! pairs) or `srcvocab.<pair>.spm` + `trgvocab.<pair>.spm` (two-vocab
//! pairs like en-ja, en-ko). If the env var is unset, the test no-ops
//! with a printed skip note (model files are large and not part of this
//! repo).

use slimt_sys::{BlockingService, ModelArch, TranslationModel};
use std::collections::BTreeMap;
use std::path::PathBuf;

struct Case {
    pair: &'static str,
    input: &'static str,
    expected: &'static str,
}

const CASES: &[Case] = &[
    // fr→en: the original looping case + a few siblings that broke under the
    // same decoder-position bug
    Case {
        pair: "fren",
        input: "bonjour, tre",
        expected: "Hello, tre",
    },
    Case {
        pair: "fren",
        input: "bonjour, tres",
        expected: "Hello, very",
    },
    Case {
        pair: "fren",
        input: "Bonjour, comment ça va?",
        expected: "Hello, how are you?",
    },
    Case {
        pair: "fren",
        input: "Comment allez-vous?",
        expected: "How are you?",
    },
    Case {
        pair: "fren",
        input: "Je voudrais un café",
        expected: "I'd like a coffee",
    },
    Case {
        pair: "fren",
        input: "Bonjour",
        expected: "Hello",
    },
    // en→es: substitute-not-loop manifestation
    Case {
        pair: "enes",
        input: "hello how are",
        expected: "Hola, ¿cómo están",
    },
    Case {
        pair: "enes",
        input: "hello how are you",
        expected: "Hola, ¿cómo estás",
    },
    Case {
        pair: "enes",
        input: "hello",
        expected: "Hola",
    },
    Case {
        pair: "enes",
        input: "Hello, how are you?",
        expected: "Hola, ¿cómo estás?",
    },
    Case {
        pair: "enes",
        input: "I would like a coffee",
        expected: "Me gustaría un café",
    },
    Case {
        pair: "enes",
        input: "good morning",
        expected: "Buenos días",
    },
    // en→ja: two-vocab model, no calibrated activation alpha for the output
    // projection. "hello ho" once produced "こんにちはhoforefulforeforefore"
    // because the synthesized alpha saturated the int8 GEMM. Outputs match
    // bergamot's marian-decoder run on the same model.
    Case {
        pair: "enja",
        input: "hello",
        expected: "こんにちは",
    },
    Case {
        pair: "enja",
        input: "hello ho",
        expected: "こんにちは",
    },
    Case {
        pair: "enja",
        input: "Hello wo",
        expected: "こんにちは",
    },
    // "hello wo" is intentionally omitted: per-request shortlists merged
    // at the batch level make its output depend on batch composition
    // ("こんにちは" alone in some batches, "こんにちは、わかりました" alone).
    // Both are reasonable; the bug we're guarding against is the
    // garbage-suffix failure mode on "hello ho", not this divergence.
    Case {
        pair: "enja",
        input: "hello how are you",
        expected: "こんにちは、お元気ですか",
    },
    Case {
        pair: "enja",
        input: "good morning",
        expected: "おはようございます",
    },
];

fn models_dir() -> Option<PathBuf> {
    std::env::var("SLIMT_TEST_MODELS_DIR")
        .ok()
        .map(PathBuf::from)
}

#[test]
fn translation_regression_table() {
    let Some(base) = models_dir() else {
        eprintln!(
            "SLIMT_TEST_MODELS_DIR not set; skipping regression table.\n\
             Set it to a directory containing fren/ and enes/ subdirs with \
             model.<pair>.intgemm.alphas.bin, vocab.<pair>.spm and \
             lex.50.50.<pair>.s2t.bin to run."
        );
        return;
    };

    let service = BlockingService::with_workers(4, 0);

    let mut by_pair: BTreeMap<&str, Vec<&Case>> = BTreeMap::new();
    for c in CASES {
        by_pair.entry(c.pair).or_default().push(c);
    }

    let mut failures: Vec<String> = Vec::new();
    for (pair, cases) in by_pair {
        let pair_dir = base.join(pair);
        let model_path = pair_dir.join(format!("model.{pair}.intgemm.alphas.bin"));
        let lex_path = pair_dir.join(format!("lex.50.50.{pair}.s2t.bin"));
        if !model_path.exists() {
            eprintln!("skip pair={pair}: missing {}", model_path.display());
            continue;
        }

        // Two-vocab pairs ship a `srcvocab` + `trgvocab` pair instead of a
        // single shared `vocab.<pair>.spm`. Detect by file presence.
        let shared_vocab = pair_dir.join(format!("vocab.{pair}.spm"));
        let src_vocab = pair_dir.join(format!("srcvocab.{pair}.spm"));
        let tgt_vocab = pair_dir.join(format!("trgvocab.{pair}.spm"));
        let model = if shared_vocab.exists() {
            TranslationModel::with_arch(
                &model_path,
                &shared_vocab,
                &lex_path,
                None,
                ModelArch::default(),
            )
        } else if src_vocab.exists() && tgt_vocab.exists() {
            TranslationModel::with_arch_and_target_vocab(
                &model_path,
                &src_vocab,
                &lex_path,
                None,
                ModelArch::default(),
                Some(&tgt_vocab),
            )
        } else {
            eprintln!(
                "skip pair={pair}: missing vocab (looked for {} and srcvocab/trgvocab)",
                shared_vocab.display()
            );
            continue;
        }
        .unwrap_or_else(|e| panic!("load {pair}: {e}"));

        let inputs: Vec<&str> = cases.iter().map(|c| c.input).collect();
        let outs = service.translate(&model, &inputs);

        for (case, got) in cases.iter().zip(outs.iter()) {
            if got.trim() != case.expected {
                failures.push(format!(
                    "  pair={pair} in={:?}\n      got={:?}\n      exp={:?}",
                    case.input,
                    got.trim(),
                    case.expected,
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "translation regressions ({} of {}):\n{}",
        failures.len(),
        CASES.len(),
        failures.join("\n")
    );
}
