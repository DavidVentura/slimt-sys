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
//!
//! Both stayed broken under several numerical "fixes" (`-ffp-contract=fast`,
//! WITH_BLAS, LN vectorize pragmas, beam>1) that masked the real cause for
//! some inputs but not others. They guard against regressions in either the
//! decoder positional offset or in the broader numerical contract with
//! marian-bergamot.
//!
//! The test loads models from `SLIMT_TEST_MODELS_DIR/<pair>/`; each pair
//! directory must contain `model.<pair>.intgemm.alphas.bin`,
//! `vocab.<pair>.spm`, and `lex.50.50.<pair>.s2t.bin`. If the env var is
//! unset, the test no-ops with a printed skip note (model files are large
//! and not part of this repo).

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
        let vocab_path = pair_dir.join(format!("vocab.{pair}.spm"));
        let lex_path = pair_dir.join(format!("lex.50.50.{pair}.s2t.bin"));
        if !model_path.exists() {
            eprintln!("skip pair={pair}: missing {}", model_path.display());
            continue;
        }

        let model = TranslationModel::with_arch(
            &model_path,
            &vocab_path,
            &lex_path,
            None,
            ModelArch::default(),
        )
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
