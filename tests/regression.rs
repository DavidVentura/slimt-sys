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
    // en→es: target-length cap. The cap is derived per row from the row's
    // own (unpadded) source length plus an additive slack; with the purely
    // multiplicative cap (1.5 × 16 source subwords = 24) this sentence
    // truncated mid-word ("…en los Arsaci"), and before that the cap came
    // from the batch's padded length, so whether it truncated depended on
    // which sentences it was batched with. The guarded property is the
    // complete "Arsacidas." — the word for "Bower" is a hallucination at
    // any shortlist size ("Irresponsador" under full-vocab decode,
    // "Irresponsante" under the topped-up shortlist) and not what this
    // case is about.
    Case {
        pair: "enes",
        input: "CHAPTER 102. A Bower in the Arsacides.",
        expected: "CAPÍTULO 102. Un Irresponsante en los Arsacidas.",
    },
    // en→es: shortlist-coverage table. Each case represents an input class
    // that once starved the shortlist into fluent garbage (ALLCAPS headers,
    // bare numbers/prices, curly apostrophes, proper nouns, leading dashes,
    // all-lowercase prose). Guards the candidate top-up: with the bare
    // 50-frequent + lex-aligned set these produced outputs like "CAPIi 93"
    // or "- " prefixes; with the full-vocab floor some hallucinated extras
    // ("Categoría: 1987"). Expectations are x86 outputs of the current
    // model files (393bc6-era models differ wildly — keep device test files
    // md5-synced with the bucket before comparing).
    Case {
        pair: "enes",
        input: "CHAPTER 93. The Castaway.",
        expected: "CAPÍTULO 93. El Náufrago.",
    },
    Case {
        pair: "enes",
        input: "THE END",
        expected: "EL FIN",
    },
    Case {
        pair: "enes",
        input: "WARNING: DO NOT ENTER",
        expected: "ADVERTENCIA: NO ENTRE",
    },
    Case {
        pair: "enes",
        input: "IMPORTANT NOTICE",
        expected: "AVISO IMPORTANTE",
    },
    Case {
        pair: "enes",
        input: "1987",
        expected: "1987",
    },
    Case {
        pair: "enes",
        input: "$4.99",
        expected: "$4.99",
    },
    Case {
        pair: "enes",
        input: "Whale’s teeth are large",
        expected: "Los dientes de ballena son grandes",
    },
    Case {
        pair: "enes",
        input: "Queequeg was a native of Rokovoko.",
        expected: "Queequeg era nativo de Rokovoko.",
    },
    Case {
        pair: "enes",
        input: "- item one",
        expected: "- artículo uno",
    },
    Case {
        pair: "enes",
        input: "the quick brown fox jumps over the lazy dog",
        expected: "El zorro marrón rápido salta sobre el perro perezoso",
    },
    // fr→en: knife-edge numerics canary. The "low"/"bas" decision at decoder
    // step 11 sits 0.84 logits apart in float32; int8 quantization eats
    // nearly all of that margin and the backend's float-epilogue noise
    // (~0.2 logits) decides the rest. marian+intgemm lands on "low tide",
    // marian+ruy on this exact output byte-for-byte. A flip here means the
    // numerical contract with marian-ruy moved — not necessarily a bug, but
    // always worth understanding before accepting.
    Case {
        pair: "fren",
        input: "ATTENTION: la boite n'est accessible qu'à \"marrée basse\"",
        expected: "ATTENTION: the box is only accessible to \"bass tie\"",
    },
    // en→es: plain prose table from the 2026-06 slimt-vs-marian numerics
    // audit (fused layernorm/softmax/FFN, weight concat, shortlist top-up).
    // At audit time slimt matched marian-ruy on 57/60 such sentences and was
    // equal-or-closer to float32 on the rest. Pins current outputs so future
    // optimizations that drift the numerics get flagged.
    Case {
        pair: "enes",
        input: "The meeting starts at nine.",
        expected: "La reunión comienza a las nueve.",
    },
    Case {
        pair: "enes",
        input: "Please close the door.",
        expected: "Por favor, cierre la puerta.",
    },
    Case {
        pair: "enes",
        input: "Where is the train station?",
        expected: "¿Dónde está la estación de tren?",
    },
    Case {
        pair: "enes",
        input: "I forgot my keys again.",
        expected: "Volví a olvidar mis llaves.",
    },
    Case {
        pair: "enes",
        input: "The weather looks terrible today.",
        expected: "El tiempo se ve terrible hoy.",
    },
    Case {
        pair: "enes",
        input: "She bought three red apples.",
        expected: "Ella compró tres manzanas rojas.",
    },
    Case {
        pair: "enes",
        input: "Turn left at the bridge.",
        expected: "Gire a la izquierda en el puente.",
    },
    Case {
        pair: "enes",
        input: "The battery is almost dead.",
        expected: "La batería está casi muerta.",
    },
    Case {
        pair: "enes",
        input: "The museum is closed on Mondays, but the garden stays open all year.",
        expected: "El museo está cerrado los lunes, pero el jardín permanece abierto todo el año.",
    },
    Case {
        pair: "enes",
        input: "If you arrive before noon, ask for the manager at the front desk.",
        expected: "Si llegas antes del mediodía, pregunta por el gerente en la recepción.",
    },
    Case {
        pair: "enes",
        input: "The package was delivered to the wrong address for the second time this month.",
        expected: "El paquete fue entregado a la dirección incorrecta por segunda vez este mes.",
    },
    Case {
        pair: "enes",
        input: "Our flight was delayed by two hours because of a storm over the mountains.",
        expected: "Nuestro vuelo se retrasó dos horas debido a una tormenta sobre las montañas.",
    },
    Case {
        pair: "enes",
        input: "The new software update fixes several bugs but introduces a slower startup time.",
        expected: "La nueva actualización de software corrija varios errores, pero introduce un tiempo de inicio más lento.",
    },
    Case {
        pair: "enes",
        input: "He promised to send the report by Friday, yet nobody has received anything.",
        expected: "Prometió enviar el informe el viernes, pero nadie ha recibido nada.",
    },
    Case {
        pair: "enes",
        input: "The recipe calls for two cups of flour, a pinch of salt, and three eggs.",
        expected: "La receta requiere dos tazas de harina, una pizca de sal y tres huevos.",
    },
    Case {
        pair: "enes",
        input: "Visitors must sign the register before entering the construction site.",
        expected: "Los visitantes deben firmar el registro antes de entrar en el sitio de construcción.",
    },
    Case {
        pair: "enes",
        input: "The river floods almost every spring, so the village built a higher bridge.",
        expected: "El río se inunda casi todas las primaveras, por lo que el pueblo construyó un puente más alto.",
    },
    Case {
        pair: "enes",
        input: "According to the manual, the red light means the filter needs to be replaced.",
        expected: "Según el manual, la luz roja significa que el filtro necesita ser reemplazado.",
    },
    Case {
        pair: "enes",
        input: "The committee will announce its final decision after next week's meeting.",
        expected: "El comité anunciará su decisión final después de la reunión de la próxima semana.",
    },
    Case {
        pair: "enes",
        input: "WARNING: the gate is locked after sunset, use the side entrance instead.",
        expected: "ADVERTENCIA: la puerta está cerrada después de la puesta del sol, utilice la entrada lateral en su lugar.",
    },
    // en→es: rare-token shortlist coverage. marian-ruy emits the nonword
    // "Prube" here (degenerate shortlist for sentence-initial "Try"); the
    // candidate top-up is what rescues the valid "Pruebe".
    Case {
        pair: "enes",
        input: "Try the bouillabaisse at the quayside bistro in Marseille.",
        expected: "Pruebe el bouillabaisse en el bistró de muelles en Marsella.",
    },
    Case {
        pair: "enes",
        input: "Dr. Brzezinski prescribed amoxicillin for the laryngitis.",
        expected: "El Dr. Brzezinski prescrito amoxicilina para la laringitis.",
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
