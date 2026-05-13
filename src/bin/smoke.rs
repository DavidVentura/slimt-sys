use std::path::PathBuf;

use slimt_sys::{BlockingService, ModelArch, TranslationModel};

fn main() {
    let base: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/david/.local/share/dev.davidv.translator/bin".to_string())
        .into();
    let pair = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "esen".to_string());
    let text = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "Hola mundo".to_string());

    let model_path = base.join(format!("model.{pair}.intgemm.alphas.bin"));
    let shared_vocab = base.join(format!("vocab.{pair}.spm"));
    let src_vocab = base.join(format!("srcvocab.{pair}.spm"));
    let tgt_vocab = base.join(format!("trgvocab.{pair}.spm"));
    let lex_path = base.join(format!("lex.50.50.{pair}.s2t.bin"));

    println!("model={}", model_path.display());
    println!("lex={}", lex_path.display());

    let arch = ModelArch::default();
    let model = if shared_vocab.exists() {
        println!("vocab={}", shared_vocab.display());
        TranslationModel::with_arch(&model_path, &shared_vocab, &lex_path, None, arch)
    } else {
        println!("srcvocab={}", src_vocab.display());
        println!("trgvocab={}", tgt_vocab.display());
        TranslationModel::with_arch_and_target_vocab(
            &model_path,
            &src_vocab,
            &lex_path,
            None,
            arch,
            Some(&tgt_vocab),
        )
    }
    .expect("load model");
    let service = BlockingService::with_workers(4, 0);

    let extra: Vec<String> = std::env::args().skip(4).collect();
    let mut inputs: Vec<&str> = vec![text.as_str()];
    inputs.extend(extra.iter().map(|s| s.as_str()));
    let outs = service.translate(&model, &inputs);
    for (i, t) in outs.iter().enumerate() {
        println!("[{i}] {t}");
    }
}
