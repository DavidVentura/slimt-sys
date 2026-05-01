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
    let vocab_path = base.join(format!("vocab.{pair}.spm"));
    let lex_path = base.join(format!("lex.50.50.{pair}.s2t.bin"));

    println!("model={}", model_path.display());
    println!("vocab={}", vocab_path.display());
    println!("lex={}", lex_path.display());

    let arch = ModelArch::default();
    let model = TranslationModel::with_arch(&model_path, &vocab_path, &lex_path, None, arch)
        .expect("load model");
    let service = BlockingService::with_workers(4, 0);

    let inputs = vec![text.as_str()];
    let outs = service.translate(&model, &inputs, false);
    for (i, t) in outs.iter().enumerate() {
        println!("[{i}] {t}");
    }
}
