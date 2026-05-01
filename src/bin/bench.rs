//! Throughput benchmark that exercises the translate path away from the
//! per-page bottleneck of the PDF pipeline. Reads a file (one source per
//! line), translates the whole batch in a single `translate` call, prints
//! wall time + words/second.
//!
//! Usage: bench <model_dir> <pair> <input_file> [WORKERS]
//!   bench /path/to/bin enes /path/to/moby-dick-2k-lines.txt

use std::path::PathBuf;
use std::time::Instant;

use slimt_sys::{BlockingService, ModelArch, TranslationModel};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base: PathBuf = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/home/david/.local/share/dev.davidv.translator/bin".to_string())
        .into();
    let pair = args.get(2).cloned().unwrap_or_else(|| "enes".to_string());
    let input_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "/home/david/git/slimt/build/moby-dick-2k-lines.txt".to_string());
    let workers: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);

    let model_path = base.join(format!("model.{pair}.intgemm.alphas.bin"));
    let vocab_path = base.join(format!("vocab.{pair}.spm"));
    let lex_path = base.join(format!("lex.50.50.{pair}.s2t.bin"));

    let raw = std::fs::read_to_string(&input_path).expect("read input");
    let lines: Vec<String> = raw
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let total_words: usize = lines.iter().map(|l| l.split_whitespace().count()).sum();

    eprintln!(
        "input: {} ({} non-empty lines, {} words)",
        input_path,
        line_refs.len(),
        total_words
    );
    eprintln!("model: {} workers={}", pair, workers);

    let model = TranslationModel::with_arch(
        &model_path,
        &vocab_path,
        &lex_path,
        None,
        ModelArch::default(),
    )
    .expect("load model");
    let service = BlockingService::with_workers(workers, 0);

    // Warm up: translate the first line once so JIT-y caches / first-touch
    // page faults don't pollute the timed run.
    let _ = service.translate(&model, &[line_refs[0]], false);

    let t0 = Instant::now();
    let outs = service.translate(&model, &line_refs, false);
    let dt = t0.elapsed();
    assert_eq!(outs.len(), line_refs.len());

    let secs = dt.as_secs_f64();
    let wps = total_words as f64 / secs;
    eprintln!(
        "wall: {:.3}s  ({:.1} lines/s, {:.0} words/s)",
        secs,
        line_refs.len() as f64 / secs,
        wps
    );
    println!("{secs:.6}");
}
