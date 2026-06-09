//! Throughput benchmark for the translate path. Mirrors the txt "reflow"
//! caller: merge hard-wrapped lines into paragraphs, then chunk paragraphs the
//! way the epub/odt/txt paths do (N units per progress tick) and flatten each
//! chunk's sentences into one slimt call — exactly what `translate_split` does.
//! Compares chunked calls against a single call over the whole document.
//!
//! Usage: bench <model_dir> <pair> <input_file> [WORKERS]

use std::path::PathBuf;
use std::time::Instant;

use slimt_sys::{BlockingService, ModelArch, TranslationModel};

/// Group consecutive non-empty lines into paragraphs, collapsing whitespace.
/// Mirrors translator-rs `txt::split_paragraphs` + `join_paragraph`.
fn split_paragraphs(text: &str) -> Vec<String> {
    let mut paragraphs = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let flush = |cur: &mut Vec<&str>, out: &mut Vec<String>| {
        if !cur.is_empty() {
            out.push(cur.join(" ").split_whitespace().collect::<Vec<_>>().join(" "));
            cur.clear();
        }
    };
    for line in text.split('\n') {
        if line.trim().is_empty() {
            flush(&mut current, &mut paragraphs);
        } else {
            current.push(line);
        }
    }
    flush(&mut current, &mut paragraphs);
    paragraphs
}

/// Sentence split mirroring translator-rs `sentence_split::split_sentences`:
/// break after a run of [.!?] when followed by whitespace then an uppercase
/// letter. Returns slices borrowed from `text`.
fn split_sentences(text: &str) -> Vec<&str> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut last_end = 0usize;
    let mut i = 0;
    while i < n {
        let c = chars[i].1;
        if c == '.' || c == '!' || c == '?' {
            let mut j = i;
            while j < n && matches!(chars[j].1, '.' | '!' | '?') {
                j += 1;
            }
            let punct_end = if j < n { chars[j].0 } else { text.len() };
            let mut k = j;
            let mut saw_ws = false;
            while k < n && chars[k].1.is_whitespace() {
                k += 1;
                saw_ws = true;
            }
            if saw_ws && k < n && chars[k].1.is_uppercase() {
                let piece = &text[last_end..punct_end];
                if !piece.trim().is_empty() {
                    out.push(piece);
                }
                last_end = chars[k].0;
                i = k;
                continue;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    if last_end < text.len() {
        let tail = &text[last_end..];
        if !tail.trim().is_empty() {
            out.push(tail);
        }
    }
    if out.is_empty() {
        out.push(text);
    }
    out
}

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
    let paragraphs = split_paragraphs(&raw);
    // Flat sentence list per paragraph, kept alongside paragraphs (which own
    // the backing strings) so chunked runs can slice without re-splitting.
    let sentences_per_para: Vec<Vec<&str>> =
        paragraphs.iter().map(|p| split_sentences(p)).collect();
    let total_sentences: usize = sentences_per_para.iter().map(Vec::len).sum();
    let total_words: usize = sentences_per_para
        .iter()
        .flatten()
        .map(|s| s.split_whitespace().count())
        .sum();

    eprintln!(
        "input: {} ({} paragraphs, {} sentences, {} words)",
        input_path,
        paragraphs.len(),
        total_sentences,
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

    // One slimt call per chunk of `chunk` paragraphs; flatten that chunk's
    // sentences (what translate_split feeds slimt). Returns wall seconds.
    let run = |chunk: usize| -> (f64, f64) {
        let mut produced = 0usize;
        let mut max_words_in_call = 0usize;
        let t0 = Instant::now();
        for paras in sentences_per_para.chunks(chunk) {
            let call: Vec<&str> = paras.iter().flatten().copied().collect();
            let w: usize = call.iter().map(|s| s.split_whitespace().count()).sum();
            max_words_in_call = max_words_in_call.max(w);
            let outs = service.translate(&model, &call);
            produced += outs.len();
        }
        assert_eq!(produced, total_sentences);
        (t0.elapsed().as_secs_f64(), max_words_in_call as f64)
    };

    // Warm up.
    let _ = run(paragraphs.len());

    // Scaling probe: is one big translate() linear in sentence count, and does
    // the per-sentence rate decelerate over a single run? Cycles the corpus to
    // larger N (cache is off, so duplicates are re-translated).
    if args.get(5).map(String::as_str) == Some("scale") {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let base: Vec<&str> = sentences_per_para.iter().flatten().copied().collect();
        eprintln!(
            "\nscaling (single dump, cache off):\n{:>7} {:>8} {:>9} {:>8}   per-decile seconds",
            "N", "wall(s)", "words/s", "sec/1k"
        );
        for &mult in &[1usize, 2, 4, 8] {
            let corpus: Vec<&str> = (0..mult).flat_map(|_| base.iter().copied()).collect();
            let n = corpus.len();
            let words: usize = corpus.iter().map(|s| s.split_whitespace().count()).sum();
            // Deciles are over the byte-weighted progress fraction (the callback
            // now delivers each completed input's byte length), so flat
            // per-decile times mean the bar climbs at a constant rate.
            let weight_total: usize = corpus.iter().map(|s| s.len()).sum();
            let cancel = AtomicBool::new(false);
            let done = AtomicUsize::new(0);
            let next = AtomicUsize::new(1);
            let marks = Mutex::new(Vec::<f64>::new());
            let t0 = Instant::now();
            let outs = service
                .translate_with_progress(&model, &corpus, &cancel, |d| {
                    let c = done.fetch_add(d, Ordering::Relaxed) + d;
                    let dec = c * 10 / weight_total;
                    if dec + 1 > next.fetch_max(dec + 1, Ordering::Relaxed) {
                        marks.lock().unwrap().push(t0.elapsed().as_secs_f64());
                    }
                })
                .expect("not cancelled");
            let secs = t0.elapsed().as_secs_f64();
            assert_eq!(outs.len(), n);
            let m = marks.lock().unwrap();
            let mut durs = Vec::new();
            let mut prev = 0.0;
            for &t in m.iter() {
                durs.push(format!("{:.2}", t - prev));
                prev = t;
            }
            eprintln!(
                "{:>7} {:>8.2} {:>9.0} {:>8.3}   [{}]",
                n,
                secs,
                words as f64 / secs,
                secs / (n as f64 / 1000.0),
                durs.join(" ")
            );
        }
        return;
    }

    let all = paragraphs.len();
    let configs = [all, 8, 16, 32, 64, 128, all];

    eprintln!(
        "\n{:>10}  {:>9}  {:>11}  {:>9}  {:>8}",
        "paras/call", "wall(s)", "words/s", "maxW/call", "vs-1call"
    );
    let mut baseline = 0.0f64;
    for (i, &chunk) in configs.iter().enumerate() {
        let (secs, maxw) = run(chunk);
        if i == 0 {
            baseline = secs;
        }
        let wps = total_words as f64 / secs;
        let label = if chunk == all {
            "all".to_string()
        } else {
            chunk.to_string()
        };
        eprintln!(
            "{:>10}  {:>9.3}  {:>11.0}  {:>9.0}  {:>7.2}x",
            label,
            secs,
            wps,
            maxw,
            secs / baseline
        );
    }

    // Option (2): single call over every sentence with a non-blocking progress
    // callback. Verifies one event fires per input and that the callback adds
    // no measurable cost versus the no-callback "all" baseline above.
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let all_sentences: Vec<&str> = sentences_per_para.iter().flatten().copied().collect();
    let counter = AtomicUsize::new(0);
    let no_cancel = AtomicBool::new(false);
    let t0 = Instant::now();
    let outs = service
        .translate_with_progress(&model, &all_sentences, &no_cancel, |n| {
            counter.fetch_add(n, Ordering::Relaxed);
        })
        .expect("not cancelled");
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(outs.len(), all_sentences.len());
    eprintln!(
        "\nprogress single-call: {:.3}s  {:.0} words/s  events={}/{}  (vs no-cb all = {:.2}x)",
        secs,
        total_words as f64 / secs,
        counter.load(Ordering::Relaxed),
        total_sentences,
        secs / baseline
    );

    // Cancellation: flip the flag from another thread shortly after the call
    // starts and confirm it returns fast (within ~one batch per worker) having
    // translated only a fraction of the corpus.
    let done = AtomicUsize::new(0);
    let cancel = AtomicBool::new(false);
    let t0 = Instant::now();
    let result = std::thread::scope(|s| {
        s.spawn(|| {
            // Busy-wait until ~150 sentences are in, then cancel.
            while done.load(Ordering::Relaxed) < 150 && t0.elapsed().as_secs_f64() < 5.0 {
                std::hint::spin_loop();
            }
            cancel.store(true, Ordering::Relaxed);
        });
        service.translate_with_progress(&model, &all_sentences, &cancel, |n| {
            done.fetch_add(n, Ordering::Relaxed);
        })
    });
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "cancel test: returned in {:.3}s  result={}  translated {}/{} before stop",
        secs,
        if result.is_none() { "None (cancelled)" } else { "Some" },
        done.load(Ordering::Relaxed),
        total_sentences,
    );
}
