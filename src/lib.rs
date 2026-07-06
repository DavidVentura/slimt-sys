use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TokenAlignment {
    pub src_begin: usize,
    pub src_end: usize,
    pub tgt_begin: usize,
    pub tgt_end: usize,
}

#[repr(C)]
struct CTranslation {
    source: *mut c_char,
    target: *mut c_char,
    alignments: *mut TokenAlignment,
    alignment_count: usize,
    alternatives: *mut CWordAlternatives,
    alternatives_count: usize,
}

pub struct TranslationWithAlignment {
    pub source: String,
    pub target: String,
    pub alignments: Vec<TokenAlignment>,
}

/// One candidate the model would accept in place of a chosen word, with its
/// softmax probability at that position.
#[derive(Debug, Clone)]
pub struct Alternative {
    pub text: String,
    pub prob: f32,
}

/// A target word the model was not fully confident about, with the alternatives
/// worth offering. `tgt_begin`/`tgt_end` are char offsets into the target text.
#[derive(Debug, Clone)]
pub struct WordAlternatives {
    pub tgt_begin: usize,
    pub tgt_end: usize,
    pub confidence: f32,
    pub options: Vec<Alternative>,
}

#[derive(Debug, Clone)]
pub struct TranslationWithAlternatives {
    pub source: String,
    pub target: String,
    pub alignments: Vec<TokenAlignment>,
    pub alternatives: Vec<WordAlternatives>,
}

/// Knobs for [`BlockingService::translate_with_alternatives`]. `max_alternatives`
/// caps how many options each word carries; `gate` is the minimum probability an
/// alternative needs to be surfaced.
#[derive(Debug, Clone, Copy)]
pub struct AlternativesOptions {
    pub max_alternatives: usize,
    pub gate: f32,
}

impl Default for AlternativesOptions {
    fn default() -> Self {
        Self {
            max_alternatives: 3,
            gate: 0.03,
        }
    }
}

#[repr(C)]
struct CAlternative {
    text: *mut c_char,
    prob: f32,
}

#[repr(C)]
struct CWordAlternatives {
    tgt_begin: usize,
    tgt_end: usize,
    confidence: f32,
    options: *mut CAlternative,
    option_count: usize,
}

#[repr(C)]
struct CTranslateOptions {
    want_alignment: bool,
    want_alternatives: bool,
    max_alternatives: usize,
    gate: f32,
}

/// Fired once per input sentence as it finishes, from a slimt worker thread
/// (concurrently across inputs). `completed` is how many just finished — always
/// 1 at the current per-Request granularity.
type ProgressCb = unsafe extern "C" fn(user_data: *mut c_void, completed: usize);

unsafe extern "C" {
    fn slimt_service_new(workers: usize, cache_size: usize) -> *mut c_void;
    fn slimt_service_delete(service_ptr: *mut c_void);

    fn slimt_model_new(
        model_path: *const c_char,
        vocabulary_path: *const c_char,
        shortlist_path: *const c_char,
        ssplit_path: *const c_char,
        encoder_layers: usize,
        decoder_layers: usize,
        feed_forward_depth: usize,
        num_heads: usize,
        target_vocabulary_path: *const c_char,
    ) -> *mut c_void;
    fn slimt_model_delete(model_ptr: *mut c_void);

    fn slimt_service_translate(
        service_ptr: *mut c_void,
        model_ptr: *mut c_void,
        inputs: *const *const c_char,
        count: usize,
        want_alignment: bool,
        on_progress: Option<ProgressCb>,
        user_data: *mut c_void,
        cancel_flag: *const c_void,
    ) -> *mut CTranslation;

    fn slimt_service_translate_ex(
        service_ptr: *mut c_void,
        model_ptr: *mut c_void,
        inputs: *const *const c_char,
        count: usize,
        opts: *const CTranslateOptions,
        on_progress: Option<ProgressCb>,
        user_data: *mut c_void,
        cancel_flag: *const c_void,
    ) -> *mut CTranslation;

    fn slimt_service_steer(
        service_ptr: *mut c_void,
        model_ptr: *mut c_void,
        sources: *const *const c_char,
        prefixes: *const *const c_char,
        count: usize,
        opts: *const CTranslateOptions,
        on_progress: Option<ProgressCb>,
        user_data: *mut c_void,
        cancel_flag: *const c_void,
    ) -> *mut CTranslation;

    fn slimt_service_pivot(
        service_ptr: *mut c_void,
        first_model_ptr: *mut c_void,
        second_model_ptr: *mut c_void,
        inputs: *const *const c_char,
        count: usize,
        want_alignment: bool,
        on_progress: Option<ProgressCb>,
        user_data: *mut c_void,
        cancel_flag: *const c_void,
    ) -> *mut CTranslation;

    fn slimt_translations_delete(results: *mut CTranslation, count: usize);

    fn slimt_last_error() -> *const c_char;
}

fn last_error() -> String {
    let raw = unsafe { slimt_last_error() };
    if raw.is_null() {
        return "unknown error".to_string();
    }
    unsafe { CStr::from_ptr(raw) }
        .to_string_lossy()
        .into_owned()
}

fn path_to_cstring(path: &Path) -> CString {
    CString::new(path.to_string_lossy().into_owned()).expect("path must not contain nul bytes")
}

/// Build a lookup table: byte_offset -> char_offset.
fn byte_to_char_offsets(s: &str) -> Vec<usize> {
    let mut table = vec![0usize; s.len() + 1];
    let mut char_idx = 0;
    for (byte_idx, _) in s.char_indices() {
        table[byte_idx] = char_idx;
        char_idx += 1;
    }
    table[s.len()] = char_idx;
    let mut last = char_idx;
    for i in (0..s.len()).rev() {
        if table[i] == 0 && i > 0 {
            table[i] = last;
        } else {
            last = table[i];
        }
    }
    table
}

/// Architecture knobs for a slimt Model. Pass 0 for any field to let the
/// wrapper auto-detect from the binary parameter index (encoder_l*/decoder_l*),
/// or fall back to slimt's tiny preset for fields it cannot detect.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelArch {
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub feed_forward_depth: usize,
    pub num_heads: usize,
}

pub struct TranslationModel {
    ptr: *mut c_void,
}

unsafe impl Send for TranslationModel {}
unsafe impl Sync for TranslationModel {}

impl Drop for TranslationModel {
    fn drop(&mut self) {
        unsafe { slimt_model_delete(self.ptr) }
    }
}

impl TranslationModel {
    pub fn new(
        model: &Path,
        vocabulary: &Path,
        shortlist: &Path,
        ssplit: Option<&Path>,
    ) -> Result<Self, String> {
        Self::with_arch(model, vocabulary, shortlist, ssplit, ModelArch::default())
    }

    pub fn with_arch(
        model: &Path,
        vocabulary: &Path,
        shortlist: &Path,
        ssplit: Option<&Path>,
        arch: ModelArch,
    ) -> Result<Self, String> {
        Self::with_arch_and_target_vocab(model, vocabulary, shortlist, ssplit, arch, None)
    }

    /// Two-vocab models (Mozilla's bergamot en-zh, en-ja, en-ko, en-zh_hant,
    /// zh_hant-en) ship a `srcvocab.*.spm` and a `trgvocab.*.spm` and have
    /// separate `encoder_Wemb` / `decoder_Wemb` tensors. Pass the source
    /// vocab as `vocabulary` and the target vocab via `target_vocabulary`.
    /// Single-vocab models pass `target_vocabulary=None`.
    pub fn with_arch_and_target_vocab(
        model: &Path,
        vocabulary: &Path,
        shortlist: &Path,
        ssplit: Option<&Path>,
        arch: ModelArch,
        target_vocabulary: Option<&Path>,
    ) -> Result<Self, String> {
        let model_c = path_to_cstring(model);
        let vocab_c = path_to_cstring(vocabulary);
        let short_c = path_to_cstring(shortlist);
        let ssplit_c = ssplit.map(path_to_cstring);
        let tgt_vocab_c = target_vocabulary.map(path_to_cstring);
        let ptr = unsafe {
            slimt_model_new(
                model_c.as_ptr(),
                vocab_c.as_ptr(),
                short_c.as_ptr(),
                ssplit_c
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                arch.encoder_layers,
                arch.decoder_layers,
                arch.feed_forward_depth,
                arch.num_heads,
                tgt_vocab_c
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
            )
        };
        if ptr.is_null() {
            return Err(format!("Failed to create slimt model: {}", last_error()));
        }
        Ok(Self { ptr })
    }
}

pub struct BlockingService {
    ptr: *mut c_void,
}

unsafe impl Send for BlockingService {}
unsafe impl Sync for BlockingService {}

impl Drop for BlockingService {
    fn drop(&mut self) {
        unsafe { slimt_service_delete(self.ptr) }
    }
}

impl BlockingService {
    pub fn new(cache_size: usize) -> Self {
        Self::with_workers(4, cache_size)
    }

    pub fn with_workers(workers: usize, cache_size: usize) -> Self {
        let ptr = unsafe { slimt_service_new(workers, cache_size) };
        assert!(
            !ptr.is_null(),
            "Failed to create slimt service: {}",
            last_error()
        );
        Self { ptr }
    }

    pub fn translate(&self, model: &TranslationModel, inputs: &[&str]) -> Vec<String> {
        let (c_inputs, ptrs) = prepare_inputs(inputs);
        if ptrs.is_empty() {
            return Vec::new();
        }
        let raw = unsafe {
            slimt_service_translate(
                self.ptr,
                model.ptr,
                ptrs.as_ptr(),
                ptrs.len(),
                false,
                None,
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        let out = collect(raw, ptrs.len(), false);
        drop(c_inputs);
        out.into_iter().map(|r| r.target).collect()
    }

    /// Translate `inputs` in a single batch (so the batcher can pack across all
    /// of them) while reporting progress and honoring cancellation.
    ///
    /// `on_progress` is called from slimt worker threads — concurrently — once
    /// per input as it finishes, with the number that just completed. It must
    /// be cheap and non-blocking (e.g. bump an atomic); blocking work there
    /// serializes the worker pool.
    ///
    /// `cancel` is polled by the worker threads. Setting it to `true` (from any
    /// thread) makes the workers abort pending batches instead of translating
    /// them, so the call returns within ~one batch per worker. Returns `None`
    /// when the run was cancelled (partial output is discarded), `Some` with
    /// one translation per input otherwise. Pass a fresh `false` flag per call.
    pub fn translate_with_progress<F: Fn(usize) + Sync>(
        &self,
        model: &TranslationModel,
        inputs: &[&str],
        cancel: &std::sync::atomic::AtomicBool,
        on_progress: F,
    ) -> Option<Vec<String>> {
        let (c_inputs, ptrs) = prepare_inputs(inputs);
        if ptrs.is_empty() {
            return Some(Vec::new());
        }
        unsafe extern "C" fn trampoline<F: Fn(usize)>(user_data: *mut c_void, completed: usize) {
            let f = unsafe { &*(user_data as *const F) };
            f(completed);
        }
        let raw = unsafe {
            slimt_service_translate(
                self.ptr,
                model.ptr,
                ptrs.as_ptr(),
                ptrs.len(),
                false,
                Some(trampoline::<F>),
                &on_progress as *const F as *mut c_void,
                cancel as *const std::sync::atomic::AtomicBool as *const c_void,
            )
        };
        drop(c_inputs);
        if raw.is_null() {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return None;
            }
            panic!("slimt translation failed: {}", last_error());
        }
        let out = collect(raw, ptrs.len(), false);
        Some(out.into_iter().map(|r| r.target).collect())
    }

    /// Alignment-bearing counterpart of [`Self::translate_with_progress`]: one
    /// batch, per-input progress, honors `cancel`. Returns `None` on cancel.
    pub fn translate_with_alignment_progress<F: Fn(usize) + Sync>(
        &self,
        model: &TranslationModel,
        inputs: &[&str],
        cancel: &std::sync::atomic::AtomicBool,
        on_progress: F,
    ) -> Option<Vec<TranslationWithAlignment>> {
        let (c_inputs, ptrs) = prepare_inputs(inputs);
        if ptrs.is_empty() {
            return Some(Vec::new());
        }
        unsafe extern "C" fn trampoline<F: Fn(usize)>(user_data: *mut c_void, completed: usize) {
            let f = unsafe { &*(user_data as *const F) };
            f(completed);
        }
        let raw = unsafe {
            slimt_service_translate(
                self.ptr,
                model.ptr,
                ptrs.as_ptr(),
                ptrs.len(),
                true,
                Some(trampoline::<F>),
                &on_progress as *const F as *mut c_void,
                cancel as *const std::sync::atomic::AtomicBool as *const c_void,
            )
        };
        drop(c_inputs);
        if raw.is_null() {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return None;
            }
            panic!("slimt translation failed: {}", last_error());
        }
        Some(collect(raw, ptrs.len(), true))
    }

    pub fn translate_with_alignment(
        &self,
        model: &TranslationModel,
        inputs: &[&str],
    ) -> Vec<TranslationWithAlignment> {
        let (c_inputs, ptrs) = prepare_inputs(inputs);
        if ptrs.is_empty() {
            return Vec::new();
        }
        let raw = unsafe {
            slimt_service_translate(
                self.ptr,
                model.ptr,
                ptrs.as_ptr(),
                ptrs.len(),
                true,
                None,
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        let out = collect(raw, ptrs.len(), true);
        drop(c_inputs);
        out
    }

    /// Translate and, for each low-confidence target word, surface the
    /// alternatives the model would accept there (filtered to whole-word,
    /// non-reordering, non-inflection candidates above `opts.gate`). Forces a
    /// greedy decode, so hard sentences may translate differently than
    /// [`Self::translate`] would.
    pub fn translate_with_alternatives(
        &self,
        model: &TranslationModel,
        inputs: &[&str],
        opts: AlternativesOptions,
    ) -> Vec<TranslationWithAlternatives> {
        let (c_inputs, ptrs) = prepare_inputs(inputs);
        if ptrs.is_empty() {
            return Vec::new();
        }
        let c_opts = CTranslateOptions {
            want_alignment: true,
            want_alternatives: true,
            max_alternatives: opts.max_alternatives,
            gate: opts.gate,
        };
        let raw = unsafe {
            slimt_service_translate_ex(
                self.ptr,
                model.ptr,
                ptrs.as_ptr(),
                ptrs.len(),
                &c_opts,
                None,
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        let out = collect_ex(raw, ptrs.len());
        drop(c_inputs);
        out
    }

    /// Re-translate `source` with `target_prefix` forced as the start of the
    /// output (the confirmed text up to and including a user-picked word), then
    /// free-run the rest. The result carries fresh alternatives on the free-run
    /// tail so the user can keep steering.
    pub fn translate_with_prefix(
        &self,
        model: &TranslationModel,
        source: &str,
        target_prefix: &str,
        opts: AlternativesOptions,
    ) -> TranslationWithAlternatives {
        let c_source = CString::new(source).expect("source contains nul");
        let c_prefix = CString::new(target_prefix).expect("prefix contains nul");
        let src_ptrs = [c_source.as_ptr()];
        let pfx_ptrs = [c_prefix.as_ptr()];
        let c_opts = CTranslateOptions {
            want_alignment: true,
            want_alternatives: true,
            max_alternatives: opts.max_alternatives,
            gate: opts.gate,
        };
        let raw = unsafe {
            slimt_service_steer(
                self.ptr,
                model.ptr,
                src_ptrs.as_ptr(),
                pfx_ptrs.as_ptr(),
                1,
                &c_opts,
                None,
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        let mut out = collect_ex(raw, 1);
        drop((c_source, c_prefix));
        out.pop().expect("steer returned no result")
    }

    pub fn pivot(
        &self,
        first: &TranslationModel,
        second: &TranslationModel,
        inputs: &[&str],
    ) -> Vec<String> {
        let (c_inputs, ptrs) = prepare_inputs(inputs);
        if ptrs.is_empty() {
            return Vec::new();
        }
        let raw = unsafe {
            slimt_service_pivot(
                self.ptr,
                first.ptr,
                second.ptr,
                ptrs.as_ptr(),
                ptrs.len(),
                false,
                None,
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        let out = collect(raw, ptrs.len(), false);
        drop(c_inputs);
        out.into_iter().map(|r| r.target).collect()
    }

    /// Cancellable, progress-reporting [`Self::pivot`]. `on_progress` fires once
    /// per input as its two-leg pivot completes, from a worker thread; `cancel`
    /// aborts both legs within ~one batch. Returns `None` on cancel.
    pub fn pivot_with_progress<F: Fn(usize) + Sync>(
        &self,
        first: &TranslationModel,
        second: &TranslationModel,
        inputs: &[&str],
        cancel: &std::sync::atomic::AtomicBool,
        on_progress: F,
    ) -> Option<Vec<String>> {
        let (c_inputs, ptrs) = prepare_inputs(inputs);
        if ptrs.is_empty() {
            return Some(Vec::new());
        }
        unsafe extern "C" fn trampoline<F: Fn(usize)>(user_data: *mut c_void, completed: usize) {
            let f = unsafe { &*(user_data as *const F) };
            f(completed);
        }
        let raw = unsafe {
            slimt_service_pivot(
                self.ptr,
                first.ptr,
                second.ptr,
                ptrs.as_ptr(),
                ptrs.len(),
                false,
                Some(trampoline::<F>),
                &on_progress as *const F as *mut c_void,
                cancel as *const std::sync::atomic::AtomicBool as *const c_void,
            )
        };
        drop(c_inputs);
        if raw.is_null() {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return None;
            }
            panic!("slimt pivot failed: {}", last_error());
        }
        let out = collect(raw, ptrs.len(), false);
        Some(out.into_iter().map(|r| r.target).collect())
    }

    pub fn pivot_with_alignment(
        &self,
        first: &TranslationModel,
        second: &TranslationModel,
        inputs: &[&str],
    ) -> Vec<TranslationWithAlignment> {
        let (c_inputs, ptrs) = prepare_inputs(inputs);
        if ptrs.is_empty() {
            return Vec::new();
        }
        let raw = unsafe {
            slimt_service_pivot(
                self.ptr,
                first.ptr,
                second.ptr,
                ptrs.as_ptr(),
                ptrs.len(),
                true,
                None,
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        let out = collect(raw, ptrs.len(), true);
        drop(c_inputs);
        out
    }

    /// Cancellable, progress-reporting [`Self::pivot_with_alignment`].
    pub fn pivot_with_alignment_progress<F: Fn(usize) + Sync>(
        &self,
        first: &TranslationModel,
        second: &TranslationModel,
        inputs: &[&str],
        cancel: &std::sync::atomic::AtomicBool,
        on_progress: F,
    ) -> Option<Vec<TranslationWithAlignment>> {
        let (c_inputs, ptrs) = prepare_inputs(inputs);
        if ptrs.is_empty() {
            return Some(Vec::new());
        }
        unsafe extern "C" fn trampoline<F: Fn(usize)>(user_data: *mut c_void, completed: usize) {
            let f = unsafe { &*(user_data as *const F) };
            f(completed);
        }
        let raw = unsafe {
            slimt_service_pivot(
                self.ptr,
                first.ptr,
                second.ptr,
                ptrs.as_ptr(),
                ptrs.len(),
                true,
                Some(trampoline::<F>),
                &on_progress as *const F as *mut c_void,
                cancel as *const std::sync::atomic::AtomicBool as *const c_void,
            )
        };
        drop(c_inputs);
        if raw.is_null() {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return None;
            }
            panic!("slimt pivot failed: {}", last_error());
        }
        Some(collect(raw, ptrs.len(), true))
    }
}

fn prepare_inputs(inputs: &[&str]) -> (Vec<CString>, Vec<*const c_char>) {
    let c_inputs: Vec<CString> = inputs
        .iter()
        .map(|s| CString::new(*s).expect("input contains nul"))
        .collect();
    let ptrs: Vec<*const c_char> = c_inputs.iter().map(|s| s.as_ptr()).collect();
    (c_inputs, ptrs)
}

fn collect(
    raw: *mut CTranslation,
    count: usize,
    want_alignment: bool,
) -> Vec<TranslationWithAlignment> {
    assert!(!raw.is_null(), "slimt translation failed: {}", last_error());
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let c_result = unsafe { &*raw.add(i) };
        let source = unsafe { CStr::from_ptr(c_result.source) }
            .to_string_lossy()
            .into_owned();
        let target = unsafe { CStr::from_ptr(c_result.target) }
            .to_string_lossy()
            .into_owned();

        let alignments = if want_alignment && c_result.alignment_count > 0 {
            let byte_alignments = unsafe {
                std::slice::from_raw_parts(c_result.alignments, c_result.alignment_count)
            };
            let src_offsets = byte_to_char_offsets(&source);
            let tgt_offsets = byte_to_char_offsets(&target);
            byte_alignments
                .iter()
                .map(|a| TokenAlignment {
                    src_begin: src_offsets[a.src_begin],
                    src_end: src_offsets[a.src_end],
                    tgt_begin: tgt_offsets[a.tgt_begin],
                    tgt_end: tgt_offsets[a.tgt_end],
                })
                .collect()
        } else {
            Vec::new()
        };

        out.push(TranslationWithAlignment {
            source,
            target,
            alignments,
        });
    }
    unsafe { slimt_translations_delete(raw, count) };
    out
}

fn collect_ex(raw: *mut CTranslation, count: usize) -> Vec<TranslationWithAlternatives> {
    assert!(!raw.is_null(), "slimt translation failed: {}", last_error());
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let c_result = unsafe { &*raw.add(i) };
        let source = unsafe { CStr::from_ptr(c_result.source) }
            .to_string_lossy()
            .into_owned();
        let target = unsafe { CStr::from_ptr(c_result.target) }
            .to_string_lossy()
            .into_owned();
        let src_offsets = byte_to_char_offsets(&source);
        let tgt_offsets = byte_to_char_offsets(&target);

        let alignments = if c_result.alignment_count > 0 {
            let byte_alignments = unsafe {
                std::slice::from_raw_parts(c_result.alignments, c_result.alignment_count)
            };
            byte_alignments
                .iter()
                .map(|a| TokenAlignment {
                    src_begin: src_offsets[a.src_begin],
                    src_end: src_offsets[a.src_end],
                    tgt_begin: tgt_offsets[a.tgt_begin],
                    tgt_end: tgt_offsets[a.tgt_end],
                })
                .collect()
        } else {
            Vec::new()
        };

        let alternatives = if c_result.alternatives_count > 0 {
            let words = unsafe {
                std::slice::from_raw_parts(c_result.alternatives, c_result.alternatives_count)
            };
            words
                .iter()
                .map(|w| {
                    let options = unsafe {
                        std::slice::from_raw_parts(w.options, w.option_count)
                    }
                    .iter()
                    .map(|o| Alternative {
                        text: unsafe { CStr::from_ptr(o.text) }
                            .to_string_lossy()
                            .into_owned(),
                        prob: o.prob,
                    })
                    .collect();
                    WordAlternatives {
                        tgt_begin: tgt_offsets[w.tgt_begin],
                        tgt_end: tgt_offsets[w.tgt_end],
                        confidence: w.confidence,
                        options,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        out.push(TranslationWithAlternatives {
            source,
            target,
            alignments,
            alternatives,
        });
    }
    unsafe { slimt_translations_delete(raw, count) };
    out
}
