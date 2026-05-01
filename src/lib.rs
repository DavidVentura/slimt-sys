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
}

pub struct TranslationWithAlignment {
    pub source: String,
    pub target: String,
    pub alignments: Vec<TokenAlignment>,
}

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
    ) -> *mut CTranslation;

    fn slimt_service_pivot(
        service_ptr: *mut c_void,
        first_model_ptr: *mut c_void,
        second_model_ptr: *mut c_void,
        inputs: *const *const c_char,
        count: usize,
        want_alignment: bool,
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
        Self::with_arch_and_target_vocab(
            model, vocabulary, shortlist, ssplit, arch, None,
        )
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
            slimt_service_translate(self.ptr, model.ptr, ptrs.as_ptr(), ptrs.len(), false)
        };
        let out = collect(raw, ptrs.len(), false);
        drop(c_inputs);
        out.into_iter().map(|r| r.target).collect()
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
            slimt_service_translate(self.ptr, model.ptr, ptrs.as_ptr(), ptrs.len(), true)
        };
        let out = collect(raw, ptrs.len(), true);
        drop(c_inputs);
        out
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
            )
        };
        let out = collect(raw, ptrs.len(), false);
        drop(c_inputs);
        out.into_iter().map(|r| r.target).collect()
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
            )
        };
        let out = collect(raw, ptrs.len(), true);
        drop(c_inputs);
        out
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
    assert!(
        !raw.is_null(),
        "slimt translation failed: {}",
        last_error()
    );
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
