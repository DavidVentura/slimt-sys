#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "slimt/Frontend.hh"
#include "slimt/Io.hh"
#include "slimt/Model.hh"
#include "slimt/Response.hh"
#include "slimt/Types.hh"

#ifdef __ANDROID__
#include <android/log.h>
#define SLIMT_LOG(...) \
    __android_log_print(ANDROID_LOG_ERROR, "slimt_wrapper", __VA_ARGS__)
#else
#define SLIMT_LOG(...) fprintf(stderr, "[slimt_wrapper] " __VA_ARGS__)
#endif

// Last error string set by any C entry point that catches a C++ exception.
// Per-thread so concurrent translate calls don't clobber each other.
static thread_local std::string g_last_error;

static void record_error(const char* where, const char* what) {
    g_last_error.assign(where);
    g_last_error.append(": ");
    g_last_error.append(what);
    SLIMT_LOG("%s: %s", where, what);
}

#define SLIMT_TRY_ENTRY(where) try {
#define SLIMT_CATCH_ENTRY(where, on_error_return)                           \
    }                                                                       \
    catch (const std::exception& _e) {                                      \
        record_error(where, _e.what());                                     \
        return on_error_return;                                             \
    }                                                                       \
    catch (...) {                                                           \
        record_error(where, "unknown C++ exception");                       \
        return on_error_return;                                             \
    }

using slimt::AnnotatedText;
using slimt::Async;
using slimt::Config;
using slimt::Handle;
using slimt::Model;
using slimt::Options;
using slimt::Package;
using slimt::Range;
using slimt::Response;

extern "C" {

struct CTokenAlignment {
    size_t src_begin;
    size_t src_end;
    size_t tgt_begin;
    size_t tgt_end;
};

struct CTranslation {
    char* source;
    char* target;
    CTokenAlignment* alignments;
    size_t alignment_count;
};

// Returns the last error string set by any wrapper entry point on this
// thread (or empty string if no error). The returned pointer is valid until
// the next slimt_* call on this thread.
const char* slimt_last_error() {
    return g_last_error.c_str();
}

void* slimt_service_new(size_t workers, size_t cache_size) {
    SLIMT_TRY_ENTRY("slimt_service_new")
        Config config;
        config.workers = workers == 0 ? 1 : workers;
        config.cache_size = cache_size;
        return new Async(config);
    SLIMT_CATCH_ENTRY("slimt_service_new", nullptr)
}

void slimt_service_delete(void* service_ptr) {
    try {
        delete static_cast<Async*>(service_ptr);
    } catch (const std::exception& e) {
        record_error("slimt_service_delete", e.what());
    } catch (...) {
        record_error("slimt_service_delete", "unknown C++ exception");
    }
}

// Walk the parameter index of a marian-format binary model and find the
// largest "encoder_lN_" / "decoder_lN_" indices present. Used to set the
// architecture knobs that slimt does not auto-detect.
static void detect_layer_counts(const std::string& model_path,
                                size_t& encoder_layers,
                                size_t& decoder_layers) {
    encoder_layers = 0;
    decoder_layers = 0;
    slimt::io::MmapFile mmap(model_path);
    auto items = slimt::io::load_items(mmap.data());
    for (const auto& item : items) {
        const std::string& name = item.name;
        auto check_prefix = [&](const char* prefix, size_t& out) {
            const size_t plen = std::strlen(prefix);
            if (name.compare(0, plen, prefix) != 0) return;
            size_t idx = 0;
            size_t pos = plen;
            while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
                idx = idx * 10 + static_cast<size_t>(name[pos] - '0');
                ++pos;
            }
            if (pos == plen) return;
            if (idx > out) out = idx;
        };
        check_prefix("encoder_l", encoder_layers);
        check_prefix("decoder_l", decoder_layers);
    }
}

void* slimt_model_new(const char* model_path,
                      const char* vocabulary_path,
                      const char* shortlist_path,
                      const char* ssplit_path,
                      size_t encoder_layers,
                      size_t decoder_layers,
                      size_t feed_forward_depth,
                      size_t num_heads) {
    SLIMT_TRY_ENTRY("slimt_model_new")
        Package<std::string> package{
            .model = model_path ? std::string(model_path) : std::string(),
            .vocabulary = vocabulary_path ? std::string(vocabulary_path) : std::string(),
            .shortlist = shortlist_path ? std::string(shortlist_path) : std::string(),
            .ssplit = ssplit_path ? std::string(ssplit_path) : std::string(),
        };

        if (encoder_layers == 0 || decoder_layers == 0) {
            size_t detected_enc = 0;
            size_t detected_dec = 0;
            detect_layer_counts(package.model, detected_enc, detected_dec);
            if (encoder_layers == 0) encoder_layers = detected_enc;
            if (decoder_layers == 0) decoder_layers = detected_dec;
        }

        Model::Config config = slimt::preset::tiny();
        if (encoder_layers > 0) config.encoder_layers = encoder_layers;
        if (decoder_layers > 0) config.decoder_layers = decoder_layers;
        if (feed_forward_depth > 0) config.feed_forward_depth = feed_forward_depth;
        if (num_heads > 0) config.num_heads = num_heads;
        auto* model = new std::shared_ptr<Model>(std::make_shared<Model>(config, package));
        return model;
    SLIMT_CATCH_ENTRY("slimt_model_new", nullptr)
}

void slimt_model_delete(void* model_ptr) {
    try {
        delete static_cast<std::shared_ptr<Model>*>(model_ptr);
    } catch (const std::exception& e) {
        record_error("slimt_model_delete", e.what());
    } catch (...) {
        record_error("slimt_model_delete", "unknown C++ exception");
    }
}

static void extract_alignments(const Response& resp, std::vector<CTokenAlignment>& out) {
    const size_t num_sentences = resp.alignments.size();
    for (size_t s = 0; s < num_sentences; ++s) {
        const size_t num_target = resp.target.word_count(s);
        const size_t num_source = resp.source.word_count(s);
        if (num_source == 0) continue;

        const auto& sentence_alignments = resp.alignments[s];
        const size_t target_rows = std::min(sentence_alignments.size(), num_target);

        for (size_t t = 0; t < target_rows; ++t) {
            Range tgt_range = resp.target.word_as_range(s, t);
            if (tgt_range.begin == tgt_range.end) continue;

            const auto& row = sentence_alignments[t];
            if (row.empty()) continue;
            const size_t row_size = std::min(row.size(), num_source);
            size_t best_src = std::max_element(row.begin(), row.begin() + row_size) - row.begin();

            Range src_range = resp.source.word_as_range(s, best_src);
            out.push_back(CTokenAlignment{
                src_range.begin, src_range.end,
                tgt_range.begin, tgt_range.end});
        }
    }
}

static CTranslation* responses_to_c(std::vector<Response>&& responses, bool include_alignments) {
    const size_t count = responses.size();
    auto* results = new CTranslation[count];
    for (size_t i = 0; i < count; ++i) {
        Response& resp = responses[i];
        const std::string& source = resp.source.text;
        const std::string& target = resp.target.text;

        results[i].source = new char[source.size() + 1];
        std::memcpy(results[i].source, source.data(), source.size());
        results[i].source[source.size()] = '\0';

        results[i].target = new char[target.size() + 1];
        std::memcpy(results[i].target, target.data(), target.size());
        results[i].target[target.size()] = '\0';

        if (include_alignments) {
            std::vector<CTokenAlignment> alignments;
            extract_alignments(resp, alignments);
            results[i].alignment_count = alignments.size();
            results[i].alignments = new CTokenAlignment[alignments.size()];
            std::memcpy(results[i].alignments, alignments.data(),
                        alignments.size() * sizeof(CTokenAlignment));
        } else {
            results[i].alignment_count = 0;
            results[i].alignments = nullptr;
        }
    }
    return results;
}

static std::vector<Response> drain(std::vector<Handle>& handles) {
    std::vector<Response> responses;
    responses.reserve(handles.size());
    for (auto& handle : handles) {
        handle.future().wait();
        responses.push_back(handle.future().get());
    }
    return responses;
}

// Truncate a string for logging — keeps the head + tail so we can
// reconstruct the failing input from logcat without flooding it.
static std::string log_excerpt(const char* s) {
    if (!s) return "<null>";
    constexpr size_t kHead = 200;
    constexpr size_t kTail = 80;
    std::string str(s);
    if (str.size() <= kHead + kTail + 8) return str;
    std::string out;
    out.reserve(kHead + kTail + 16);
    out.append(str, 0, kHead);
    out.append("...[+");
    out.append(std::to_string(str.size() - kHead - kTail));
    out.append("B]...");
    out.append(str, str.size() - kTail, kTail);
    return out;
}

CTranslation* slimt_service_translate(void* service_ptr,
                                      void* model_ptr,
                                      const char* const* inputs,
                                      size_t count,
                                      bool html,
                                      bool want_alignment) {
    try {
        auto* service = static_cast<Async*>(service_ptr);
        auto& model = *static_cast<std::shared_ptr<Model>*>(model_ptr);

        Options options;
        options.html = html;
        options.alignment = want_alignment;

        std::vector<Handle> handles;
        handles.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            std::string source(inputs[i] ? inputs[i] : "");
            handles.push_back(service->translate(model, std::move(source), options));
        }

        return responses_to_c(drain(handles), want_alignment);
    } catch (const std::exception& e) {
        record_error("slimt_service_translate", e.what());
        for (size_t i = 0; i < count; ++i) {
            SLIMT_LOG("input[%zu] (html=%d): %s", i, (int)html,
                      log_excerpt(inputs[i]).c_str());
        }
        return nullptr;
    } catch (...) {
        record_error("slimt_service_translate", "unknown C++ exception");
        return nullptr;
    }
}

CTranslation* slimt_service_pivot(void* service_ptr,
                                  void* first_model_ptr,
                                  void* second_model_ptr,
                                  const char* const* inputs,
                                  size_t count,
                                  bool html,
                                  bool want_alignment) {
    SLIMT_TRY_ENTRY("slimt_service_pivot")
        auto* service = static_cast<Async*>(service_ptr);
        auto& first = *static_cast<std::shared_ptr<Model>*>(first_model_ptr);
        auto& second = *static_cast<std::shared_ptr<Model>*>(second_model_ptr);

        Options options;
        options.html = html;
        // See slimt_service_translate: HTML restore needs alignments
        // internally regardless of whether the caller asked for them.
        options.alignment = want_alignment || html;

        std::vector<Handle> handles;
        handles.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            std::string source(inputs[i] ? inputs[i] : "");
            handles.push_back(service->pivot(first, second, std::move(source), options));
        }

        return responses_to_c(drain(handles), want_alignment);
    SLIMT_CATCH_ENTRY("slimt_service_pivot", nullptr)
}

void slimt_translations_delete(CTranslation* results, size_t count) {
    if (!results) return;
    for (size_t i = 0; i < count; ++i) {
        delete[] results[i].source;
        delete[] results[i].target;
        delete[] results[i].alignments;
    }
    delete[] results;
}

}  // extern "C"
