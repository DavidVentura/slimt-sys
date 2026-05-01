use std::env;
use std::fs;
use std::path::PathBuf;

fn latest_android_ndk_from_sdk() -> Option<PathBuf> {
    let sdk_root = env::var_os("ANDROID_SDK_ROOT")
        .or_else(|| env::var_os("ANDROID_HOME"))
        .map(PathBuf::from)?;
    let ndk_root = sdk_root.join("ndk");
    let mut entries = fs::read_dir(ndk_root)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    entries.sort();
    entries.pop()
}

fn android_ndk_root() -> Option<PathBuf> {
    env::var_os("ANDROID_NDK_ROOT")
        .or_else(|| env::var_os("ANDROID_NDK_HOME"))
        .or_else(|| env::var_os("ANDROID_NDK"))
        .map(PathBuf::from)
        .or_else(latest_android_ndk_from_sdk)
}

fn slimt_source_dir() -> PathBuf {
    if let Ok(dir) = env::var("SLIMT_SOURCE_DIR") {
        return PathBuf::from(dir);
    }
    let cargo_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(parent) = cargo_manifest.parent() {
        let candidate = parent.join("slimt");
        if candidate.join("CMakeLists.txt").exists() {
            return candidate;
        }
    }
    let home = env::var("HOME").expect("HOME is required to locate slimt sources");
    PathBuf::from(home).join("git/slimt")
}

fn main() {
    println!("cargo:rerun-if-changed=bindings/slimt_wrapper.cpp");
    println!("cargo:rerun-if-changed=bindings/CMakeLists.txt");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SLIMT_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=SLIMT_ENABLE_LTO");

    // Re-run when the upstream slimt sources change so local edits are picked
    // up without manually wiping the cargo cache.
    if let Ok(entries) = fs::read_dir(slimt_source_dir().join("slimt")) {
        for entry in entries.filter_map(Result::ok) {
            if let Some(p) = entry.path().to_str() {
                println!("cargo:rerun-if-changed={p}");
            }
        }
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target = env::var("TARGET").unwrap();
    let host = env::var("HOST").unwrap();
    let is_android = target_os == "android";
    let is_x86 = target_arch == "x86_64" || target_arch == "x86";
    let is_arm = target_arch == "aarch64" || target_arch == "arm";

    let slimt_dir = slimt_source_dir();
    assert!(
        slimt_dir.join("CMakeLists.txt").exists(),
        "slimt source not found at {} (set SLIMT_SOURCE_DIR)",
        slimt_dir.display(),
    );

    // Default to clang on x86 hosts: ~10 % faster than the system gcc on the
    // moby-dick benchmark, and clang's LLVM bitcode lines up with rust-lld so
    // we can also enable LTO without fat-object workarounds. Honor explicit
    // CC/CXX overrides; fall through to the cmake-rs default (gcc) elsewhere.
    let want_clang = env::var("CC").is_err()
        && env::var("CXX").is_err()
        && !is_android
        && target == host
        && std::path::Path::new("/usr/bin/clang").exists()
        && std::path::Path::new("/usr/bin/clang++").exists();
    if want_clang {
        unsafe {
            env::set_var("CC", "clang");
            env::set_var("CXX", "clang++");
        }
    }

    let mut config = cmake::Config::new("bindings");
    config
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("SLIMT_SOURCE_DIR", slimt_dir.to_str().unwrap())
        .define("WITH_TESTS", "OFF")
        .define("WITH_GEMMOLOGY", "OFF")
        .define("BUILD_SHARED", "OFF")
        .define("BUILD_STATIC", "ON")
        .define("BUILD_PYTHON", "OFF")
        .define("BUILD_JNI", "OFF")
        .define("USE_BUILTIN_SENTENCEPIECE", "ON")
        .define("SPM_ENABLE_SHARED", "OFF")
        .define("SPM_ENABLE_TCMALLOC", "OFF")
        // Vendor pcre2 statically: slimt downloads + builds pcre2-8 inside the
        // CMake build dir. This keeps the resulting .a self-contained for APK
        // distribution without depending on the host's libpcre2-8.
        .define("SLIMT_USE_INTERNAL_PCRE2", "ON");

    // C++ LTO. On by default when the C++ compiler is clang (LLVM bitcode
    // matches rust-lld natively, ~8 % perf win on the moby-dick bench).
    // Opt out via SLIMT_ENABLE_LTO=0 / off / false.
    let lto_default = want_clang;
    let lto_env = env::var("SLIMT_ENABLE_LTO").ok();
    let enable_lto = match lto_env.as_deref() {
        Some("0") | Some("OFF") | Some("off") | Some("false") => false,
        Some(_) => true,
        None => lto_default,
    };
    if enable_lto {
        config.define("SLIMT_ENABLE_LTO", "ON");
    }

    if matches!(
        env::var("SLIMT_PROFILE_BUILD").as_deref(),
        Ok("1") | Ok("ON") | Ok("on") | Ok("true")
    ) {
        config.define("SLIMT_PROFILE_BUILD", "ON");
    }

    // Mirror bergamot's compute backend layout, fully static:
    //   * x86  -> intgemm drives int8 GEMM, ruy provides sgemm
    //             (we keep WITH_RUY=ON to build & link ruy; the wrapper's
    //             CMakeLists strips SLIMT_HAS_RUY from slimt-static so the QMM
    //             dispatch picks intgemm without colliding with ruy's qmm).
    //   * arm  -> ruy for both int8 and sgemm.
    // BLAS is always OFF: linking system BLAS would break standalone
    // distribution (e.g. APKs) and we don't need it when ruy is available.
    config.define("WITH_BLAS", "OFF");
    if is_x86 {
        let baseline = if target_arch == "x86_64" {
            "x86-64-v2"
        } else {
            "i686"
        };
        config
            .define("WITH_INTGEMM", "ON")
            .define("WITH_RUY", "ON")
            .define("SLIMT_FORCE_INTGEMM_QMM_WITH_RUY_SGEMM", "ON")
            .define("SLIMT_X86_BASELINE", baseline);
    } else {
        config
            .define("WITH_INTGEMM", "OFF")
            .define("WITH_RUY", "ON");
    }

    if is_android {
        let android_abi = match target_arch.as_str() {
            "aarch64" => "arm64-v8a",
            "arm" => "armeabi-v7a",
            "x86_64" => "x86_64",
            "x86" => "x86",
            _ => panic!("Unsupported Android target arch: {target_arch}"),
        };
        let android_platform = env::var("CARGO_NDK_PLATFORM")
            .ok()
            .filter(|value| value.chars().all(|ch| ch.is_ascii_digit()))
            .or_else(|| {
                env::var("ANDROID_PLATFORM")
                    .ok()
                    .and_then(|value| value.strip_prefix("android-").map(str::to_string))
                    .filter(|value| value.chars().all(|ch| ch.is_ascii_digit()))
            })
            .unwrap_or_else(|| "21".to_string());
        let ndk_root = android_ndk_root()
            .expect("Android target requires ANDROID_NDK_ROOT or ANDROID_SDK_ROOT");

        config
            .generator("Ninja")
            .define(
                "CMAKE_TOOLCHAIN_FILE",
                ndk_root.join("build/cmake/android.toolchain.cmake"),
            )
            .define("ANDROID_ABI", android_abi)
            .define("ANDROID_PLATFORM", format!("android-{android_platform}"));
    }

    if target != host && !is_android {
        let cmake_system_processor = match target_arch.as_str() {
            "x86_64" => "x86_64",
            "x86" => "i686",
            "aarch64" => "aarch64",
            "arm" => "armv7",
            _ => &target_arch,
        };

        let cmake_c_compiler = match target_arch.as_str() {
            "aarch64" => "aarch64-linux-gnu-gcc",
            "arm" => "arm-linux-gnueabihf-gcc",
            _ => "gcc",
        };

        let cmake_cxx_compiler = match target_arch.as_str() {
            "aarch64" => "aarch64-linux-gnu-g++",
            "arm" => "arm-linux-gnueabihf-g++",
            _ => "g++",
        };

        config.define("CMAKE_SYSTEM_NAME", "Linux");
        config.define("CMAKE_SYSTEM_PROCESSOR", cmake_system_processor);
        config.define("CMAKE_C_COMPILER", cmake_c_compiler);
        config.define("CMAKE_CXX_COMPILER", cmake_cxx_compiler);
    }

    let dst = config.build();
    let build_dir = dst.join("build");

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!(
        "cargo:rustc-link-search=native={}/slimt/slimt",
        build_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/slimt/3rd-party/sentencepiece/src",
        build_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/slimt/3rd-party/intgemm",
        build_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/slimt/3rd-party/ruy/ruy",
        build_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/slimt/3rd-party/ruy/ruy/profiler",
        build_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/slimt/3rd-party/ruy/third_party/cpuinfo",
        build_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/slimt/3rd-party/ruy/third_party/cpuinfo/deps/clog",
        build_dir.display()
    );
    println!("cargo:rustc-link-search=native={}/lib", build_dir.display());

    println!("cargo:rustc-link-lib=static=slimt_wrapper");
    println!("cargo:rustc-link-lib=static=slimt");
    println!("cargo:rustc-link-lib=static=sentencepiece");

    let link_ruy = is_x86 || is_arm || (target_arch != "x86_64" && target_arch != "x86");

    if is_x86 {
        println!("cargo:rustc-link-lib=static=intgemm");
    }

    if link_ruy {
        for lib in [
            "ruy_ctx",
            "ruy_context",
            "ruy_context_get_ctx",
            "ruy_frontend",
            "ruy_trmul",
            "ruy_prepare_packed_matrices",
            "ruy_system_aligned_alloc",
            "ruy_allocator",
            "ruy_block_map",
            "ruy_blocking_counter",
            "ruy_cpuinfo",
            "ruy_denormal",
            "ruy_thread_pool",
            "ruy_tune",
            "ruy_wait",
            "ruy_prepacked_cache",
            "ruy_apply_multiplier",
            "ruy_profiler_instrumentation",
        ] {
            println!("cargo:rustc-link-lib=static={lib}");
        }
        if is_x86 {
            for lib in [
                "ruy_have_built_path_for_avx",
                "ruy_have_built_path_for_avx2_fma",
                "ruy_have_built_path_for_avx512",
                "ruy_kernel_avx",
                "ruy_kernel_avx2_fma",
                "ruy_kernel_avx512",
                "ruy_pack_avx",
                "ruy_pack_avx2_fma",
                "ruy_pack_avx512",
            ] {
                println!("cargo:rustc-link-lib=static={lib}");
            }
        } else {
            println!("cargo:rustc-link-lib=static=ruy_kernel_arm");
            println!("cargo:rustc-link-lib=static=ruy_pack_arm");
        }
        println!("cargo:rustc-link-lib=static=cpuinfo");
        println!("cargo:rustc-link-lib=static=clog");
    }

    // PCRE2 is built and installed by slimt (SLIMT_USE_INTERNAL_PCRE2=ON) into
    // the build dir's `lib/` (already on the search path).
    println!("cargo:rustc-link-lib=static=pcre2-8");

    if is_android {
        println!("cargo:rustc-link-lib=c++_shared");
    } else if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    if !is_android {
        println!("cargo:rustc-link-lib=pthread");
    }
}
