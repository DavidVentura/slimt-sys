cmake_minimum_required(VERSION 3.5.1)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "/usr/bin/clang")
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
endif()

project(slimt-sys CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

# Optional C++ link-time optimization. Off by default to match bergamot, but
# turn-on-able for benchmarking or production builds where the longer link
# time is acceptable. Honors CMP0069 NEW so it propagates to all targets
# created after this point (slimt, intgemm, ruy, sentencepiece, our wrapper).
if(SLIMT_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _slimt_ipo_supported OUTPUT _slimt_ipo_msg)
  if(_slimt_ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    # Emit fat LTO objects so the resulting .a has both bytecode for
    # cross-TU optimization AND a normal object section the system linker can
    # consume when rustc later links the static archive into the final
    # binary. Without -ffat-lto-objects, rust-lld can't see the C symbols
    # because they only exist as LLVM bitcode.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffat-lto-objects")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffat-lto-objects")
    message(STATUS "slimt-sys: IPO/LTO enabled (fat objects)")
  else()
    message(WARNING "slimt-sys: IPO/LTO requested but not supported: ${_slimt_ipo_msg}")
  endif()
endif()

# slimt build options - we choose our own backend per arch and avoid extras
set(WITH_TESTS OFF CACHE BOOL "" FORCE)
set(WITH_GEMMOLOGY OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC ON CACHE BOOL "" FORCE)
set(BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(BUILD_JNI OFF CACHE BOOL "" FORCE)
set(USE_BUILTIN_SENTENCEPIECE ON CACHE BOOL "" FORCE)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)

# Size / reproducibility / perf flags ported from bergamot-sys's CMakeLists so
# the resulting .a behaves the same way ours used to:
#   * -fdebug-prefix-map + -Wno-builtin-macro-redefined: keep paths and
#     __FILE__ macros stable across build dirs (reproducible builds)
#   * -ffunction-sections -fdata-sections -fvisibility=hidden
#     -fvisibility-inlines-hidden + linker --gc-sections / --strip-all: drop
#     unreferenced symbols and debug info from the final shared object
#   * -g0 / -s: no debug info; let the linker strip
# A C++ side LTO toggle is intentionally not enabled (matches bergamot — it's
# commented out there). Rust-side LTO is on via Cargo.toml [profile.release].
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdebug-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=. -Wno-builtin-macro-redefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdebug-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=. -Wno-builtin-macro-redefined")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections -fvisibility=hidden -fvisibility-inlines-hidden")
if(SLIMT_PROFILE_BUILD)
  # Keep frame pointers + DWARF info so `perf record --call-graph=dwarf` can
  # produce useful flamegraphs. Skip --strip-all so symbols survive.
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g0 -s")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g0 -s")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -Wl,--strip-all")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--strip-all")
endif()

# Pin the x86 baseline ISA to x86-64-v2 (SSE4.2-era) like bergamot's
# BUILD_ARCH did. Keeps generated kernels reproducible across hosts and
# avoids accidentally baking AVX-only instructions into Android-shippable
# binaries. ARM doesn't get a counterpart — slimt's ruy backend does runtime
# CPU detection.
if(SLIMT_X86_BASELINE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=${SLIMT_X86_BASELINE}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${SLIMT_X86_BASELINE}")
endif()

# slimt's CMakeLists references CMAKE_SOURCE_DIR for include paths, which
# resolves to the top-level project (this directory) when consumed via
# add_subdirectory from cargo. Add the include path globally so slimt's own
# sources can locate "slimt/*.hh" headers regardless.
include_directories(${SLIMT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/slimt)

add_subdirectory(${SLIMT_SOURCE_DIR} slimt EXCLUDE_FROM_ALL)

# When building on x86 we want intgemm to drive the int8 GEMM kernels (matches
# bergamot's x86 layout) while ruy handles float sgemm out of TensorOps.cc.
# slimt's CMake wires both WITH_INTGEMM and WITH_RUY into a shared
# `kProvider` constexpr in QMM.cc — defining both ABIs in the same TU is a
# hard error. Strip SLIMT_HAS_RUY from slimt-static's compile definitions:
# the ruy 3rd-party library is still built and linked (so TensorOps's
# `#include "ruy/ruy.h"` path resolves) but the QMM dispatch falls
# exclusively to intgemm.
if(SLIMT_FORCE_INTGEMM_QMM_WITH_RUY_SGEMM)
  get_target_property(_slimt_defs slimt-static COMPILE_DEFINITIONS)
  if(_slimt_defs)
    list(REMOVE_ITEM _slimt_defs SLIMT_HAS_RUY)
    set_target_properties(slimt-static PROPERTIES COMPILE_DEFINITIONS "${_slimt_defs}")
  endif()
endif()

add_library(slimt_wrapper STATIC slimt_wrapper.cpp)
target_link_libraries(slimt_wrapper PRIVATE slimt-static)
target_include_directories(slimt_wrapper PRIVATE ${SLIMT_SOURCE_DIR})

install(TARGETS slimt_wrapper ARCHIVE DESTINATION lib)
