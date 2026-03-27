//! Build script for outerlink-client.
//!
//! Compiles the C interposition library (csrc/interpose.c) and links it into
//! the final cdylib. The resulting shared library can be loaded via LD_PRELOAD
//! to intercept CUDA Driver API calls.
//!
//! Only compiles the C code on Linux -- the interposition mechanism
//! (__libc_dlsym, LD_PRELOAD) is Linux/glibc-specific. On Windows, the Rust
//! code compiles normally but the C interposition layer is skipped (no
//! LD_PRELOAD on Windows).

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let cuda_stubs_dir = std::path::Path::new(&manifest_dir)
        .join("..")
        .join("..")
        .join("cuda-stubs");

    // Only compile the C interposition layer on Linux
    if target_os == "linux" {
        cc::Build::new()
            .file("csrc/interpose.c")
            .include("csrc")
            .include(&cuda_stubs_dir)
            // Use default visibility so ALL hook functions (hook_cuInit,
            // hook_cuMemAlloc_v2, dlsym, etc.) appear in the .so's dynamic
            // symbol table. LD_PRELOAD interception requires these symbols
            // to be visible -- hidden visibility would make them internal
            // and defeat the entire interposition mechanism.
            .flag("-fvisibility=default")
            .flag("-Wall")
            .flag("-Wextra")
            .flag("-Wno-unused-parameter")
            .flag("-O2")
            .compile("outerlink_interpose");

        // Link system libraries needed by the interposition layer
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");

        // Apply version script to control symbol visibility.
        // Only dlsym, hook_*, nvml_hook_*, and ol_* are exported;
        // everything else is hidden to avoid namespace pollution.
        println!("cargo:rustc-cdylib-link-arg=-Wl,--version-script=csrc/exports.map");
        println!("cargo:rerun-if-changed=csrc/exports.map");
    }

    // Rebuild if any C source or the build script changes
    println!("cargo:rerun-if-changed=csrc/interpose.c");
    println!("cargo:rerun-if-changed=csrc/interpose.h");
    println!("cargo:rerun-if-changed={}", cuda_stubs_dir.join("cuda.h").display());
    println!("cargo:rerun-if-changed=build.rs");
}
