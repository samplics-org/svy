use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

// --- diagnostics -------------------------------------------------------------

fn debug_on() -> bool {
    matches!(
        env::var("READSTAT_BUILD_DEBUG").as_deref(),
        Ok("1") | Ok("true") | Ok("yes") | Ok("on")
    )
}

macro_rules! diag {
    ($($t:tt)*) => {{
        if debug_on() {
            println!("cargo:warning={}", format!($($t)*));
        }
    }};
}

fn dump_env(keys: &[&str]) {
    for k in keys {
        match env::var(k) {
            Ok(v) => diag!("env {k}={v}"),
            Err(_) => diag!("env {k}=(unset)"),
        }
    }
}

// --- bindgen ----------------------------------------------------------------

fn detect_sysroot_for_target(target: &str) -> Option<PathBuf> {
    if let Ok(p) = env::var("BINDGEN_SYSROOT") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    for cc in [
        format!("{target}-gcc"),
        format!("{target}-clang"),
        "gcc".to_string(),
        "clang".to_string(),
    ] {
        if let Ok(out) = Command::new(&cc).arg("-print-sysroot").output() {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !s.is_empty() && Path::new(&s).exists() {
                    diag!("Detected sysroot from {cc}: {s}");
                    return Some(PathBuf::from(s));
                }
            }
        }
    }
    None
}

fn bindgen_with_includes(include_dir: &Path) {
    let target = env::var("TARGET").unwrap_or_default();
    let host = env::var("HOST").unwrap_or_default();

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_function("readstat_.*")
        .allowlist_type("readstat_.*")
        .allowlist_var("READSTAT_.*")
        .layout_tests(false)
        .clang_arg(format!("-I{}", include_dir.display()));

    if target != host && target.contains("linux") {
        if let Some(sysroot) = detect_sysroot_for_target(&target) {
            builder = builder
                .clang_arg(format!("--sysroot={}", sysroot.display()))
                .clang_arg(format!("-I{}/usr/include", sysroot.display()));
            let trip = if target.starts_with("aarch64") {
                "aarch64-linux-gnu"
            } else if target.starts_with("x86_64") {
                "x86_64-linux-gnu"
            } else {
                ""
            };
            if !trip.is_empty() {
                builder = builder.clang_arg(format!(
                    "-I{}/usr/include/{}",
                    sysroot.display(),
                    trip
                ));
            }
            diag!("bindgen using sysroot {}", sysroot.display());
        }
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    builder
        .generate()
        .expect("bindgen failed for readstat")
        .write_to_file(out.join("bindings.rs"))
        .expect("Couldn't write bindings!");
    println!("cargo:rerun-if-changed=wrapper.h");
}

// --- locating ReadStat sources ----------------------------------------------

fn find_readstat_dir() -> Option<PathBuf> {
    if let Some(p) = env::var_os("READSTAT_SRC") {
        let p = PathBuf::from(p);
        if p.join("src/readstat.h").exists() {
            return Some(p);
        }
    }

    let mut cur = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    for _ in 0..6 {
        let third_party = cur.join("native/readstat-sys/third_party/readstat");
        if third_party.join("src/readstat.h").exists() {
            return Some(third_party);
        }
        let readstat_top = cur.join("ReadStat");
        if readstat_top.join("src/readstat.h").exists() {
            return Some(readstat_top);
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

// --- zlib detection / configuration -----------------------------------------

fn link_static_z_from_dir(dir: &Path) {
    println!("cargo:rustc-link-search=native={}", dir.display());
    #[cfg(target_os = "windows")]
    {
        let z = dir.join("z.lib");
        let zstatic = dir.join("zlibstatic.lib");
        if zstatic.exists() {
            println!("cargo:rustc-link-lib=static=zlibstatic");
        } else if z.exists() {
            println!("cargo:rustc-link-lib=static=z");
        } else {
            println!("cargo:rustc-link-lib=static=z");
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        println!("cargo:rustc-link-lib=static=z");
    }
}

/// Configure zlib; enable ONLY if we have headers/paths.
/// Prefer DEP_Z_* (from libz-sys dependency), then pkg-config, then sysroot probe.
fn configure_zlib(build: &mut cc::Build) -> bool {
    if let Ok(v) = env::var("READSTAT_WITH_ZLIB") {
        let on = v != "0";
        if on {
            println!("cargo:rustc-link-lib=z");
        }
        diag!("READSTAT_WITH_ZLIB override -> {}", if on { "ON" } else { "OFF" });
        return on;
    }

    // 0) libz-sys (target) exports
    let dep_z_include = env::var("DEP_Z_INCLUDE").ok();
    let dep_z_root = env::var("DEP_Z_ROOT").ok();
    let dep_z_lib = env::var("DEP_Z_LIB").ok();
    if dep_z_include.is_some() || dep_z_root.is_some() || dep_z_lib.is_some() {
        if let Some(inc) = dep_z_include.as_deref() {
            diag!("Using zlib headers from DEP_Z_INCLUDE={inc}");
            build.include(inc);
        }
        if let Some(lib) = dep_z_lib.as_deref() {
            diag!("Using zlib lib dir from DEP_Z_LIB={lib}");
            link_static_z_from_dir(Path::new(lib));
        } else if let Some(root) = dep_z_root.as_deref() {
            for cand in ["lib", "lib64", ""].iter() {
                let p = Path::new(root).join(cand);
                if p.exists() {
                    diag!("Using zlib lib dir {}", p.display());
                    link_static_z_from_dir(&p);
                    break;
                }
            }
            if dep_z_include.is_none() {
                let inc = Path::new(root).join("include");
                if inc.exists() {
                    diag!("Using zlib headers from {}", inc.display());
                    build.include(inc);
                }
            }
        }
        return true;
    }

    // 1) pkg-config (native)
    if let Ok(lib) = pkg_config::Config::new().env_metadata(true).probe("zlib") {
        diag!("Found zlib via pkg-config");
        for p in &lib.include_paths {
            diag!("  zlib include: {}", p.display());
            build.include(p);
        }
        for p in &lib.link_paths {
            diag!("  zlib link:    {}", p.display());
            println!("cargo:rustc-link-search=native={}", p.display());
        }
        println!("cargo:rustc-link-lib=z");
        return true;
    }

    // 2) sysroot probe (cross or native): only enable if header actually exists
    let target = env::var("TARGET").unwrap_or_default();
    let _host = env::var("HOST").unwrap_or_default();
    if let Some(sysroot) = detect_sysroot_for_target(&target) {
        let base = sysroot.join("usr/include");
        let trip = if target.starts_with("aarch64") {
            base.join("aarch64-linux-gnu")
        } else if target.starts_with("x86_64") {
            base.join("x86_64-linux-gnu")
        } else {
            PathBuf::new()
        };
        let candidates = [
            base.join("zlib.h"),
            if trip.as_os_str().is_empty() { PathBuf::new() } else { trip.join("zlib.h") },
        ];
        if candidates.iter().any(|p| p.exists()) {
            if base.exists() { build.include(&base); }
            if !trip.as_os_str().is_empty() && trip.exists() { build.include(&trip); }
            println!("cargo:rustc-link-lib=z");
            diag!("Using zlib from sysroot {}", sysroot.display());
            return true;
        }
    }

    // No headers => do NOT pretend zlib exists
    diag!("zlib headers not found; building WITHOUT zlib (.zsav disabled)");
    false
}

// --- build vendored ReadStat -------------------------------------------------

fn build_vendored(rs_dir: &Path) {
    let src_dir = rs_dir.join("src");
    let inc_dir = rs_dir.join("src");
    assert!(
        src_dir.exists(),
        "ReadStat sources not found at {}",
        src_dir.display()
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let mut build = cc::Build::new();

    // diagnostics
    let target = env::var("TARGET").unwrap_or_default();
    let host = env::var("HOST").unwrap_or_default();
    dump_env(&[
        "READSTAT_BUILD_DEBUG",
        "TARGET",
        "HOST",
        "CFLAGS",
        "LIBCLANG_PATH",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_SYSROOT_DIR",
        "PKG_CONFIG_LIBDIR",
        "LIBRARY_PATH",
        "CPATH",
        "READSTAT_SRC",
        "READSTAT_WITH_ZLIB",
        "PKG_CONFIG",
        "PKG_CONFIG_ALLOW_CROSS",
        "DEP_Z_INCLUDE",
        "DEP_Z_ROOT",
        "DEP_Z_LIB",
        "BINDGEN_SYSROOT",
    ]);
    diag!("Using ReadStat sources at {}", rs_dir.display());
    diag!("Compiling for TARGET={target} (host={host})");

    // OUT_DIR first so our stubs win where needed
    build.include(&out_dir);
    build.include(&inc_dir);
    build.include(&rs_dir);

    // Minimal config.h
    build.define("HAVE_STDDEF_H", Some("1"));
    build.define("HAVE_STDINT_H", Some("1"));
    build.define("HAVE_INTTYPES_H", Some("1"));
    build.define("HAVE_STDLIB_H", Some("1"));
    build.define("HAVE_STRING_H", Some("1"));
    build.define("HAVE_STRINGS_H", Some("1"));

    // zlib detection
    let has_zlib = configure_zlib(&mut build);
    if has_zlib {
        build.define("READSTAT_HAVE_ZLIB", Some("1"));
        build.define("HAVE_ZLIB", Some("1"));
        diag!("Building WITH zlib support");
    } else {
        build.define("READSTAT_HAVE_ZLIB", Some("0"));
        build.define("HAVE_ZLIB", Some("0"));
        diag!("Building WITHOUT zlib support (.zsav disabled)");
    }

    // iconv / Windows stubs
    if cfg!(target_os = "windows") {
        build.define("HAVE_ICONV", Some("0"));
        let stub_h = out_dir.join("iconv.h");
        fs::write(&stub_h, r#"
#ifndef ICONV_STUB_H
#define ICONV_STUB_H
#include <stddef.h>
typedef void* iconv_t;
#define ICONV_CONST const
#endif
"#).expect("write iconv.h stub");

        let stub_c = out_dir.join("posix_stubs.c");
        fs::write(&stub_c, r#"
#include <stddef.h>
#include <errno.h>

typedef void* iconv_t;

iconv_t iconv_open(const char* tocode, const char* fromcode) {
    (void)tocode; (void)fromcode; return (iconv_t)-1;
}
int iconv_close(iconv_t cd) { (void)cd; return 0; }
size_t iconv(iconv_t cd, const char** inbuf, size_t* inbytesleft,
             char** outbuf, size_t* outbytesleft) {
    (void)cd; (void)inbuf; (void)inbytesleft; (void)outbuf; (void)outbytesleft;
    errno = EINVAL; return (size_t)-1;
}
// Fallback for Windows I/O init (real symbol lives in readstat_io_win.c)
typedef struct readstat_io_s readstat_io_t;
readstat_io_t* unistd_io_init(void) { return NULL; }
"#).expect("write posix_stubs.c");

        build.file(&stub_c);
        println!("cargo:rerun-if-changed={}", stub_c.display());
        diag!("Created POSIX stubs for Windows");
    } else {
        build.define("HAVE_ICONV", Some("1"));
        #[cfg(target_os = "macos")]
        {
            println!("cargo:rustc-link-lib=iconv");
            diag!("Linking with -liconv (macOS)");
        }
    }

    // Force-include readstat.h
    if cfg!(target_env = "msvc") {
        build.flag("/FIreadstat.h");
    } else {
        build.flag("-include").flag("readstat.h");
    }

    // Collect C files
    let mut files: Vec<PathBuf> = Vec::new();
    let mut skipped_files: Vec<String> = Vec::new();

    for entry in walkdir::WalkDir::new(&src_dir) {
        let entry = entry.unwrap();
        let p = entry.path();
        if !entry.file_type().is_file() { continue; }
        if p.extension().and_then(|s| s.to_str()) != Some("c") { continue; }

        let rel = p.strip_prefix(&src_dir).unwrap();

        let skip_dir = rel.components().any(|c| {
            matches!(c.as_os_str().to_str(), Some("bin" | "fuzz" | "test" | "tests" | "txt"))
        });
        if skip_dir { continue; }

        let name = rel.file_name().and_then(|s| s.to_str()).unwrap_or("");

        if cfg!(target_os = "windows") {
            if name == "readstat_io_unistd.c" { skipped_files.push(format!("{name} (Unix I/O)")); continue; }
        } else if name == "readstat_io_win.c" {
            skipped_files.push(format!("{name} (Windows I/O)")); continue;
        }

        if !has_zlib && (name == "readstat_zsav_compress.c"
                      || name == "readstat_zsav_read.c"
                      || name == "readstat_zsav_write.c") {
            skipped_files.push(format!("{name} (requires zlib)"));
            continue;
        }

        files.push(p.to_path_buf());
    }

    if !skipped_files.is_empty() {
        diag!("Skipping {} files:", skipped_files.len());
        for f in &skipped_files { diag!("  - {}", f); }
    }
    diag!("Compiling {} C source files", files.len());

    for f in &files {
        println!("cargo:rerun-if-changed={}", f.display());
        build.file(f);
    }

    build.define("READSTAT_VERSION", Some("\"vendored\""));
    build.warnings(false).compile("readstat");

    // Generate Rust bindings
    bindgen_with_includes(&inc_dir);
}

// --- non-vendored link paths -------------------------------------------------

fn link_from_prefix(prefix: &str) {
    println!("cargo:rustc-link-search=native={prefix}/lib");
    println!("cargo:rustc-link-lib=readstat");
    println!("cargo:rustc-link-lib=z");
    println!("cargo:include={prefix}/include");
    bindgen_with_includes(&PathBuf::from(format!("{prefix}/include")));
}

fn link_from_pkg_config() -> bool {
    match pkg_config::Config::new().probe("readstat") {
        Ok(lib) => {
            if let Some(inc) = lib.include_paths.get(0) {
                bindgen_with_includes(inc);
            } else {
                bindgen_with_includes(Path::new("."));
            }
            true
        }
        Err(_) => false,
    }
}

// --- main -------------------------------------------------------------------

fn main() {
    // Re-run on env changes
    for k in [
        "READSTAT_BUILD_DEBUG","READSTAT_SRC","READSTAT_WITH_ZLIB","READSTAT_PREFIX",
        "PKG_CONFIG","PKG_CONFIG_PATH","PKG_CONFIG_SYSROOT_DIR","PKG_CONFIG_LIBDIR",
        "LIBCLANG_PATH","PKG_CONFIG_ALLOW_CROSS","DEP_Z_INCLUDE","DEP_Z_ROOT",
        "DEP_Z_LIB","BINDGEN_SYSROOT","ZLIB_NO_PKG_CONFIG"
    ] {
        println!("cargo:rerun-if-env-changed={k}");
    }

    if cfg!(feature = "vendored") {
        if let Some(dir) = find_readstat_dir() {
            build_vendored(&dir);
            return;
        }
        panic!(
            "`vendored` enabled but could not find ReadStat sources.\n\
             - Set READSTAT_SRC to the ReadStat directory, or\n\
             - Add a submodule at native/readstat-sys/third_party/readstat, or\n\
             - Add ./ReadStat at the repository root."
        );
    }

    if link_from_pkg_config() { return; }
    if let Ok(prefix) = env::var("READSTAT_PREFIX") { link_from_prefix(&prefix); return; }
    if let Ok(home) = env::var("HOME") { link_from_prefix(&format!("{home}/.local")); return; }

    panic!(
        "Unable to locate ReadStat.\n\
         Options:\n\
         - Enable feature `vendored` (and include the ReadStat sources), or\n\
         - Install `libreadstat` and expose it via pkg-config, or\n\
         - Set READSTAT_PREFIX to a prefix containing include/ and lib/."
    );
}
