// build.rs — Cargo 构建脚本：链接预构建的 ggml CUDA 静态库 + cudart。
//
// 依赖 build_ggml.ps1 预先将产物复制到（相对仓库根目录）：
//   build-ggml\lib\Release\*.lib   （链接用）
//   build-ggml\bin\Release\*.dll   （运行时用）

use std::env;
use std::path::PathBuf;

fn main() {
    eprintln!("DEBUG: build.rs running");
    eprintln!("DEBUG: CARGO_TARGET_DIR={:?}", env::var("CARGO_TARGET_DIR"));
    // **项目根就是 crate 根**：`Cargo.toml` 与 `build-ggml/`、`reference/` 同级，
    // 都在仓库根目录下。所以直接用 `CARGO_MANIFEST_DIR`。
    //
    // 这里原来写的是 `manifest_path.parent()`（= 仓库的**上一级目录**），指向了仓库**外面**，
    // 于是 `bin_dir` 不存在 → `read_dir` 失败 → 而下面用 `.ok()` 把错误吞了 →
    // **DLL 一个都没复制**，运行时直接 STATUS_DLL_NOT_FOUND (0xC0000135)。
    // 一直没暴露是因为 worktree 的 target/ 里早就有拷好的 DLL；换到干净 target
    // 全量重建才现形。
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_root = PathBuf::from(&manifest_dir);

    // 优先使用 build_ggml.ps1 复制后的本地副本；否则回退到 llama.cpp 原始构建目录
    let local_lib_dir = project_root.join("build-ggml").join("lib").join("Release");
    let llama_cpp_lib_dir = project_root.join("reference").join("llama.cpp").join("build").join("ggml").join("src").join("Release");
    let lib_dir = if local_lib_dir.exists() { &local_lib_dir } else { &llama_cpp_lib_dir };

    println!(
        "cargo:rustc-link-search=native={}",
        lib_dir.display()
    );

    // 注意：Cargo 1.98.1 会丢弃 build.rs 中的 rustc-link-lib 指令，
    // 链接通过 src/lib.rs 中的 #[link] 属性完成。
    // 这里仅保留搜索路径和 DLL 复制逻辑。

    // 验证产物存在性（编译时给出友好提示）
    let check_libs = ["ggml", "ggml-base", "ggml-cpu", "ggml-cuda", "cudart"];
    for lib in &check_libs {
        let path = lib_dir.join(format!("{}.lib", lib));
        if !path.exists() {
            eprintln!(
                "WARNING: {} not found at {:?}. Run build_ggml.ps1 first.",
                lib, path
            );
        }
    }

    // 运行时：将 DLL 复制到 target 输出目录（方便调试/运行）
    let bin_dir = project_root.join("build-ggml").join("bin").join("Release");
    let out_dir = env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| "target".to_string());
    let out_path = PathBuf::from(out_dir);

    // 复制 DLL 到 debug/deps 和 release/deps 目录，并**同时**放到 target/<profile>/ 下 ——
    // 可执行文件在 target/<profile>/<name>.exe，Windows 只从 exe 所在目录（和 PATH）
    // 找依赖 DLL，只拷 deps/ 会让 target/release/sqwen.exe 直接 STATUS_DLL_NOT_FOUND (0xC0000135)。
    for profile in ["debug", "release"] {
        let profile_dir = out_path.join(profile);
        let deps_dir = profile_dir.join("deps");
        std::fs::create_dir_all(&deps_dir).ok();
        if let Ok(entries) = std::fs::read_dir(&bin_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().map_or(false, |e| e == "dll") {
                    for dst in [deps_dir.join(entry.file_name()), profile_dir.join(entry.file_name())] {
                        println!(
                            "cargo:warning=copying {} -> {:?}",
                            entry.file_name().display(),
                            dst
                        );
                        std::fs::copy(&entry.path(), &dst).ok();
                    }
                }
            }
        }
    }
}
