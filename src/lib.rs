// sqwen — Qwen3.6-35B-A3B (qwen35moe) 特化 GGUF 推理引擎

// 链接预构建的 ggml CUDA 库（import libraries for DLLs in build-ggml/bin/Release）
#[link(name = "ggml-base")]
#[link(name = "ggml-cpu")]
#[link(name = "ggml-cuda")]
#[link(name = "ggml")]
#[link(name = "cudart")]
extern "C" {}

pub mod backend;
pub mod model;
pub mod engine;
pub mod template;
pub mod server;
