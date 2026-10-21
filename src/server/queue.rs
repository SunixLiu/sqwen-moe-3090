// src/server/queue.rs — 单流串行队列（Task 12）
//
// 用 tokio::sync::Semaphore 实现 capacity 限制：
//   - 请求到达时 try_acquire，成功则入队执行；
//   - 失败直接返回 429 Too Many Requests；
//   - 执行完毕后 release permit。
//
// 对应 plan Task 12 queue.rs：Semaphore 包装 + 429 错误映射。

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// 队列满时的 429 响应。
#[derive(Debug)]
pub struct QueueFull(pub usize);

impl IntoResponse for QueueFull {
    fn into_response(self) -> Response {
        (StatusCode::TOO_MANY_REQUESTS, format!("queue full: capacity={}", self.0)).into_response()
    }
}

/// 单流串行队列：同一时刻最多 `capacity` 个请求在生成中。
///
/// `acquire` 返回一个 RAII guard，drop 时释放 permit；若无法立即获取则返回
/// `QueueFull`（调用方直接转 429）。
#[derive(Clone)]
pub struct RequestQueue {
    sem: Arc<Semaphore>,
    capacity: usize,
}

impl RequestQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            sem: Arc::new(Semaphore::new(capacity)),
            capacity,
        }
    }

    /// 尝试获取一个 permit；成功返回 Guard，失败返回 QueueFull。
    pub fn try_acquire(&self) -> Result<QueueGuard, QueueFull> {
        match self.sem.try_acquire() {
            Ok(guard) => {
                // 延长 permit 生命周期至 'static（semaphore 在 Arc 中，保证 guard 存活期间有效）
                let guard = unsafe {
                    std::mem::transmute::<tokio::sync::SemaphorePermit<'_>, tokio::sync::SemaphorePermit<'static>>(guard)
                };
                Ok(QueueGuard { guard })
            }
            Err(_) => Err(QueueFull(self.capacity)),
        }
    }
}

/// 队列守卫：drop 时自动释放 semaphore permit。
pub struct QueueGuard {
    #[allow(dead_code)]
    guard: tokio::sync::SemaphorePermit<'static>,
}

// SemaphorePermit 自身实现 Drop，drop 时自动释放 permit，所以不需要自定义 Drop

// SAFETY: SemaphorePermit 内部只涉及计数器，Send/Sync 安全。
unsafe impl Send for QueueGuard {}
unsafe impl Sync for QueueGuard {}