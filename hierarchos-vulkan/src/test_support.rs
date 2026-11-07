//! Process-wide test coordination for environment-variable overrides.
//!
//! Several parity tests temporarily change process environment switches that
//! select Vulkan kernel routes. Rust runs tests in parallel by default, so a
//! reader can otherwise observe another test's temporary switch and silently
//! execute a different numerical path. The affected mutators and readers hold
//! this lock for the duration of the switch; unrelated tests keep running in
//! parallel.

use std::sync::{Mutex, MutexGuard, OnceLock};

static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
