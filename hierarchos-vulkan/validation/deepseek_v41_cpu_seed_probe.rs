//! Development-only SSE reciprocal-square-root instruction observer.
//! The product runtime uses the frozen numeric table on Vulkan.
use std::arch::x86_64::*;

#[no_mangle]
pub unsafe extern "C" fn rsqrt_seeds(input: *const f32, output: *mut f32, count: usize) {
    for i in 0..count {
        let value = _mm_set_ss(*input.add(i));
        *output.add(i) = _mm_cvtss_f32(_mm_rsqrt_ss(value));
    }
}
