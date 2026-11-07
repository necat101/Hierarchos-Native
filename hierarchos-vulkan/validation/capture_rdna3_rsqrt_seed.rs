//! Capture the host's RSQRTSS instruction, independently of any model fixture.
//! rustc -O validation/capture_rdna3_rsqrt_seed.rs -o target/capture-rsqrt.exe
//! target/capture-rsqrt.exe shaders/rdna3_zen4_rsqrt_seed.bin
//! The qualified MKL HA sqrt uses this seed and two unfused corrections.
//! Exhaustively checks all 2^24 mantissas in [0.5,2), not training values.
use std::{arch::x86_64::*, io::Write};
fn seed(x: u32) -> u32 {
    unsafe { _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(f32::from_bits(x)))) }.to_bits()
}
fn main() {
    let path = std::env::args().nth(1).expect("output path");
    let mut table = Vec::new();
    for exponent in [126u32, 127] {
        for bucket in 0..8192u32 {
            let bits = (exponent << 23) | (bucket << 10);
            let value = seed(bits);
            assert_eq!(value & 2047, 0);
            assert_eq!(value >> 23, if exponent == 126 { 127 } else { 126 });
            for offset in 0..1024 {
                assert_eq!(seed(bits + offset), value, "nonconstant bucket");
            }
            table.push(((value & 0x7fffff) >> 11) as u16);
        }
    }
    // Independently check power-of-two scaling across ordinary FP32 exponents.
    for exponent in 8..247u32 {
        for mantissa in [0, 1, 1023, 1024, 0x123456, 0x654321, 0x7fffff] {
            let normalized = 126 + (exponent & 1);
            let index = ((exponent & 1) * 8192 + (mantissa >> 10)) as usize;
            let base = (if normalized == 126 { 127 } else { 126 }) << 23;
            let adjusted = ((base | (u32::from(table[index]) << 11)) as i64
                + ((normalized as i64 - exponent as i64) / 2) * (1 << 23))
                as u32;
            assert_eq!(seed((exponent << 23) | mantissa), adjusted);
        }
    }
    let mut file = std::fs::File::create(path).unwrap();
    for value in table {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    let id = unsafe { __cpuid(1) };
    println!(
        "RSQRT exhaustive capture passed; CPUID.1:EAX={:08x}; 32768 bytes",
        id.eax
    );
}
