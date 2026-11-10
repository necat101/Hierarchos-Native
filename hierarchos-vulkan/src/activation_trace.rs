//! Observation-only activation tracing for the resident Transformer graph.
//!
//! Tracing is disabled unless `HIERARCHOS_ACTIVATION_TRACE=1` is present. When
//! enabled, each tap records one `copy_f32` of the *materialized* production
//! buffer into a staging buffer inside the same command batch. The graphs'
//! arithmetic, dispatch order and buffer contents are unchanged; the only
//! additions are copies that the caller reads after submission.
//!
//! The tracer deliberately lives outside the graph structs: a thread-local
//! session keeps `VulkanTransformer` free of diagnostic state, and the process
//! mode is read from the existing `HIERARCHOS_WMMA_MODE` switch so a capture
//! cannot claim a precision the graph did not execute.
//!
//! `finish` reads the staging buffers and stashes a [`TraceBundle`]; `take`
//! removes it. The harness runs one forward per process, so a bundle describes
//! exactly one batch.

use std::cell::RefCell;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

use crate::vulkan::{ComputeBatch, GpuBuffer, VulkanDevice};
use crate::wmma::WmmaMode;

/// One captured activation with its exact identity.
#[derive(Clone, Debug)]
pub struct TraceTensor {
    pub layer: usize,
    pub operation: String,
    /// `[batch, sequence, width]` for every tap in this phase.
    pub shape: [usize; 3],
    pub values: Vec<f32>,
}

/// All activations captured by one forward pass.
#[derive(Clone, Debug)]
pub struct TraceBundle {
    pub precision: String,
    pub tensors: Vec<TraceTensor>,
}

impl TraceBundle {
    /// SHA-256 over the tensor name, shape and the little-endian bits of every
    /// value. The harness records this digest next to the raw capture so a
    /// committed summary can be checked against the retained bytes.
    pub fn digest(&self) -> String {
        let mut hash = Sha256::new();
        for tensor in &self.tensors {
            hash.update((tensor.operation.len() as u64).to_le_bytes());
            hash.update(tensor.operation.as_bytes());
            hash.update((tensor.layer as u64).to_le_bytes());
            for extent in tensor.shape {
                hash.update((extent as u64).to_le_bytes());
            }
            for value in &tensor.values {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        format!("{:x}", hash.finalize())
    }
}

struct Pending {
    layer: usize,
    operation: &'static str,
    shape: [usize; 3],
    buffer: GpuBuffer,
    len: usize,
}

struct Session {
    device: VulkanDevice,
    pending: Vec<Pending>,
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
    static BUNDLE: RefCell<Option<TraceBundle>> = const { RefCell::new(None) };
}

/// Enable tracing for this thread when the opt-in environment variable is set.
///
/// Returns true if this call started a session. A nested forward call while a
/// session is already active leaves the existing session untouched.
pub fn begin(device: &VulkanDevice) -> bool {
    if std::env::var("HIERARCHOS_ACTIVATION_TRACE").ok().as_deref() != Some("1") {
        return false;
    }
    SESSION.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return false;
        }
        *slot = Some(Session {
            device: device.clone(),
            pending: Vec::new(),
        });
        true
    })
}

pub fn active() -> bool {
    SESSION.with(|slot| slot.borrow().is_some())
}

/// Copy `buffer[..len]` into the trace staging area.
///
/// The copy is recorded in the caller's batch, so it observes exactly the
/// values the production path produced at this point. A no-op when tracing is
/// disabled, which is the default and the qualified training path.
pub fn tap(
    commands: &mut ComputeBatch,
    buffer: &GpuBuffer,
    len: usize,
    layer: usize,
    operation: &'static str,
    shape: [usize; 3],
) -> Result<()> {
    let staging = SESSION.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(session) = slot.as_mut() else {
            return Ok::<Option<()>, anyhow::Error>(None);
        };
        let staging = GpuBuffer::zeros_f32(&session.device, len)
            .with_context(|| format!("allocate trace staging for {operation}"))?;
        commands
            .copy_f32(buffer, &staging, len)
            .with_context(|| format!("record trace copy for {operation}"))?;
        session.pending.push(Pending {
            layer,
            operation,
            shape,
            buffer: staging,
            len,
        });
        Ok(Some(()))
    })?;
    let _ = staging;
    Ok(())
}

/// Read every captured staging buffer and stash the bundle.
///
/// Must run after `commands.submit()`. The session ends here so a later
/// forward in the same process does not mix activations from two batches.
pub fn finish(device: &VulkanDevice) -> Result<()> {
    let session = SESSION.with(|slot| slot.borrow_mut().take());
    let Some(session) = session else {
        return Ok(());
    };
    let _ = device;
    let mut tensors = Vec::with_capacity(session.pending.len());
    for pending in session.pending {
        let values = pending
            .buffer
            .read_f32(pending.len)
            .with_context(|| format!("read trace capture {}", pending.operation))?;
        tensors.push(TraceTensor {
            layer: pending.layer,
            operation: pending.operation.to_owned(),
            shape: pending.shape,
            values,
        });
    }
    let precision = match WmmaMode::from_env() {
        Ok(WmmaMode::Mixed) => "mixed",
        Ok(WmmaMode::FullFp16) => "full-fp16",
        _ => "fp32",
    }
    .to_owned();
    BUNDLE.with(|slot| {
        *slot.borrow_mut() = Some(TraceBundle { precision, tensors });
    });
    Ok(())
}

/// Remove and return the last completed bundle.
pub fn take() -> Option<TraceBundle> {
    BUNDLE.with(|slot| slot.borrow_mut().take())
}
