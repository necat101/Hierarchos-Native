use super::*;
use crate::deepseek_v41::indexer::LinearKernel;
use crate::deepseek_v41::peft::DeepseekV41LoraHost;
use crate::deepseek_v41::{
    DeepseekV41LinearWeights, DeepseekV41MixedLinearKernel, DeepseekV41WeightOutput,
};
use crate::{GpuBuffer, VulkanDevice};
use std::path::PathBuf;

#[derive(Deserialize)]
struct Fixture {
    tolerance: f64,
    rtol: f64,
    comparison: String,
    units: Vec<Unit>,
    mixed_moe: MoeTrajectory,
}
#[derive(Deserialize)]
struct Module {
    target: String,
    rank: usize,
    width: usize,
    out: usize,
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct ProbeSpec {
    rows: usize,
    width: usize,
    out: usize,
    vocab: usize,
    x_bits: Vec<u32>,
    base_bits: Vec<u32>,
    head_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Unit {
    name: String,
    hyper: DeepseekV41AdamWConfig,
    modules: Vec<Module>,
    adapter_config: serde_json::Value,
    probe: ProbeSpec,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
pub(crate) struct GradientBits {
    pub a_bits: Option<Vec<u32>>,
    pub b_bits: Option<Vec<u32>>,
}
#[derive(Deserialize)]
pub(crate) struct State {
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
    exp_avg_a_bits: Vec<u32>,
    exp_avg_sq_a_bits: Vec<u32>,
    exp_avg_b_bits: Vec<u32>,
    exp_avg_sq_b_bits: Vec<u32>,
    step_a: u64,
    step_b: u64,
}
#[derive(Deserialize)]
pub(crate) struct Step {
    pub gradients: BTreeMap<String, GradientBits>,
    pub states: BTreeMap<String, State>,
    #[serde(default)]
    logits_bits: Vec<u32>,
    #[serde(default)]
    pub output_bits: Vec<u32>,
    #[serde(default)]
    pub updated_output_bits: Vec<u32>,
}
#[derive(Deserialize)]
pub(crate) struct MoeTrajectory {
    pub parent_sha256: String,
    pub case_index: usize,
    pub name: String,
    pub hyper: DeepseekV41AdamWConfig,
    pub steps: Vec<Step>,
}
fn fixture() -> Fixture {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_conditional_adamw.json"
    ))
    .unwrap();
    assert_eq!(f.tolerance, 2e-7);
    assert_eq!(f.rtol, 0.0);
    assert_eq!(f.comparison, "strict_less_than");
    f
}
pub(crate) fn mixed_fixture() -> MoeTrajectory {
    fixture().mixed_moe
}
fn values(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
}
pub(crate) fn gate(label: &str, actual: &[f32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut worst = 0.0f64;
    let mut at = 0;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let b = f32::from_bits(b);
        assert!(a.is_finite() && b.is_finite(), "{label} non-finite");
        let delta = (a as f64 - b as f64).abs();
        if delta > worst {
            worst = delta;
            at = i;
        }
    }
    eprintln!(
        "V4.1 conditional AdamW {label}: {} values, worst {worst:.17e}",
        actual.len()
    );
    assert!(
        worst < 2e-7,
        "{label}: {worst:.17e} at {at}, native={} CPU={}",
        actual[at],
        f32::from_bits(expected[at])
    );
}
fn bits_equal(a: &[f32], b: &[f32]) {
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
pub(crate) fn assert_exact(a: &DeepseekV41AdapterTrainer, b: &DeepseekV41AdapterTrainer) {
    assert_eq!(a.step_count(), b.step_count());
    assert_eq!(a.config(), b.config());
    assert_eq!(
        a.targets().collect::<Vec<_>>(),
        b.targets().collect::<Vec<_>>()
    );
    for target in a.targets() {
        let (la, lb) = a.factors(target).unwrap();
        let (ra, rb) = b.factors(target).unwrap();
        bits_equal(la, ra);
        bits_equal(lb, rb);
        let (ma, va, mb, vb) = a.moments(target).unwrap();
        let (rma, rva, rmb, rvb) = b.moments(target).unwrap();
        for (left, right) in [(ma, rma), (va, rva), (mb, rmb), (vb, rvb)] {
            bits_equal(left, right);
        }
        assert_eq!(
            a.parameter_step_counts(target),
            b.parameter_step_counts(target)
        );
    }
}
pub(crate) fn compare_state(
    label: &str,
    trainer: &DeepseekV41AdapterTrainer,
    states: &BTreeMap<String, State>,
) {
    assert_eq!(trainer.targets().count(), states.len());
    for (target, s) in states {
        let (a, b) = trainer.factors(target).unwrap();
        let (ma, va, mb, vb) = trainer.moments(target).unwrap();
        for (name, actual, expected) in [
            ("A", a, &s.a_bits),
            ("B", b, &s.b_bits),
            ("mA", ma, &s.exp_avg_a_bits),
            ("vA", va, &s.exp_avg_sq_a_bits),
            ("mB", mb, &s.exp_avg_b_bits),
            ("vB", vb, &s.exp_avg_sq_b_bits),
        ] {
            if actual.is_empty() {
                assert!(name.starts_with('m') || name.starts_with('v'));
                assert!(expected.iter().all(|&bits| f32::from_bits(bits) == 0.0));
                continue;
            }
            gate(&format!("{label} {target} {name}"), actual, expected);
        }
        assert_eq!(
            trainer.parameter_step_counts(target),
            Some((s.step_a, s.step_b)),
            "{label} {target} counters"
        );
    }
}
fn apply(trainer: &mut DeepseekV41AdapterTrainer, step: &Step) -> Result<()> {
    let before = trainer.clone();
    if step
        .gradients
        .values()
        .all(|g| g.a_bits.is_some() && g.b_bits.is_some())
    {
        let pairs = step
            .gradients
            .iter()
            .map(|(target, g)| DeepseekV41AdapterGradient {
                target: target.clone(),
                grad_a: values(g.a_bits.as_ref().unwrap()),
                grad_b: values(g.b_bits.as_ref().unwrap()),
            })
            .collect::<Vec<_>>();
        trainer.step(&pairs)?;
    } else {
        let gradients = step
            .gradients
            .iter()
            .map(|(target, g)| DeepseekV41AdapterParameterGradient {
                target: target.clone(),
                grad_a: g.a_bits.as_deref().map(values),
                grad_b: g.b_bits.as_deref().map(values),
            })
            .collect::<Vec<_>>();
        trainer.step_parameters(&gradients)?;
    }
    for target in before.targets() {
        let g = step.gradients.get(target);
        let counts = before.parameter_step_counts(target).unwrap();
        let after = trainer.parameter_step_counts(target).unwrap();
        let (a, b) = before.factors(target).unwrap();
        let (na, nb) = trainer.factors(target).unwrap();
        let (ma, va, mb, vb) = before.moments(target).unwrap();
        let (nma, nva, nmb, nvb) = trainer.moments(target).unwrap();
        if g.is_none_or(|g| g.a_bits.is_none()) {
            bits_equal(a, na);
            bits_equal(ma, nma);
            bits_equal(va, nva);
            assert_eq!(counts.0, after.0);
        }
        if g.is_none_or(|g| g.b_bits.is_none()) {
            bits_equal(b, nb);
            bits_equal(mb, nmb);
            bits_equal(vb, nvb);
            assert_eq!(counts.1, after.1);
        }
    }
    Ok(())
}

struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
impl Temp {
    fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "hierarchos-v41-conditional-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
fn adapter(c: &Unit, path: &Path) -> Result<DeepseekV41LoraAdapter> {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    fs::create_dir(path)?;
    fs::write(
        path.join("adapter_config.json"),
        serde_json::to_vec_pretty(&c.adapter_config)?,
    )?;
    let mut owned = Vec::new();
    for m in &c.modules {
        for (name, shape, bits) in [
            ("A", vec![m.rank, m.width], &m.a_bits),
            ("B", vec![m.out, m.rank], &m.b_bits),
        ] {
            owned.push((
                format!("base_model.model.{}.lora_{name}.weight", m.target),
                shape,
                bits.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ));
        }
    }
    let views = owned
        .iter()
        .map(|(name, shape, data)| {
            Ok((
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), data)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(views, None, &path.join("adapter_model.safetensors"))?;
    DeepseekV41LoraAdapter::load(path)
}
struct Probe {
    device: VulkanDevice,
    mixed: DeepseekV41MixedLinearKernel,
    linear: LinearKernel,
}
impl Probe {
    fn new(device: &VulkanDevice) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            mixed: DeepseekV41MixedLinearKernel::new(device.clone())?,
            linear: LinearKernel::new(device)?,
        })
    }
    fn logits(&self, c: &Unit, trainer: &DeepseekV41AdapterTrainer) -> Result<Vec<f32>> {
        let p = &c.probe;
        let adapter = trainer.adapter()?;
        let module = adapter
            .module_for_tensor(&format!("{}.weight", c.modules[0].target))
            .unwrap();
        let w = DeepseekV41LinearWeights::Fp32 {
            rows: p.out,
            cols: p.width,
            values: values(&p.base_bits),
            lora: Some(DeepseekV41LoraHost::from_module(module)),
        }
        .upload(&self.device)?;
        let x = GpuBuffer::from_u32(&self.device, &p.x_bits)?;
        let head = GpuBuffer::from_u32(&self.device, &p.head_bits)?;
        let h = self
            .mixed
            .apply(&x, p.rows, &w, DeepseekV41WeightOutput::Fp32)?;
        self.linear
            .apply(&self.device, &h, &head, p.rows, p.out, p.vocab)?
            .read_f32(p.rows * p.vocab)
    }
}

#[test]
fn conditional_adamw_none_zero_and_independent_factor_steps_match_cpu() -> Result<()> {
    let root = Temp::new()?;
    let device = VulkanDevice::new()?;
    let probe = Probe::new(&device)?;
    for c in fixture().units {
        let initial = adapter(&c, &root.0.join(&c.name))?;
        let mut trainer = DeepseekV41AdapterTrainer::new(&initial, c.hyper)?;
        for (i, step) in c.steps.iter().enumerate() {
            apply(&mut trainer, step)?;
            assert_eq!(trainer.step_count(), i as u64 + 1);
            compare_state(&format!("{} step {i}", c.name), &trainer, &step.states);
            gate(
                &format!("{} step {i} logits", c.name),
                &probe.logits(&c, &trainer)?,
                &step.logits_bits,
            );
        }
        // Explicit None is the same skip as an omitted target.
        let mut none = trainer.clone();
        none.step_parameters(&[DeepseekV41AdapterParameterGradient {
            target: c.modules[0].target.clone(),
            grad_a: None,
            grad_b: None,
        }])?;
        assert_eq!(none.step_count(), trainer.step_count() + 1);
        none.step = trainer.step;
        assert_exact(&none, &trainer);
    }
    Ok(())
}

#[test]
fn conditional_adamw_checkpoint_resume_and_export_preserve_factor_clocks() -> Result<()> {
    let root = Temp::new()?;
    let device = VulkanDevice::new()?;
    let probe = Probe::new(&device)?;
    for c in fixture().units {
        let initial = adapter(&c, &root.0.join(&c.name))?;
        let mut uninterrupted = DeepseekV41AdapterTrainer::new(&initial, c.hyper)?;
        for step in &c.steps {
            apply(&mut uninterrupted, step)?;
        }
        for boundary in [0, 3, 6] {
            let mut partial = DeepseekV41AdapterTrainer::new(&initial, c.hyper)?;
            for step in &c.steps[..boundary] {
                apply(&mut partial, step)?;
            }
            let directory = root.0.join(format!("{}-checkpoint-{boundary}", c.name));
            partial.save_checkpoint(&directory)?;
            let mut resumed = DeepseekV41AdapterTrainer::resume_from_checkpoint(&directory)?;
            assert_exact(&partial, &resumed);
            let metadata: serde_json::Value =
                serde_json::from_slice(&fs::read(directory.join(OPTIMIZER_METADATA))?)?;
            assert_eq!(metadata["schema"], 4);
            for step in &c.steps[boundary..] {
                apply(&mut resumed, step)?;
            }
            assert_exact(&uninterrupted, &resumed);
            compare_state(
                &format!("{} resume {boundary}", c.name),
                &resumed,
                &c.steps.last().unwrap().states,
            );
            bits_equal(
                &probe.logits(&c, &uninterrupted)?,
                &probe.logits(&c, &resumed)?,
            );
        }
        let export = root.0.join(format!("{}-export", c.name));
        uninterrupted.adapter()?.export(&export)?;
        let reloaded =
            DeepseekV41AdapterTrainer::new(&DeepseekV41LoraAdapter::load(&export)?, c.hyper)?;
        bits_equal(
            &probe.logits(&c, &uninterrupted)?,
            &probe.logits(&c, &reloaded)?,
        );
        let untouched = c.modules.last().unwrap();
        assert_eq!(
            uninterrupted.parameter_step_counts(&untouched.target),
            Some((0, 0))
        );
    }
    Ok(())
}

#[test]
fn conditional_adamw_invalid_late_parameter_and_overflow_are_atomic() -> Result<()> {
    let root = Temp::new()?;
    let c = fixture().units.remove(0);
    let initial = adapter(&c, &root.0.join("source"))?;
    let base = DeepseekV41AdapterTrainer::new(&initial, c.hyper)?;
    let first = &c.modules[0];
    let last = c.modules.last().unwrap();
    let valid = DeepseekV41AdapterParameterGradient {
        target: first.target.clone(),
        grad_a: Some(vec![0.1; first.a_bits.len()]),
        grad_b: Some(vec![0.1; first.b_bits.len()]),
    };
    for kind in 0..6 {
        let mut trainer = base.clone();
        let mut bad = DeepseekV41AdapterParameterGradient {
            target: last.target.clone(),
            grad_a: Some(vec![0.1; last.a_bits.len()]),
            grad_b: Some(vec![0.1; last.b_bits.len()]),
        };
        match kind {
            0 => bad.target = "unknown.module".into(),
            1 => bad = valid.clone(),
            2 => {
                bad.grad_b.as_mut().unwrap().pop();
            }
            3 => bad.grad_b.as_mut().unwrap()[0] = f32::NAN,
            4 => bad.grad_b.as_mut().unwrap()[0] = f32::INFINITY,
            _ => bad.grad_b.as_mut().unwrap()[0] = f32::MAX,
        }
        assert!(
            trainer.step_parameters(&[valid.clone(), bad]).is_err(),
            "bad cohort {kind}"
        );
        assert_exact(&trainer, &base);
    }
    let mut trainer = base.clone();
    assert!(trainer
        .step_parameters(&[DeepseekV41AdapterParameterGradient {
            target: "unknown.module".into(),
            grad_a: None,
            grad_b: None
        }])
        .is_err());
    assert_exact(&trainer, &base);
    assert!(trainer.step_complete(&[]).is_err());
    assert_exact(&trainer, &base);
    trainer.step = u64::MAX;
    let before = trainer.clone();
    assert!(trainer.step(&[]).is_err());
    assert_exact(&trainer, &before);
    trainer = base.clone();
    trainer.step = i32::MAX as u64;
    trainer.slots.get_mut(&last.target).unwrap().step_b = i32::MAX as u64;
    let before = trainer.clone();
    let overflow = DeepseekV41AdapterParameterGradient {
        target: last.target.clone(),
        grad_a: None,
        grad_b: Some(vec![0.0; last.b_bits.len()]),
    };
    assert!(trainer.step_parameters(&[valid, overflow]).is_err());
    assert_exact(&trainer, &before);
    Ok(())
}

#[test]
fn conditional_adamw_sparse_moments_allocate_on_first_parameter_gradient() -> Result<()> {
    let root=Temp::new()?;
    let c=fixture().units.remove(0);
    let initial=adapter(&c,&root.0.join("source"))?;
    let mut trainer=DeepseekV41AdapterTrainer::new(&initial,c.hyper)?;
    assert_eq!(trainer.allocated_moment_elements(),0);
    trainer.step_parameters(&[])?;
    assert_eq!(trainer.allocated_moment_elements(),0);
    let m=&c.modules[0];
    trainer.step_parameters(&[DeepseekV41AdapterParameterGradient {target:m.target.clone(),grad_a:None,grad_b:Some(vec![0.0;m.b_bits.len()])}])?;
    assert_eq!(trainer.parameter_step_counts(&m.target),Some((0,1)));
    assert_eq!(trainer.allocated_moment_elements(),2*m.b_bits.len());
    let path=root.0.join("sparse");trainer.save_checkpoint(&path)?;
    let bytes=fs::read(path.join(OPTIMIZER_STATE))?;
    let tensors=safetensors::SafeTensors::deserialize(&bytes)?;
    assert_eq!(tensors.len(),2*trainer.targets().count()+2);
    assert!(tensors.tensor(&format!("{}.exp_avg_a",m.target)).is_err());
    let resumed=DeepseekV41AdapterTrainer::resume_from_checkpoint(&path)?;
    assert_exact(&trainer,&resumed);
    assert_eq!(resumed.allocated_moment_elements(),2*m.b_bits.len());
    // A schema-3 checkpoint with explicit zero moments for never-used
    // parameters still resumes, then releases those redundant payloads.
    let mut dense=DeepseekV41AdapterTrainer::new(&initial,c.hyper)?;
    for (target,s) in &mut dense.slots {
        let f=&dense.factors[target];
        s.exp_avg_a=vec![0.0;f.a.len()];s.exp_avg_sq_a=s.exp_avg_a.clone();
        s.exp_avg_b=vec![0.0;f.b.len()];s.exp_avg_sq_b=s.exp_avg_b.clone();
    }
    let legacy=root.0.join("legacy-3");dense.adapter()?.export(&legacy)?;dense.save(&legacy)?;
    rewrite_state(&legacy,3,|_,_,_,_|true)?;
    let migrated=DeepseekV41AdapterTrainer::resume_from_checkpoint(&legacy)?;
    assert_eq!(migrated.allocated_moment_elements(),0);
    assert_eq!(migrated.step_count(),0);
    Ok(())
}

fn rewrite_state(
    directory: &Path,
    schema: u32,
    change: impl Fn(&str, &mut safetensors::Dtype, &mut Vec<usize>, &mut Vec<u8>) -> bool,
) -> Result<()> {
    use safetensors::tensor::{serialize_to_file, TensorView};
    let path = directory.join(OPTIMIZER_STATE);
    let bytes = fs::read(&path)?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut owned = Vec::new();
    for (name, view) in tensors.tensors() {
        let mut dtype = view.dtype();
        let mut shape = view.shape().to_vec();
        let mut data = view.data().to_vec();
        if change(&name, &mut dtype, &mut shape, &mut data) {
            owned.push((name, dtype, shape, data));
        }
    }
    let views = owned
        .iter()
        .map(|(n, t, s, b)| Ok((n.as_str(), TensorView::new(*t, s.clone(), b)?)))
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(views, None, &path)?;
    let metadata = directory.join(OPTIMIZER_METADATA);
    let mut meta: serde_json::Value = serde_json::from_slice(&fs::read(&metadata)?)?;
    meta["schema"] = serde_json::json!(schema);
    meta["optimizer_file_sha256"] = serde_json::json!(hash_file(&path)?);
    fs::write(metadata, serde_json::to_vec_pretty(&meta)?)?;
    Ok(())
}

#[test]
fn conditional_adamw_counter_checkpoint_rejects_invalid_payloads_and_migrates_v2() -> Result<()> {
    use safetensors::Dtype;
    let root = Temp::new()?;
    let c = fixture().units.remove(0);
    let initial = adapter(&c, &root.0.join("source"))?;
    let mut trained = DeepseekV41AdapterTrainer::new(&initial, c.hyper)?;
    apply(&mut trained, &c.steps[0])?;
    let target = &c.modules[0].target;
    for kind in 0..4 {
        let checkpoint = root.0.join(format!("invalid-{kind}"));
        trained.save_checkpoint(&checkpoint)?;
        rewrite_state(&checkpoint, 3, |name, dtype, shape, data| {
            if name == format!("{target}.step_a") {
                match kind {
                    0 => *data = u64::MAX.to_le_bytes().to_vec(),
                    1 => {
                        *dtype = Dtype::F32;
                        *data = 1.0f32.to_le_bytes().to_vec();
                    }
                    2 => {
                        *shape = vec![2];
                        data.extend_from_slice(&0u64.to_le_bytes());
                    }
                    _ => *data = 0u64.to_le_bytes().to_vec(),
                }
            }
            true
        })?;
        assert!(
            DeepseekV41AdapterTrainer::resume_from_checkpoint(&checkpoint).is_err(),
            "invalid counter {kind}"
        );
    }
    let mut dense = DeepseekV41AdapterTrainer::new(&initial, c.hyper)?;
    let gradients = c
        .modules
        .iter()
        .map(|m| DeepseekV41AdapterGradient {
            target: m.target.clone(),
            grad_a: vec![0.1; m.a_bits.len()],
            grad_b: vec![0.1; m.b_bits.len()],
        })
        .collect::<Vec<_>>();
    dense.step_complete(&gradients)?;
    dense.step_complete(&gradients)?;
    let old = root.0.join("legacy-v2");
    dense.save_checkpoint(&old)?;
    rewrite_state(&old, 2, |name, _, _, _| {
        !name.ends_with(".step_a") && !name.ends_with(".step_b")
    })?;
    let mut resumed = DeepseekV41AdapterTrainer::resume_from_checkpoint(&old)?;
    assert_exact(&dense, &resumed);
    resumed.step_complete(&gradients)?;
    dense.step_complete(&gradients)?;
    assert_exact(&dense, &resumed);
    let upgraded = root.0.join("upgraded-v3");
    resumed.save_checkpoint(&upgraded)?;
    assert_exact(
        &resumed,
        &DeepseekV41AdapterTrainer::resume_from_checkpoint(&upgraded)?,
    );
    Ok(())
}
