"""Build a narrowly repaired reference without modifying the clean stable checkout.

The physical Ally's clean stable GPT-2 exceeded the independent 2e-7 HF gate.
The user explicitly prioritized repairing that numerical drift. This adapter
copies the exact stable sources, applies only the two qualified arithmetic
repairs, and leaves the clean stable executable available for the legacy-bit
comparison. No candidate library implementation is used as the reference.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

BASE_COMMIT = "2e7893d4e3db17370304a8d67adacab963b22bbc"
BINARY = "hierarchos-vulkan-corrected-stable-logits"
REPAIRED_FAMILIES = frozenset({"gpt2", "gpt2_wide"})
REPAIR_FILES = (
    "shaders/transformer_layer_norm_welford_forward.comp",
    "shaders/transformer_layer_norm_welford_forward.spv",
    "shaders/linear_bias_forward_amd_rdna3_kblock128.spv",
    "shaders/linear_bias_forward_amd_rdna3.comp",
)


def source_digest(files: dict[str, bytes]) -> str:
    return hashlib.sha256(b"".join(name.encode() + b"\0" + hashlib.sha256(data).digest()
                                  for name, data in sorted(files.items()))).hexdigest()


def verify_prepared(crate: Path, metadata: dict) -> None:
    root = crate.parent
    files = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*")
             if path.is_file() and path != root / "repair-receipt.json"}
    if source_digest(files) != metadata["copied_source_sha256"]:
        raise ValueError("isolated numerical reference sources changed during validation")


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"numerical reference seam changed: {old[:100]!r}")
    return source.replace(old, new, 1)


def repair_source(source: str) -> str:
    source = replace_once(source,
        'const LAYER_NORM_FORWARD_SPV: &[u8] = include_bytes!("../shaders/layer_norm_forward.spv");',
        'const LAYER_NORM_FORWARD_SPV: &[u8] = include_bytes!("../shaders/layer_norm_forward.spv");\n'
        'const LAYER_NORM_WELFORD_FORWARD_SPV: &[u8] = include_bytes!("../shaders/transformer_layer_norm_welford_forward.spv");\n'
        'const LINEAR_BIAS_FORWARD_AMD_KBLOCK128_SPV: &[u8] = include_bytes!("../shaders/linear_bias_forward_amd_rdna3_kblock128.spv");')
    source = replace_once(source, '    linear_bias_forward: vendor::VendorMatmulKernel,',
        '    linear_bias_forward: vendor::VendorMatmulKernel,\n'
        '    linear_bias_forward_kblock128: Option<vulkan::ComputeKernel>,')
    source = replace_once(source,
        '    fn new(device: &VulkanDevice) -> Result<Self> {\n        Ok(Self {\n'
        '            linear_forward: vendor::VendorMatmulKernel::new(',
        '    fn new(device: &VulkanDevice) -> Result<Self> {\n'
        '        Self::new_with_layer_norm(device, LAYER_NORM_FORWARD_SPV)\n    }\n'
        '    fn new_with_layer_norm(device: &VulkanDevice, layer_norm_spv: &[u8]) -> Result<Self> {\n'
        '        Ok(Self {\n            linear_forward: vendor::VendorMatmulKernel::new(')
    source = replace_once(source,
        'layer_norm_forward: vulkan::ComputeKernel::new(device, LAYER_NORM_FORWARD_SPV, 6, 12)?,',
        'layer_norm_forward: vulkan::ComputeKernel::new(device, layer_norm_spv, 6, 12)?,')
    source = replace_once(source,
        '                vendor::VendorKernelFamily::LinearBiasForward,\n            )?,',
        '                vendor::VendorKernelFamily::LinearBiasForward,\n            )?,\n'
        '            linear_bias_forward_kblock128: None,')
    source = replace_once(source,
        '            } else {\n                kernels.linear_bias_forward.record_dispatch(',
        '''            } else if let Some(kernel) = kernels.linear_bias_forward_kblock128.as_ref()
                .filter(|_| self.input_dim > 128)
            {
                kernel.record_dispatch(commands,
                    &[input, &self.weight.values, &self.bias.values, base_output],
                    bytemuck::bytes_of(&push),
                    [div_ceil_u32(self.output_dim, 16), div_ceil_u32(rows, 16), 1])?;
            } else {
                kernels.linear_bias_forward.record_dispatch(''')
    source = replace_once(source, '        let kernels = Box::new(TransformerKernels::new(&device)?);',
        '''        // Match the candidate's actual RDNA3 PCI scope, without an
        // identity override or the candidate's vendor module implementation.
        let repaired_gpt2 = device.vendor_id() == 0x1002
            && config.architecture == VulkanTransformerArchitecture::Gpt2
            && [0x7448u32,0x7449,0x744a,0x744b,0x744c,0x745e,
            0x7460,0x7461,0x7470,0x747e,0x7480,0x7481,0x7483,0x7487,0x7489,
            0x748b,0x7499,0x749f,0x15bf,0x15c8,0x1900,0x1901].contains(&device.device_id());
        let mut kernels = Box::new(TransformerKernels::new_with_layer_norm(&device,
            if repaired_gpt2 { LAYER_NORM_WELFORD_FORWARD_SPV } else { LAYER_NORM_FORWARD_SPV })?);
        if repaired_gpt2 {
            kernels.linear_bias_forward_kblock128 = Some(vulkan::ComputeKernel::new(
                &device, LINEAR_BIAS_FORWARD_AMD_KBLOCK128_SPV, 4, 12)?);
        }''')
    return source


def prepare(stable: Path, candidate_crate: Path, destination: Path) -> tuple[Path, dict]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(stable), *args], text=True).strip()
    if git("rev-parse", "HEAD") != BASE_COMMIT or git("status", "--porcelain"):
        raise ValueError("numerical repair reference requires the exact clean Windows stable oracle")
    archive = subprocess.check_output(["git", "-C", str(stable), "archive", BASE_COMMIT,
                                       "hierarchos-vulkan", "hierarchos-inference"])
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            if member.isfile():
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError("unsafe stable archive member")
                files[name.as_posix()] = stream.extractfile(member).read()
    prefix = "hierarchos-vulkan/"
    key = prefix + "src/transformer.rs"
    original = files[key]
    files[key] = repair_source(original.decode().replace("\r\n", "\n")).encode()
    manifest = prefix + "Cargo.toml"
    files[manifest] = replace_once(files[manifest].decode(),
        'name = "hierarchos-vulkan-transformer-logits"', f'name = "{BINARY}"').encode()
    for name in REPAIR_FILES:
        files[prefix + name] = (candidate_crate / name).read_bytes()
    # The exact division/sqrt helpers are the stable ones, not candidate copies.
    helper = "shaders/fp32_sqrt_recip.glsl"
    if files[prefix + helper] != (candidate_crate / helper).read_bytes():
        raise ValueError("numerical reference helper changed beyond the approved repair")
    destination = destination.resolve()
    if destination.exists():
        unexpected = [path for path in destination.rglob("*") if path.is_file()
                      and path.relative_to(destination).as_posix() not in {*files, "repair-receipt.json"}]
        if unexpected:
            raise ValueError(f"unexpected file in isolated numerical reference: {unexpected[0]}")
    for name, data in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_bytes() != data:
            path.write_bytes(data)
    metadata = {"base_commit": BASE_COMMIT,
        "repair": "AMD GPT-2 eight-lane Welford and independently accumulated K=128 addmm blocks",
        "authorization": "User: bring the numerical drift down! that is utmost important!",
        "legacy_transformer_source_sha256": hashlib.sha256(original).hexdigest(),
        "corrected_transformer_source_sha256": hashlib.sha256(files[key]).hexdigest(),
        "repair_files_sha256": {name: hashlib.sha256(files[prefix + name]).hexdigest() for name in REPAIR_FILES},
        "copied_source_sha256": source_digest(files)}
    (destination / "repair-receipt.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return destination / "hierarchos-vulkan", metadata
