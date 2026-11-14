"""Check preserved pre-WMMA FP32 shader bytes and actual device-admission policy.

This CPU gate does not claim an execution on the Gen9 qualification host.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 'ead6cd3fa5f8943e478dcd0aec27e6014f62d1d5'
QUALIFIED_RDNA3 = '92cfa703acb3345b5af5e607ed97f697704b064d'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def verify():
    object_format = git('rev-parse', '--show-object-format').decode().strip()
    preserved = []
    entries = git('ls-tree', '-r', '-z', BASELINE, '--', 'hierarchos-vulkan/shaders')
    for entry in entries.split(b'\0'):
        if not entry:
            continue
        metadata, encoded_path = entry.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        if kind != 'blob':
            raise ValueError('unexpected shader entry type')
        name = encoded_path.decode()
        data = (ROOT / name).read_bytes()
        current = hashlib.new(object_format, b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if current != oid:
            raise ValueError('pre-WMMA shared shader changed: ' + name)
        preserved.append(name)
    if not preserved:
        raise ValueError('empty pre-WMMA shader inventory')
    vendor = 'hierarchos-vulkan/src/vendor.rs'
    if (ROOT / vendor).read_bytes() != git('show', BASELINE + ':' + vendor):
        raise ValueError('pre-WMMA FP32 vendor selector changed')
    rdna3 = {}
    for extension in ('comp', 'spv'):
        original = 'hierarchos-vulkan/shaders/rwkv_key_state_forward_fused_wide.' + extension
        variant = ROOT / ('hierarchos-vulkan/shaders/rwkv_key_state_forward_fused_wide_amd_rdna3.' + extension)
        data = variant.read_bytes()
        if data != git('show', QUALIFIED_RDNA3 + ':' + original):
            raise ValueError('qualified RDNA3 key-state variant changed')
        rdna3[variant.name] = hashlib.sha256(data).hexdigest()
    with tempfile.TemporaryDirectory(prefix='hierarchos-device-policy-') as scratch:
        binary = Path(scratch) / 'device-policy-tests.exe'
        subprocess.run(['rustc', '--edition=2021', '--test',
                        str(ROOT / 'hierarchos-vulkan/src/device_policy.rs'), '-o', str(binary)], check=True)
        controls = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    return dict(status='PASS_CPU_COMPATIBILITY_CONTROLS', baseline=BASELINE,
                preserved_shared_shader_files=len(preserved), fp32_vendor_selector='unchanged',
                rdna3_variant_sha256=rdna3, controls=controls.stdout,
                gen9_hardware_qualification='PENDING_ON_GEN9_HOST')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = verify()
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('MAIN COMPATIBILITY: PASS (' + str(report['preserved_shared_shader_files']) +
          ' original shader files, unchanged FP32 vendor selector, device-policy controls)')


if __name__ == '__main__':
    main()
