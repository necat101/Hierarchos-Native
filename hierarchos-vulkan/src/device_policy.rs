//! Device admission policy, independent of Vulkan handles and environment overrides.
//!
//! WMMA admission is deliberately separate from the existing FP32 vendor
//! selector. Recognising RDNA3.5/4 here must not retune an older AMD or Intel
//! device's FP32 kernels. Unknown devices keep the existing FP32 path.

/// AMD RDNA3, RDNA3.5 and RDNA4 PCI IDs. Vulkan features remain mandatory.
/// Source: AMD GPUOpen device_info, commit dff4582fe929ae85e6224a5b4a8761eec4872d18,
/// https://github.com/GPUOpen-Tools/device_info/blob/dff4582fe929ae85e6224a5b4a8761eec4872d18/device_info.cpp
/// device_info.cpp (gfx11/gfx11.5/gfx12 ASIC rows), plus the existing tuned
/// Navi 33 ID 749f. Do not infer architecture from a device name or ID range.
pub(crate) const fn wmma_hardware_eligible(vendor_id: u32, device_id: u32) -> bool {
    vendor_id == 0x1002 && matches!(device_id,
        // RDNA3: Navi 31/32/33 and Phoenix/Hawk Point.
        0x73a8 | 0x7448 | 0x7449 | 0x744a | 0x744b | 0x744c | 0x745e |
        0x73c4 | 0x73c5 | 0x73c8 | 0x7460 | 0x7461 | 0x7470 | 0x7478 | 0x747e |
        0x73f0 | 0x7480 | 0x7481 | 0x7483 | 0x7487 | 0x7489 | 0x748b | 0x7499 | 0x749f |
        0x15bf | 0x15c8 | 0x1900 | 0x1901 |
        // RDNA3.5: Strix Point/Halo and Krackan Point.
        0x1114 | 0x150e | 0x1586 | 0x1902 |
        // RDNA4: gfx1200/gfx1201.
        0x7550 | 0x7551 | 0x7590)
}

pub(crate) const fn wmma_probe_requested(requested: bool, vendor_id: u32, device_id: u32) -> bool {
    requested && wmma_hardware_eligible(vendor_id, device_id)
}

/// The pre-WMMA composition gate, recovered verbatim from main `ead6cd3`.
///
/// A recorded verdict admits the graph outright -- the record measured the
/// complete graph, including its portable fallback -- and without one the
/// vendor variant gate decides. This is deliberately device-independent: the
/// RDNA3 routing work must not make an Intel Gen9 or RDNA1/RDNA2 dispatch
/// decision differ from pre-WMMA `main`. Requiring `vendor_variant` for a
/// recorded geometry would strand the portable compositions those records
/// measured, which is exactly the regression this predicate prevents.
pub(crate) const fn composition_admitted(
    unmeasured_qualified: bool,
    vendor_variant: bool,
    measured: bool,
) -> bool {
    measured || (unmeasured_qualified && vendor_variant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wmma_requires_amd_rdna3_or_newer() {
        for id in [0x744c, 0x747e, 0x7480, 0x15bf, 0x150e, 0x1586, 0x7550, 0x7551, 0x7590] {
            assert!(wmma_hardware_eligible(0x1002, id));
            for vendor in [0x8086, 0x10de, 0x13b5, 0, 0xffff] {
                assert!(!wmma_hardware_eligible(vendor, id));
            }
        }
    }

    #[test]
    fn rdna1_and_rdna2_remain_fp32() {
        for id in [
            0x7310, 0x7312, 0x7318, 0x7319, 0x731a, 0x731b, 0x731e, 0x731f,
            0x7340, 0x7341, 0x7343, 0x7347, 0x734f, 0x7360, 0x7362,
            0x13c0, 0x1506, 0x163f, 0x164d, 0x164e, 0x1681,
            0x73a0, 0x73a1, 0x73a2, 0x73a3, 0x73a5, 0x73ab, 0x73ae, 0x73af, 0x73bf,
            0x73c0, 0x73c1, 0x73c3, 0x73ce, 0x73df, 0x73e0, 0x73e1, 0x73e3,
            0x73ef, 0x73ff, 0x7420, 0x7421, 0x7422, 0x7423, 0x743f,
        ] {
            assert!(!wmma_hardware_eligible(0x1002, id), "older AMD device {id:04x}");
        }
    }

    #[test]
    fn intel_gen9_cdna_and_unknown_devices_cannot_request_wmma() {
        for (vendor, id) in [(0x8086, 0x1916), (0x8086, 0x5916),
            (0x1002, 0x740f), (0x1002, 0x74a1), (0x1002, 0xffff), (0x1002, 0x744d)] {
            assert!(!wmma_hardware_eligible(vendor, id));
        }
    }

    #[test]
    fn fp32_off_never_probes_wmma_on_any_device() {
        for (vendor, id) in [(0x1002, 0x15bf), (0x1002, 0x731f), (0x1002, 0x73bf), (0x8086, 0x1916)] {
            assert!(!wmma_probe_requested(false, vendor, id));
        }
        assert!(wmma_probe_requested(true, 0x1002, 0x15bf));
        assert!(!wmma_probe_requested(true, 0x8086, 0x1916));
    }

    /// The literal pre-WMMA `main` expression, kept as an independent oracle:
    /// `measured.is_some() || (unmeasured_composition_is_qualified(vendor) &&
    /// selects_variant_for(..))`.
    const fn pre_wmma_gate(qualified: bool, variant: bool, measured: bool) -> bool {
        measured || (qualified && variant)
    }

    #[test]
    fn composition_admission_is_exactly_the_pre_wmma_rule_on_every_device() {
        for qualified in [false, true] {
            for variant in [false, true] {
                for measured in [false, true] {
                    assert_eq!(
                        composition_admitted(qualified, variant, measured),
                        pre_wmma_gate(qualified, variant, measured),
                        "qualified={qualified} variant={variant} measured={measured}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_recorded_geometry_is_admitted_even_where_the_vendor_variant_falls_back() {
        // The RDNA3 batch-32 `staged` entry, and the equivalent portable
        // compositions recorded on Intel, must stay buildable: pre-WMMA main
        // admitted them and this branch must not narrow that.
        assert!(composition_admitted(false, false, true));
        assert!(composition_admitted(true, false, true));
        assert!(!composition_admitted(false, false, false));
        assert!(!composition_admitted(true, false, false));
        assert!(composition_admitted(true, true, false));
    }
}
