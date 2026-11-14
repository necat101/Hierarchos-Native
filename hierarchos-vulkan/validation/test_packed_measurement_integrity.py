"""Measurement identity and numeric evidence must survive route qualification."""
import copy
import json
import unittest
from pathlib import Path
import check_measured_producer_routes as guard
from qualify_packed_current import merge_batch_receipts


class PackedMeasurementIntegrity(unittest.TestCase):
    def forward_row(self, route='packed'):
        chain=next(s for s in guard.CHAIN_PRODUCERS if s.name=='tbptt-packed-forward')
        row=dict(batch=1,timing_protocol='shipped-route-v1',default_label=route,default_ms=30.0,
            default=dict(label=route,median_ms=30.0),legacy=dict(label='legacy',median_ms=25.0),
            packed=dict(label='packed',median_ms=20.0),legacy_ms=25.0,packed_ms=20.0,
            decision_timing_sources=dict(legacy='forced-arm',packed='forced-arm'))
        row[route+'_ms']=30.0;row['decision_timing_sources'][route]='shipped-default'
        return chain,row

    def test_slower_production_timing_is_used_for_either_route(self):
        for route in ('legacy','packed'):
            chain,row=self.forward_row(route)
            guard.check_shipped_route_timings(chain,row)
            row[route+'_ms']=row[route]['median_ms']
            row['decision_timing_sources'][route]='forced-arm'
            with self.assertRaises(guard.Problem):guard.check_shipped_route_timings(chain,row)

    def test_invented_candidate_time_is_rejected(self):
        chain,row=self.forward_row();row['legacy_ms']=1.0
        with self.assertRaises(guard.Problem):guard.check_shipped_route_timings(chain,row)

    def test_missing_raw_readings_are_rejected(self):
        chain,row=self.forward_row();row.pop('packed')
        with self.assertRaises(guard.Problem):guard.check_shipped_route_timings(chain,row)

    def test_one_refusal_does_not_waive_a_bit_exact_geometry(self):
        producer=guard.PRODUCERS_BY_NAME['low-rank-forward']
        row=dict(batch=8,worst_max_abs=1e-5,tensors=[dict(max_abs=1e-5)])
        receipt=dict(withheld_routes=[dict(batch=8,route='staged',worst_max_abs=1e-5,reason='non-bit-identical')])
        self.assertTrue(guard.row_refuses_staged(producer,receipt,row))
        row['worst_max_abs']=0.0;row['tensors'][0]['max_abs']=0.0
        self.assertFalse(guard.row_refuses_staged(producer,receipt,row))

    def test_refusal_requires_this_geometry_and_numeric_evidence(self):
        producer=guard.PRODUCERS_BY_NAME['low-rank-forward']
        row=dict(batch=8,worst_max_abs=1e-5,tensors=[dict(max_abs=0.0)])
        receipt=dict(withheld_routes=[dict(batch=1,route='staged',worst_max_abs=1e-5,reason='other geometry')])
        self.assertFalse(guard.row_refuses_staged(producer,receipt,row))

    def test_vendor_availability_does_not_expand_vendor_admission(self):
        producer=guard.PRODUCERS_BY_NAME['time-mix-projection']
        row=dict(fused_label='fused',composed_ran_vendor_linear3=True)
        self.assertEqual(guard.unrecorded_default(producer,dict(device='Intel(R) HD Graphics 520'),row),'composed')
        self.assertEqual(guard.unrecorded_default(producer,dict(device='AMD Radeon Graphics [pci=1002:15bf;driver=00800184]'),row),'fused')

    def folded_row(self):
        crate=Path(__file__).resolve().parents[1]
        path=crate/'validation/release_parity/multimodal_training/20261113-phase22-recovery/packed-cell-backward-folded-same-schedule/tbptt_backward_fusion_bench.json'
        row=copy.deepcopy(json.loads(path.read_text(encoding='utf-8'))['batches'][0])
        self.assertEqual(row['fused_schedule'],row['folded_schedule'])
        row['folded_candidate']=False
        chain=next(s for s in guard.CHAIN_PRODUCERS if s.name=='tbptt-backward-fusion')
        return chain,row

    def test_duplicate_folded_schedule_keeps_numeric_gates(self):
        chain,row=self.folded_row()
        self.assertNotIn('folded',guard.check_folded_evidence(chain,row))
        for field,value in [('folded_key_parameter_max_abs',1e-8),('folded_max_abs',float('nan')),
                            ('staged_key_parameter_max_abs',0.0),('folded_parameter_delta',float('inf'))]:
            with self.subTest(field=field):
                bad=copy.deepcopy(row);bad[field]=value
                with self.assertRaises(guard.Problem):guard.check_folded_evidence(chain,bad)

    def test_duplicate_cannot_be_promoted_as_an_independent_candidate(self):
        chain,row=self.folded_row();row['folded_candidate']=True
        with self.assertRaises(guard.Problem):guard.check_folded_evidence(chain,row)

    def test_isolated_batch_aggregate_uses_every_row(self):
        receipts=[dict(result='pass',device='same',samples=1200,batches=[dict(batch=b,packed_over_legacy=r)],
                       median_packed_over_legacy=r) for b,r in [(1,0.8),(4,1.1),(8,1.2)]]
        merged=merge_batch_receipts(receipts)
        self.assertEqual(merged['median_packed_over_legacy'],1.1)
        self.assertEqual([r['batch'] for r in merged['batches']],[1,4,8])
        with self.assertRaises(ValueError):merge_batch_receipts([receipts[0],receipts[0]])
        receipts[1]['samples']=120
        with self.assertRaises(ValueError):merge_batch_receipts(receipts)


if __name__=='__main__':unittest.main()
