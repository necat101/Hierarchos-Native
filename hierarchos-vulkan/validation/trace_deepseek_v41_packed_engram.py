"""Observe the existing packed Engram fixture without changing acceptance data."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import validate_deepseek_v41_engram as reference
import validate_deepseek_v41_packed_weights as packed


class CapturedProjection(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values
    def forward(self, x):
        return self.values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path",type=Path,required=True)
    p.add_argument("--native-output",type=Path,required=True)
    args = p.parse_args()
    ns = reference.load_reference(args.reference_path)
    fixture = json.loads(packed.FIXTURE.read_text())["engrams"]
    native = json.loads(args.native_output.read_text())["cases"]
    torch.set_num_threads(1)
    assert len(native) == len(fixture)
    for f,n in zip(fixture,native):
        assert f["name"] == n["name"]
        rows,hc,dim = f["rows"],f["hc"],f["hidden"]
        lookup = torch.tensor(f["expected_lookup_bits"],dtype=torch.uint32).view(torch.float32).reshape(rows,-1)
        w = torch.tensor(f["wkv"]).reshape((hc+1)*dim,-1)
        expected_kv = F.linear(lookup,w)
        actual_kv = torch.tensor(n["kv"]).reshape_as(expected_kv)
        actual = torch.tensor(n["output"]).reshape(rows,hc,dim)
        module = ns["Engram"].__new__(ns["Engram"])
        nn.Module.__init__(module)
        module.dim,module.hc_mult,module.clamp_value,module.eps = dim,hc,1e-6,1e-20
        module.embed = reference.EmbedStub(torch.tensor(f["table"]).reshape(f["table_rows"],f["head_dim"]))
        module.wkv = CapturedProjection(actual_kv)
        module.q_weight = torch.tensor(f["q_weight"]).reshape(hc,dim)
        module.k_weight = torch.tensor(f["k_weight"]).reshape(hc,dim)
        stream = torch.tensor(f["stream"]).reshape(rows,hc,dim)
        hashes = torch.tensor(f["hashes"]).reshape(rows,f["cols"])
        output = module(stream,hashes,torch.tensor(f["mask"],dtype=torch.bool))
        expected = torch.tensor(f["expected"]).reshape_as(actual)
        print(f"{f['name']}: projection={(actual_kv.double()-expected_kv.double()).abs().max().item():.17g}, residual-on-identical-KV={(actual.double()-output.double()).abs().max().item():.17g}, total={(actual.double()-expected.double()).abs().max().item():.17g}")


if __name__ == "__main__": main()
