"""Fail closed on the expanded AVX512/RDNA3 GPT2 qualification receipts."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
RECEIPTS=Path(__file__).parent/'release_parity/multimodal_training/20261106-rdna3-gpt2-shapes'
SHAPES=[(2,8,4),(1,8,4),(3,8,4),(2,6,4),(2,10,4),(2,16,4),(2,8,2),(2,8,8)]
COUNTS={'shapes':16,'shared':74,'kimi':3,'seq2seq':6,'gemma3':4,'gemma4':2,'llava':4}

def require(condition,message):
    if not condition:raise ValueError(message)

def load(name):return json.loads((RECEIPTS/name).read_text())

def validate_qualification(q,count):
    require(q.get('target')=='rdna3' and q.get('pass') is True and q.get('complete') is True,'incomplete RDNA3 qualification')
    require(q.get('rtol')==0 and q.get('comparison')=='strict_less_than','acceptance policy changed')
    require(q.get('absolute_limit',q.get('absolute_logit_limit'))==2e-7,'logit limit changed')
    rows=q.get('results',[])
    require(q.get('expected_cases')==count and len(rows)==count and q.get('completed_cases',len(rows))==count,'case coverage changed')
    require(len({r.get('case',r.get('name')) for r in rows})==count,'duplicate cases')
    for row in rows:
        checks=row.get('checks',[])
        require(row.get('pass') is True and checks and all(c.get('pass') is True for c in checks),'failed required check')
        for check in checks:
            if 'logits' in check.get('metric','') and 'max_abs' in check:
                require(0<=check['max_abs']<2e-7,'logit gate violated')
    return rows

def main(historical_only=False):
    hashes=load('receipt-hashes.json')
    required={f'{name}-qualification.json' for name in COUNTS}|{
        'cpu-manifest.json','primitives.log','shader-provenance.json','spirv-validation.json',
        'isolation.json','benchmark.json','v41-native-qualification.json','v41-bridge-qualification.json','summary.json'}
    require(required<=hashes['receipts'].keys(),'missing receipt bindings')
    for name,expected in hashes['receipts'].items():
        require(hashlib.sha256((RECEIPTS/name).read_bytes()).hexdigest()==expected,'receipt hash mismatch: '+name)
    current,historical=verify_sources(hashes['sources'])
    rows=None;profiles=set()
    for name,count in COUNTS.items():
        result=validate_qualification(load(name+'-qualification.json'),count)
        if name=='shapes':rows=result
        if name in ['shared','kimi','seq2seq']:profiles.update(r['profile'] for r in result)
    expected={f'gpt2_b{b}_s{s}_h{h}/{mask}' for b,s,h in SHAPES for mask in ['unmasked','padded']}
    require({r['case'] for r in rows}==expected,'expanded shape coverage changed')
    cpu=load('cpu-manifest.json')
    require(cpu['complete'] and cpu['oracle']['text_attention']=='sdpa','CPU backend contract changed')
    require(len(cpu['original_control'])==2 and all(c['logit_word_differences']==0 for c in cpu['original_control']),
        'original GPT2 CPU control changed')
    tests=(RECEIPTS/'primitives.log').read_text()
    require(re.search(r'test result: ok\. 9 passed; 0 failed;',tests),'primitive tests incomplete')
    require(len(re.findall(r'^test .*rdna3_gpt2_.* \.\.\. ok$',tests,re.M))==9,'primitive test coverage changed')
    provenance=load('shader-provenance.json')
    require(provenance['pass'] and provenance['status_counts']=={'identical':43},'shader provenance failed')
    require(load('spirv-validation.json')['pass'] is True,'SPIR-V validation failed')
    isolation=load('isolation.json')
    require(isolation['unchanged'] and isolation['files_checked']==1341 and isolation['existing_recipes']==672
        and isolation['failures']==[],'upstream bytes changed')
    benchmark=load('benchmark.json')
    require(benchmark['outputs_word_identical'] and benchmark['rounds']==8 and benchmark['order']=='ABBA'
        and len(benchmark['cases'])==4,'paired benchmark coverage changed')
    require(all(s['timer']=='gpu_timestamp' for c in benchmark['cases'] for s in c['samples']),
        'retained benchmark lacks GPU timestamp coverage')
    native=load('v41-native-qualification.json');bridge=load('v41-bridge-qualification.json')
    require(native['pass'] and native['completed_tests']==native['expected_tests']==30,'V4.1 native regression failed')
    require(bridge['pass'] and bridge['completed_cases']==bridge['expected_cases']==2,'V4.1 bridge regression failed')
    profiles.add(bridge['profile']);require(len(profiles)==44,'retained backbone inventory changed')
    print(f'HISTORICAL RECEIPT INTEGRITY: PASS ({len(current)} sources match this checkout, {len(historical)} verified against immutable history)')
    for name,binding in sorted(historical.items()):
        print(f'  historical binding: {name} at {binding["revision"][:12]}')
    if historical_only:
        return 0
    return verify_current(hashes['sources'])


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args):
    return subprocess.run(
        ['git', '-C', str(ROOT), *args],
        capture_output=True,
        check=True,
    ).stdout


def _history_available():
    """A shallow checkout cannot answer historical questions at all."""
    try:
        return _git('rev-parse', '--is-shallow-repository').decode().strip() != 'true'
    except Exception:
        return True


def _blob_sha256(oid):
    return hashlib.sha256(_git('cat-file', 'blob', oid)).hexdigest()


def _history_lookup(path, expected):
    """Return a commit that contains exactly `expected` at `path`, else None.

    Only committed blobs reachable from the repository are consulted, so a
    receipt can never validate itself against its own recorded hash or against
    bytes supplied by the checkout.
    """
    revisions = _git('rev-list', '--all', '--', path).decode().split()
    seen = set()
    for revision in revisions:
        blob = _git('rev-parse', f'{revision}:{path}').decode().strip()
        if not blob or blob in seen:
            continue
        seen.add(blob)
        if _blob_sha256(blob) == expected:
            return revision
    return None


def verify_sources(sources, current_sha256=None, history_lookup=None, history_available=None):
    """Verify every bound source as either current bytes or immutable history.

    A historical receipt binds the bytes that were qualified. It does not
    require the working tree to still contain them; it requires those bytes to
    exist in a verified commit. A source that matches neither is a real
    integrity failure and must fail closed.
    """
    current_sha256 = current_sha256 or (lambda name: _sha256_file(ROOT / name))
    history_lookup = history_lookup or _history_lookup
    history_available = history_available or _history_available
    current = {}
    historical = {}
    blocked = []
    for name, expected in sorted(sources.items()):
        try:
            if current_sha256(name) == expected:
                current[name] = expected
                continue
        except (OSError, FileNotFoundError):
            pass
        revision = history_lookup(name, expected)
        if revision is None:
            blocked.append(name)
        else:
            historical[name] = {'sha256': expected, 'revision': revision}
    if blocked and not history_available():
        require(
            False,
            'historical history unavailable in a shallow checkout; fetch full history '
            '(actions/checkout with fetch-depth: 0) to verify: ' + ', '.join(blocked),
        )
    require(not blocked, 'source bound to no verifiable revision: ' + ', '.join(blocked))
    return current, historical


CURRENT_RECEIPT = 'current-qualification.json'


def verify_current(sources, receipt_path=None, source_head=None, current_sha256=None):
    """Require a fresh receipt that binds this checkout, never a historical one.

    Returns 0 when the current source is qualified and 3 when it is not. A
    historical receipt cannot satisfy this: it lists the bytes of its own
    revision, so it fails as soon as the checkout differs from it.
    """
    receipt_path = Path(receipt_path) if receipt_path else RECEIPTS / CURRENT_RECEIPT
    current_sha256 = current_sha256 or (lambda name: _sha256_file(ROOT / name))
    if source_head is None:
        source_head = _git('rev-parse', 'HEAD').decode().strip()
    if not receipt_path.is_file():
        print(f'CURRENT SOURCE QUALIFICATION: UNQUALIFIED (no {CURRENT_RECEIPT} for {source_head[:12]})')
        return 3
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get('source_head') == source_head, 'current qualification is bound to another revision')
    for name in sources:
        require(
            receipt.get('sources', {}).get(name) == current_sha256(name),
            'current qualification does not bind ' + name,
        )
    require(receipt.get('pass') is True, 'current qualification did not pass')
    print(f'CURRENT SOURCE QUALIFICATION: PASS ({source_head[:12]})')
    return 0


if __name__=='__main__':sys.exit(main('--historical-only' in sys.argv))
