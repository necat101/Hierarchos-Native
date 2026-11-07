"""Check raw Git blob identity only after verifying every clean-filter contract."""
import hashlib,json,subprocess
from pathlib import Path
CRATE=Path(__file__).resolve().parents[3];ROOT=CRATE.parent
BASE='528e3f722bc609cb2d0bdc5dcb032141c662dc5b'
def git(*args,input=None):return subprocess.check_output(['git','-C',str(ROOT),*args],input=input)
entries=git('ls-tree','-r',BASE,'--','hierarchos-vulkan/shaders','hierarchos-vulkan/validation/fixtures').decode().splitlines()
expected={line.split('\t',1)[1]:line.split('\t',1)[0].split()[2] for line in entries}
attrs=git('check-attr','--stdin','-z','text','filter',input=b'\0'.join(p.encode() for p in expected)+b'\0').decode().split('\0')
for i in range(0,len(attrs)-1,3):
    path,attribute,value=attrs[i:i+3]
    assert value==('unset' if attribute=='text' else 'unspecified'),(path,attribute,value)
failures=[]
for path,oid in expected.items():
    data=(ROOT/path).read_bytes()
    actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if actual!=oid:failures.append(path)
def recipes(v):
    if isinstance(v,dict):
        if 'artifact' in v:yield v
        else:
            for value in v.values():yield from recipes(value)
    elif isinstance(v,list):
        for value in v:yield from recipes(value)
name='hierarchos-vulkan/validation/training_shader_manifest.json'
old={v['artifact']:v for v in recipes(json.loads(git('show',BASE+':'+name)))}
new={v['artifact']:v for v in recipes(json.loads((ROOT/name).read_text()))}
failures.extend('recipe:'+n for n,v in old.items() if new.get(n)!=v)
result=dict(baseline=BASE,files_checked=len(expected),existing_recipes=len(old),unchanged=not failures,
    failures=failures,method='Git blob SHA1 after checking -text and absent clean filters for every path',
    limitation='Static preservation only; no physical Intel execution')
(Path(__file__).parent/'isolation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));assert not failures
