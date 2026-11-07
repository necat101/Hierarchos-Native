"""Freeze the host CRT scalar logf contract on independent softmax sums."""
import argparse,ctypes,hashlib,json,random,struct
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    library=ctypes.CDLL('ucrtbase');library.logf.argtypes=[ctypes.c_float];library.logf.restype=ctypes.c_float
    def value(x):return struct.unpack('<f',struct.pack('<I',x))[0]
    def signed(x):return struct.unpack('<i',struct.pack('<f',x))[0]
    rng=random.Random(0x106F512)
    inputs=[rng.randrange(0x3f800000,0x41800001) for _ in range(4096)]
    # Table boundaries and the near-one polynomial boundary, independent of model data.
    inputs.extend((exponent<<23)+(i<<16)+offset for exponent in range(127,131)
        for i in range(128) for offset in [0,0x7fff,0x8000,0x8001])
    inputs.extend([0x3f880000+i for i in [-1,0,1]])
    with a.output.open('x') as f:
        json.dump(dict(scope='positive normal sums in [1,16], plus table-boundary sums below 16',
            producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ucrt_sha256=hashlib.sha256(Path('C:/Windows/System32/ucrtbase.dll').read_bytes()).hexdigest(),
            x=[signed(value(x)) for x in inputs],y=[signed(library.logf(value(x))) for x in inputs]),f,separators=(',',':'))
        f.write('\n')
if __name__=='__main__':main()
