"""Build a serial scheduling control and measure identical GPT2 GPU arithmetic."""
import argparse,hashlib,json,os,subprocess
import re
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--forward-lanes',type=int,choices=[0,1,2,4,8,16,32,64],default=0,
        help='0 uses the production geometry policy');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);a.output=a.output.resolve()
    source=Path('shaders/image_text_gpt2_avx512_amd_rdna3.comp')
    text=source.read_text()
    text=re.sub(r'layout\(local_size_x=\d+\)','layout(local_size_x=1)',text)
    replacements={
        'if(local<pc.seq) {\n        uint query=local,qb=':
        'for(uint query=local;query<pc.seq;query+=1u) {\n        uint qb=',
        'index+=64u':'index+=1u'}
    for old,new in replacements.items():
        if text.count(old)!=1:raise ValueError('control source pattern changed: '+old)
        text=text.replace(old,new)
    control=a.output/'serial.comp';control.write_text(text)
    for mode in ['forward','backward']:
        command=['glslangValidator','-V','-S','comp','-Ishaders',str(control),'-o',str(a.output/(mode+'.spv'))]
        if mode=='backward':command.append('-DRDNA3_GPT2_BACKWARD=1')
        subprocess.run(command,check=True,capture_output=True)
    candidate=a.output/'forward-packed.comp'
    candidate.write_text(re.sub(r'layout\(local_size_x=\d+\)',f'layout(local_size_x={a.forward_lanes or 8})',source.read_text()))
    subprocess.run(['glslangValidator','-V','-S','comp','-Ishaders',str(candidate),'-o',
        str(a.output/'forward-packed.spv')],check=True,capture_output=True)
    (a.output/'benchmark-config.json').write_text(json.dumps(dict(forward_lanes=a.forward_lanes)))
    env={**os.environ,'HIERARCHOS_ATEN_VECTOR_WIDTH':'16','HIERARCHOS_RDNA3_GPT2_BENCH':str(a.output)}
    process=subprocess.run(['cargo','test','--lib','rdna3_gpt2_paired_gpu_benchmark','--','--ignored','--nocapture',
        '--test-threads=1'],env=env,capture_output=True,text=True)
    (a.output/'cargo.log').write_text(process.stdout+process.stderr);process.check_returncode()
    report=json.loads((a.output/'timings.json').read_text())
    report['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    report['control_sha256']=hashlib.sha256(control.read_bytes()).hexdigest()
    report['forward_lanes']=a.forward_lanes
    report['candidate_sha256']=hashlib.sha256((a.output/'forward-packed.spv').read_bytes()).hexdigest()
    report['binary_sha256']=hashlib.sha256(Path(report['binary']).read_bytes()).hexdigest()
    for case in report['cases']:
        import statistics
        med={k:statistics.median(x['batch_ms'] for x in case['samples'] if x['mode']==k) for k in ['serial','packed']}
        case.update(median_batch_ms=med,packed_speedup=med['serial']/med['packed'])
        print(case['sequence'],case['phase'],case['packed_speedup'])
    (a.output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
