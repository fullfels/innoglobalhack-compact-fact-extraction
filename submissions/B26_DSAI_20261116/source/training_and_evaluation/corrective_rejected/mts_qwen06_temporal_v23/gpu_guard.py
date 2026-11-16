"""Avoid launching a second model worker in the same Colab VM."""
import os,subprocess

def check_idle_worker():
    try:
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True,timeout=10)
    except (OSError,subprocess.SubprocessError):return {'query_available':False}
    others=[]
    for line in output.splitlines():
        parts=line.split(',',1)
        if len(parts)!=2 or not parts[0].strip().isdigit():continue
        pid=int(parts[0]);name=parts[1].strip()
        if pid not in {os.getpid(),os.getppid()} and 'python' in name.lower():others.append({'pid':pid,'process':name})
    if others:raise RuntimeError('Another Python GPU worker is active in this VM. Finish or stop the previous evaluation before training. '+str(others))
    return {'query_available':True,'other_python_gpu_workers':0}
