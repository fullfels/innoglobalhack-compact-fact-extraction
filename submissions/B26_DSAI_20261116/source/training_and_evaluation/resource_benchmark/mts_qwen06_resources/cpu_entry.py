from pathlib import Path
import subprocess,sys

out=Path('/results/cpu_run');out.mkdir(parents=True,exist_ok=True)
code=1
try:
    result=subprocess.run([sys.executable,'-u','/bench/resource_benchmark.py','--root','/bench','--out',str(out),
        '--device','cpu','--modes','fp16','--budget-minutes','50'],cwd='/bench',timeout=55*60)
    code=result.returncode
except subprocess.TimeoutExpired:
    print('CPU supervisor wall-clock limit reached; partial records retained',flush=True)
finally:
    subprocess.run([sys.executable,'/bench/pack_run.py',str(out)],cwd='/bench',check=True)
sys.exit(code)
