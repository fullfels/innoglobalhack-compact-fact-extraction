from pathlib import Path
import argparse,csv,json

def main(run):
    obj=json.loads((run/'summary.json').read_text());rows=[]
    for name,r in obj['modes'].items():
        m=r.get('metrics',{});p=m.get('fact_precision',0);rec=m.get('fact_recall',0)
        rows.append({'device':obj['device'],'mode':name,'complete':r.get('complete',False),
            'calls':r.get('n_attempted',0),'errors':r.get('generation_failures'),
            'mean_seconds_all_attempts':r.get('mean_seconds_all_attempts'),
            'RAM_MiB':r['peak_rss_bytes']/1024**2 if r.get('peak_rss_bytes') is not None else None,
            'VRAM_reserved_MiB':r['peak_cuda_reserved_bytes']/1024**2 if r.get('peak_cuda_reserved_bytes') is not None else None,
            'precision':p,'recall':rec,'F1':2*p*rec/(p+rec) if p+rec else 0,
            'temporal':m.get('temporal_recall'),'n_temporal':m.get('n_temporal'),
            'state':m.get('state_accuracy'),'n_state':m.get('n_state'),'status':r.get('status')})
    if rows:
        with (run/'resource_curve.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    points=[r for r in rows if r['complete'] and r['mean_seconds_all_attempts'] is not None]
    if not points or obj['device']!='cuda':return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for r in points:
        label=r['mode']+f"; errors={r['errors']}/4"
        for ax,x in [(axes[0],r['mean_seconds_all_attempts']),(axes[1],r['VRAM_reserved_MiB'])]:
            ax.scatter(x,r['F1'],s=65);ax.annotate(label,(x,r['F1']),xytext=(5,8),textcoords='offset points',fontsize=8)
    axes[0].set_xlabel('Pipeline seconds/call (all attempts)')
    axes[1].set_xlabel('Peak GPU reserved memory (MiB)')
    for ax in axes:ax.set_ylabel('Fact F1');ax.set_ylim(-.03,1.08);ax.grid(alpha=.25)
    fig.suptitle(f'Qwen +500 / tariff-catalog-v1: {len(points)} measured modes\nN=4 calls, one pass; diagnostic only')
    fig.savefig(run/'resource_curve.png',dpi=180);fig.savefig(run/'resource_curve.svg');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();main(a.run)
