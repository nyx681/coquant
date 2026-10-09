"""Evaluate existing complete B fusion centers without repeating CPU fusion."""
import os,json,time,csv,sys
from pathlib import Path
from run_source_scale_worst_out_centers_ready import evaluate,write_json
r=Path(sys.argv[1]);cfg=json.loads((r/'target.json').read_text());s=json.loads((r/'run.json').read_text())
assert s['completed_modules']==s['total_modules']==cfg['layers']*(6 if cfg['model_type']=='opt' else 7)
assert len(list((r/'centers').glob('*.npz')))==s['total_modules']
s.update(pid=os.getpid(),evaluation_restart_reason='Independent Wiki GPU lane; reuse all completed fusion centers')
try:
 result=evaluate(r,cfg,s,'/mnt/data/zch/quantization_experiments/B_wiki_evaluation.gpu.lock')
 baseline=r.parent/'fusion_baselines.csv'
 with (r/'comparison.csv').open('w') as fp:
  w=csv.writer(fp);w.writerow(['method','WikiText2_PPL','quantized_weight_bits','source'])
  if baseline.exists():
   with baseline.open() as source:
    for row in csv.DictReader(source):w.writerow([row['method'],row['FP16_parameters_ppl'],row['actual_quantized_weight_bits'],row['source']])
  w.writerow(['source_scale_worst_out_fusion',result['ppl'],result['actual_quantized_weight_bits'],str(r/'ppl.json')])
 s.update(status='complete',finished_at=time.strftime('%F %T'),ppl=result['ppl']);write_json(r/'run.json',s)
except BaseException as e:
 s.update(status='failed',error=repr(e));write_json(r/'run.json',s);raise
