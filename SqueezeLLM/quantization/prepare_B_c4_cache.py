import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib
from pathlib import Path
import torch
from squeezellm.datautils import get_awq_ppl_tokens
cfg=json.loads(Path(sys.argv[1]).read_text());r=Path(sys.argv[2]);r.mkdir(parents=True,exist_ok=True)
p=r/'c4_tokens.pt'
if p.exists():
 cached=torch.load(p,map_location='cpu');assert cached['model']==cfg['model'];tokens=cached['tokens'];path=cached['dataset_path']
else:
 tokens,path=get_awq_ppl_tokens('c4',cfg['model'],c4_num_proc=2)
 tmp=r/'c4_tokens.pt.tmp';torch.save(dict(model=cfg['model'],tokens=tokens,dataset_path=path),tmp);tmp.replace(p)
s=dict(dataset_path=path,tokens=tokens.numel(),evaluated_tokens=tokens.numel()//2048*2048,dropped_tokens=tokens.numel()%2048,blocks=tokens.numel()//2048,block_size=2048,batch_size=8,token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),tokenizer='use_fast=False',sampling='all documents in local C4 validation shard in stored order')
(r/'evaluation_protocol.json').write_text(json.dumps(s,indent=2));print(json.dumps(s),flush=True)
