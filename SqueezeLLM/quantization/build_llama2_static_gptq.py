"""Export static G128 GPTQ using local Pile calibration, without test evaluation."""
import sys, random, json
from pathlib import Path
from types import SimpleNamespace
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
sys.path.insert(0, '/home/zch/project/gptq_zch0722')
import llamazch as implementation

output=Path(sys.argv[1])
if output.exists(): raise FileExistsError(output)
model_path='/mnt/dfq/pretrained_model/llama-2-7b'
args=SimpleNamespace(nsamples=128,wbits=4,groupsize=128,sym=False,true_sequential=True,
                     percdamp=.01,act_order=True,static_groups=True)
implementation.args=args
torch.set_num_threads(4)
random.seed(0);torch.manual_seed(0)
model=implementation.get_llama(model_path).eval()
tokenizer=AutoTokenizer.from_pretrained(model_path,use_fast=False)
data=load_dataset('/mnt/dfq/datasets/mit-han-lab/pile-val-backup',split='validation').shuffle(seed=0)
loader=[]; intervals=[]
for index,record in enumerate(data):
 text=record['text'].strip()
 if not text:continue
 ids=tokenizer(text,return_tensors='pt').input_ids
 if ids.shape[1]<=2048:continue
 start=random.randint(0,ids.shape[1]-2048)
 block=ids[:,start:start+2048];loader.append((block,block.clone()));intervals.append([index,start])
 if len(loader)==128:break
if len(loader)!=128:raise ValueError('Insufficient calibration samples')
output.with_suffix('.calibration.json').write_text(json.dumps(dict(model=model_path,seed=0,parameters=vars(args),dataset='local Pile validation',shuffled_document_and_start=intervals),indent=2))
quantizers=implementation.llama_sequential(model,loader,torch.device('cuda:0'))
# Preserve the native quantization output before validation for diagnostics.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_group_centers import validate_record
modules=dict(model.named_modules())
records={}
for name,q in quantizers.items():
 records[name]=dict(weight=modules[name].weight.detach().cpu().contiguous(),
     scale=q.scale.detach().cpu(),zero=q.zero.detach().cpu(),maxq=int(q.maxq),
     group_size=128,static_groups=True,g_idx=q.gptq_g_idx.detach().cpu())
temporary=str(output)+'.incomplete'
torch.save(records,temporary)
errors={}
for name,record in records.items():
 errors[name]=validate_record(record,record['weight'].shape)
output.with_suffix('.validation.json').write_text(json.dumps(dict(
    rule='FP16-rounded reconstruction checked per element using dtype-relative tolerance',
    max_abs_error=max(errors.values()),modules=errors),indent=2))
import os
os.replace(temporary,output)
print('Validated export complete:',output,flush=True)
