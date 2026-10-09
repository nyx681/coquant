import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,json,sys,time
from pathlib import Path
import torch,numpy as np
from nuq import build_inverse_channel_relative_weights,normalize_score_row
from chunk_gptq_params import LLAMA_MODULES
from run_llama1_activation_fusion import digest_arrays
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
data=Path('/mnt/data/zch/quantization_experiments');qr=data/'extended_five_models_delta_only_20260921_090804/qwen3_8b_base'
configs=[dict(tag='llama1_7b',layers=32,original='/mnt/dfq/squeezellm_awq_delta/original_chunks',awq='/mnt/dfq/squeezellm_awq_delta/awq_dequant_chunks',params='/mnt/dfq/gptq_output/llama-7b-w4-g128-gptq_params-fixed.pt',endpoints=str(data/'llama1_delta_no_scale_no_offset_20260919_172224')),dict(json.loads((qr/'target.json').read_text()),tag='qwen3_8b_base',endpoints=str(qr/'endpoints'))]
summary=[];records=[]
header=['layer','module','input_channel','group128','channel_in_group','awq_delta_S','gptq_delta_S']
for cfg in configs:
 out=root/cfg['tag'];out.mkdir();(out/'modules').mkdir();arrays={};count=0
 params=torch.load(cfg['params'],map_location='cpu',mmap=True) if 'params' in cfg else None
 with (out/'coefficients.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(header)
  for l in range(cfg['layers']):
   orig=torch.load(Path(cfg['original'])/f'layer_{l}.pt',map_location='cpu');aq=torch.load(Path(cfg['awq'])/f'layer_{l}.pt',map_location='cpu')
   gq=torch.load(Path(cfg['gptq'])/f'layer_{l}.pt',map_location='cpu') if params is None else None
   for name,suffix in LLAMA_MODULES.items():
    key=f'{l}.{name}';w=orig[name].float().numpy();ss=[]
    for variant,q in [('awq_delta',aq[name]),('gptq_delta',gq[name] if params is None else params[f'model.layers.{l}.{suffix}']['weight'])]:
     s=normalize_score_row(build_inverse_channel_relative_weights(w,np.abs(w-q.float().numpy()),gamma=1,min_weight=.1))
     assert np.isfinite(s).all() and (s>0).all()
     with np.load(Path(cfg['endpoints'])/variant/'centers'/f'{key}.npz') as z:
      assert str(z['fingerprint'])==digest_arrays(w,s),(cfg['tag'],key,variant,'fingerprint mismatch')
     arrays[key+'.'+variant]=s;ss.append(s)
     summary.append(dict(model=cfg['tag'],layer=l,module=suffix,method=variant,channels=len(s),minimum=float(s.min()),median=float(np.median(s)),mean=float(s.mean()),maximum=float(s.max()),std=float(s.std()),fingerprint_verified=True))
    lines=[[l,suffix,j,j//128,j%128,format(float(ss[0][j]),'.9g'),format(float(ss[1][j]),'.9g')] for j in range(len(ss[0]))]
    writer.writerows(lines);count+=len(lines)
    with (out/'modules'/f'layer_{l:02d}_{name}.csv').open('w',newline='') as mf:
     mw=csv.writer(mf);mw.writerow(header);mw.writerows(lines)
    del w,q,s,ss,lines
   print(cfg['tag'],l+1,'/',cfg['layers'],'verified',flush=True)
   del orig,aq,gq
 np.savez_compressed(out/'coefficients.npz',**arrays)
 records.append(dict(model=cfg['tag'],rows=count,modules=cfg['layers']*7,config=cfg));del params,arrays
with (root/'module_summary.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
(root/'manifest.json').write_text(json.dumps(dict(status='complete',bit=4,group_size=128,formula='S=N(d)',verification='Every digest_arrays(W,S) matches original endpoint checkpoint fingerprint',models=records,finished_at=time.strftime('%F %T')),ensure_ascii=False,indent=2))
(root/'README.md').write_text('''# 实际delta重要性系数 S_j=N(d_j)\n\n当前 W4-G128 去 scale、去外层1+版本。每个Linear每个输入通道一个系数，同列所有输出行共享。编号从0开始。group128用于定位；归一化范围是整个Linear的输入通道，不是每个group。\n\ncoefficients.csv：全模型两种方法系数。modules/：逐模块CSV。coefficients.npz：无损float32数组，键为层号.模块简称.方法。module_summary.csv：模块统计。CSV保留9位有效数字。\n\n调用实验原函数（gamma=1,min_weight=0.1），每一个模块、每一种方法均验证digest_arrays(W,S)与原实验中心检查点中的fingerprint完全一致。没有重新量化或评测。\n''')
print('COMPLETE',root,flush=True)
