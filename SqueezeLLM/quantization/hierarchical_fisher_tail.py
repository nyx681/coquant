"""B quantizer with final short G2048 group, retaining G128 scales."""
import numpy as np
import torch
from qwen_hierarchical_fisher import fit_group,importance,DESIGNS,GROUP_SIZE

def quantize_row(task):
 w,s=task
 if len(w)%128 or w.shape!=s.shape:raise ValueError('G128 alignment')
 books=[];scales=[];cost=np.zeros(2,np.float64)
 for start in range(0,len(w),2048):
  n=min(2048,len(w)-start);x=w[start:start+n].astype(np.float64);a=s[start:start+n].astype(np.float64)
  xp=np.pad(x,(0,2048-n));ap=np.pad(a,(0,2048-n))
  c,t=fit_group(xp,ap);levels=t[:n//128].astype(np.float64)[:,None]*c.astype(np.float64)[None,:]
  z=np.abs(x.reshape(-1,128)[:,:,None]-levels[:,None,:]).argmin(-1);y=np.take_along_axis(levels,z,axis=1).ravel();e=(x-y)**2
  cost+=np.array([np.sum(a*e),np.sum(e)]);books.append(c);scales.append(t)
 return np.stack(books),np.stack(scales),cost

def fill_weight(weight,original,centers,scales):
 rows,cols=original.shape;groups=(cols+2047)//2048
 if cols%128 or centers.shape!=(rows,groups,16) or scales.shape!=centers.shape:raise ValueError('Invalid shape')
 with torch.no_grad():
  for start in range(0,rows,4):
   for g in range(groups):
    lo=g*2048;hi=min(cols,lo+2048);ng=(hi-lo)//128
    x=original[start:start+4,lo:hi].astype(np.float64).reshape(-1,ng,128)
    levels=scales[start:start+4,g,:ng].astype(np.float64)[:,:,None]*centers[start:start+4,g].astype(np.float64)[:,None,:]
    z=np.abs(x[:,:,:,None]-levels[:,:,None,:]).argmin(-1);y=np.take_along_axis(levels,z,axis=2).reshape(-1,hi-lo)
    weight[start:start+4,lo:hi].copy_(torch.from_numpy(y))
