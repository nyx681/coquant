"""Per-weight delta ablations, retaining original column RMS denominator."""
import numpy as np
from nuq import normalize_score_row

def importance(w,q,scope):
    w=np.asarray(w,dtype=np.float32);q=np.asarray(q,dtype=np.float32)
    if w.shape!=q.shape or w.ndim!=2:raise ValueError('Invalid G128 shape')
    eps=np.finfo(np.float32).eps
    denom=np.maximum(np.sqrt(np.mean(np.square(w),axis=0,dtype=np.float64)),eps)
    r=np.abs(w-q)/denom[None,:]
    if scope=='module':
        pos=r[r>0]
        if not len(pos):return np.ones_like(w)
        tau=float(np.median(pos))
        d=np.asarray(.1+.9/(1+r/(tau+eps)),dtype=np.float32)
        s=normalize_score_row(d)
    elif scope=='row':
        groups=r.reshape(w.shape);s=np.empty_like(w).reshape(w.shape)
        for i,row in enumerate(groups):
            pos=row[row>0]
            if not len(pos):s[i]=1;continue
            d=np.asarray(.1+.9/(1+row/(float(np.median(pos))+eps)),dtype=np.float32)
            s[i]=normalize_score_row(d)
        s=s.reshape(w.shape)
    else:raise ValueError(scope)
    if not np.isfinite(s).all() or (s<=0).any():raise ValueError('Invalid importance')
    return s

import torch
from nuq import kmeans_fit
from group_center_selection import assign

def quantize_row(task):
    w,s=task
    c,_=kmeans_fit((w[:,None],s,16))
    _,weighted,ordinary=assign(w,s,c)
    return c.reshape(1,16).astype(np.float32),np.array([weighted,ordinary],np.float64)

def fill_weight(weight,original,centers):
    if centers.shape!=(len(original),1,16):raise ValueError('Whole-row center shape mismatch')
    with torch.no_grad():
        for start in range(0,len(original),16):
            x=original[start:start+16].astype(np.float64)
            c=centers[start:start+16,0]
            labels=np.abs(x[:,:,None]-c.astype(np.float64)[:,None,:]).argmin(-1)
            weight[start:start+16].copy_(torch.from_numpy(np.take_along_axis(c,labels,axis=1)))
