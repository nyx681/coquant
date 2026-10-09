"""Per-weight delta ablations, retaining original column RMS denominator."""
import numpy as np
from nuq import normalize_score_row

def importance(w,q,scope):
    w=np.asarray(w,dtype=np.float32);q=np.asarray(q,dtype=np.float32)
    if w.shape!=q.shape or w.ndim!=2 or w.shape[1]%128:raise ValueError('Invalid G128 shape')
    eps=np.finfo(np.float32).eps
    denom=np.maximum(np.sqrt(np.mean(np.square(w),axis=0,dtype=np.float64)),eps)
    r=np.abs(w-q)/denom[None,:]
    if scope=='module':
        pos=r[r>0]
        if not len(pos):return np.ones_like(w)
        tau=float(np.median(pos))
        d=np.asarray(.1+.9/(1+r/(tau+eps)),dtype=np.float32)
        s=normalize_score_row(d)
    elif scope=='group':
        groups=r.reshape(-1,128);s=np.empty_like(w).reshape(-1,128)
        for i,row in enumerate(groups):
            pos=row[row>0]
            if not len(pos):s[i]=1;continue
            d=np.asarray(.1+.9/(1+row/(float(np.median(pos))+eps)),dtype=np.float32)
            s[i]=normalize_score_row(d)
        s=s.reshape(w.shape)
    else:raise ValueError(scope)
    if not np.isfinite(s).all() or (s<=0).any():raise ValueError('Invalid importance')
    return s
