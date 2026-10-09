"""Absolute per-weight delta with row median; no activation or AWQ scale."""
import numpy as np
from nuq import normalize_score_row
DESIGNS=('uniform','absolute_floor05','absolute_floor08')
def importance(w,q,scope,design):
    if scope!='row' or design not in DESIGNS:raise ValueError((scope,design))
    w=np.asarray(w,np.float32);q=np.asarray(q,np.float32)
    if w.shape!=q.shape or w.ndim!=2:raise ValueError('Shape mismatch')
    if design=='uniform':return np.ones_like(w)
    eta={'absolute_floor05':.5,'absolute_floor08':.8}[design]
    delta=np.abs(w-q);s=np.empty_like(w);eps=np.finfo(np.float32).eps
    for i,row in enumerate(delta):
        pos=row[row>0]
        if not len(pos):s[i]=1;continue
        d=eta+(1-eta)/(1+row/(float(np.median(pos))+eps))
        s[i]=normalize_score_row(d)
    if not np.isfinite(s).all() or (s<=0).any():raise ValueError('Invalid importance')
    return s
