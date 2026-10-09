"""G2048 independent codebooks; bounded per-element relative AWQ error."""
import numpy as np
import torch
from nuq import kmeans_fit,normalize_score_row
from group_center_selection import assign
GROUP_SIZE=2048
DESIGNS=('self_floor05','self_floor08','uniform','absolute_inverse_control')
def importance(w,q,scope,design):
    w=np.asarray(w,np.float32);q=np.asarray(q,np.float32)
    if design not in DESIGNS or w.shape!=q.shape or w.ndim!=2:raise ValueError('Invalid input')
    if not np.isfinite(w).all() or not np.isfinite(q).all():raise ValueError('Nonfinite input')
    if design=='uniform':return np.ones_like(w)
    delta=np.abs(w-q);eps=np.finfo(np.float32).eps
    if design.startswith('self_'):
        eta={'self_floor05':.5,'self_floor08':.8}[design]
        s=eta+(1-eta)*(np.abs(w)+eps)/(np.abs(w)+delta+eps)
        if not np.isfinite(s).all() or np.any(s<eta) or np.any(s>1):raise ValueError('Invalid score')
        return s.astype(np.float32)
    # Historical best absolute-delta formula, now evaluated at the same G2048.
    s=np.empty_like(w)
    for i,row in enumerate(delta):
        pos=row[row>0]
        s[i]=normalize_score_row(.1+.9/(1+row/(float(np.median(pos))+eps))) if len(pos) else 1
    return s

def quantize_row(task):
    w,s=task
    if len(w)%GROUP_SIZE or w.shape!=s.shape:raise ValueError('G2048 alignment')
    books=[];cost=np.zeros(2,np.float64)
    for start in range(0,len(w),GROUP_SIZE):
        x=w[start:start+GROUP_SIZE];a=s[start:start+GROUP_SIZE]
        c,_=kmeans_fit((x[:,None],a,16));_,weighted,ordinary=assign(x,a,c)
        books.append(np.asarray(c,np.float32).reshape(16));cost+=np.array([weighted,ordinary])
    return np.stack(books),cost

def fill_weight(weight,original,centers):
    rows,cols=original.shape
    if cols%GROUP_SIZE or centers.shape!=(rows,cols//GROUP_SIZE,16):raise ValueError('Invalid centers')
    with torch.no_grad():
        for start in range(0,rows,8):
            x=original[start:start+8].astype(np.float64).reshape(-1,cols//GROUP_SIZE,GROUP_SIZE)
            c=centers[start:start+8].astype(np.float64)
            z=np.abs(x[:,:,:,None]-c[:,:,None,:]).argmin(-1)
            y=np.take_along_axis(c,z,axis=2).reshape(-1,cols)
            weight[start:start+8].copy_(torch.from_numpy(y))
