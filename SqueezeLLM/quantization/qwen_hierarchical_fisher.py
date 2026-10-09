"""G2048 codebook with one learned positive scale per G128 subgroup."""
import numpy as np
import torch
from nuq import kmeans_fit
from qwen_self_delta_g2048 import importance
GROUP_SIZE=2048
SMALL_GROUP_SIZE=128
DESIGNS=('self_floor05','self_floor08','fisher_only')

def fit_group(x,a,return_history=False):
    x=np.asarray(x,np.float64).reshape(16,128)
    a=np.asarray(a,np.float64).reshape(16,128)
    if not np.isfinite(x).all() or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('Invalid samples/importance')
    if not np.any(a):a=np.ones_like(a)  # zero-Fisher group: ordinary reconstruction tie-break
    scale=np.sqrt(np.mean(x*x,axis=1))
    scale[scale==0]=1
    if np.all(x==0):
        c=np.zeros(16,np.float32)
        return (c,scale.astype(np.float32),[0.]) if return_history else (c,scale.astype(np.float32))
    u=x/scale[:,None]
    weights=a*scale[:,None]**2
    c,_=kmeans_fit((u.reshape(-1,1).astype(np.float32),(weights/weights.max()).ravel().astype(np.float32),16))
    c=np.asarray(c,np.float64).reshape(16)
    def assignment():
        return np.abs(x[:,:,None]-scale[:,None,None]*c[None,None,:]).argmin(-1)
    z=assignment();history=[float(np.sum(a*(x-scale[:,None]*c[z])**2))]
    for _ in range(30):
        num=np.bincount(z.ravel(),weights=(a*scale[:,None]*x).ravel(),minlength=16)
        den=np.bincount(z.ravel(),weights=(a*scale[:,None]**2).ravel(),minlength=16)
        np.divide(num,den,out=c,where=den>0)
        v=c[z];num=np.sum(a*v*x,axis=1);den=np.sum(a*v*v,axis=1)
        proposed=scale.copy();np.divide(num,den,out=proposed,where=den>0)
        scale=np.maximum(proposed,1e-20)
        z=assignment();loss=float(np.sum(a*(x-scale[:,None]*c[z])**2))
        if loss>history[-1]+1e-10*max(1.,history[-1]):raise ValueError('Objective increased')
        history.append(loss)
        if history[-2]-loss<=1e-6*max(history[-2],1e-30):break
    # Fix scale/codebook ambiguity without changing reconstructed weights.
    norm=np.max(np.abs(c))
    if norm>0:c/=norm;scale*=norm
    result=(c.astype(np.float32),scale.astype(np.float32))
    return (*result,history) if return_history else result

def quantize_row(task):
    w,s=task
    if len(w)%GROUP_SIZE or w.shape!=s.shape:raise ValueError('Group alignment')
    books=[];scales=[];cost=np.zeros(2,np.float64)
    for start in range(0,len(w),GROUP_SIZE):
        x=w[start:start+GROUP_SIZE].astype(np.float64).reshape(16,128)
        a=s[start:start+GROUP_SIZE].astype(np.float64).reshape(16,128)
        c,t=fit_group(x,a);levels=t.astype(np.float64)[:,None]*c.astype(np.float64)[None,:]
        z=np.abs(x[:,:,None]-levels[:,None,:]).argmin(-1)
        y=np.take_along_axis(levels,z,axis=1);error=(x-y)**2
        books.append(c);scales.append(t);cost+=np.array([np.sum(a*error),np.sum(error)])
    return np.stack(books),np.stack(scales),cost

def fill_weight(weight,original,centers,scales):
    rows,cols=original.shape
    shape=(rows,cols//GROUP_SIZE,16)
    if cols%GROUP_SIZE or centers.shape!=shape or scales.shape!=shape:raise ValueError('Invalid codebooks/scales')
    with torch.no_grad():
        for start in range(0,rows,4):
            x=original[start:start+4].astype(np.float64).reshape(-1,cols//GROUP_SIZE,16,128)
            levels=scales[start:start+4].astype(np.float64)[:,:,:,None]*centers[start:start+4].astype(np.float64)[:,:,None,:]
            z=np.abs(x[:,:,:,:,None]-levels[:,:,:,None,:]).argmin(-1)
            y=np.take_along_axis(levels,z,axis=3).reshape(-1,cols)
            weight[start:start+4].copy_(torch.from_numpy(y))
