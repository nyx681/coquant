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
        c=np.zeros(8,np.float32)
        return (c,scale.astype(np.float32),[0.]) if return_history else (c,scale.astype(np.float32))
    u=x/scale[:,None]
    weights=a*scale[:,None]**2
    c,_=kmeans_fit((u.reshape(-1,1).astype(np.float32),(weights/weights.max()).ravel().astype(np.float32),8))
    c=np.asarray(c,np.float64).reshape(8)
    def assignment():
        return np.abs(x[:,:,None]-scale[:,None,None]*c[None,None,:]).argmin(-1)
    z=assignment();history=[float(np.sum(a*(x-scale[:,None]*c[z])**2))]
    for _ in range(30):
        num=np.bincount(z.ravel(),weights=(a*scale[:,None]*x).ravel(),minlength=8)
        den=np.bincount(z.ravel(),weights=(a*scale[:,None]**2).ravel(),minlength=8)
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
